"""
orchestrator.py – LangGraph Multi-Agent StateGraph
────────────────────────────────────────────────────
Defines a directed StateGraph where each node corresponds to one micro-agent.
The graph is compiled into a runnable chain callable as:

    from orchestrator import run_grid_pipeline
    result = run_grid_pipeline(GridPipelineInput(...))

Execution sequence (single linear pass, suitable for real-time polling):

    [solar_node] → [demand_node] → [battery_node] → [outage_node] → [dashboard_node]

State is threaded through a shared TypedDict (GridState) so every downstream
agent has access to all upstream results.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.battery_agent import BatteryDecision, BatteryInput, run_battery_agent
from agents.demand_agent  import BalancingResult, DemandInput,  run_demand_agent
from agents.outage_agent  import OutageInput, OutageRiskResult, run_outage_agent
from agents.solar_agent   import SolarForecastResult, SolarInput, run_solar_agent
from config import get_settings
from utils.logger import get_logger

logger   = get_logger("orchestrator")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Input Model
# ──────────────────────────────────────────────────────────────────────────────

class GridPipelineInput(TypedDict, total=False):
    """Flat dict of all sensor readings fed into the pipeline."""
    # Solar inputs
    irradiance_wm2:      float
    ambient_temp_c:      float
    panel_capacity_mw:   float
    hour_of_day:         int
    cloud_cover_pct:     float

    # Demand inputs
    realtime_demand_mw:  float
    thermal_capacity_mw: float
    load_profile:        str        # "residential" | "industrial" | "mixed"

    # Battery inputs
    battery_soc_pct:     float
    battery_capacity_mwh: float

    # Outage inputs
    transformer_temp_c:  float
    voltage_drop_pct:    float
    line_loading_pct:    float
    weather_alert:       int        # 0–3
    asset_id:            str


# ──────────────────────────────────────────────────────────────────────────────
# Shared State (threaded through every graph node)
# ──────────────────────────────────────────────────────────────────────────────

class GridState(TypedDict, total=False):
    """Mutable state object passed through all LangGraph nodes."""
    pipeline_input:   GridPipelineInput

    # Agent outputs (populated as pipeline progresses)
    solar_result:    Optional[SolarForecastResult]
    demand_result:   Optional[BalancingResult]
    battery_result:  Optional[BatteryDecision]
    outage_result:   Optional[OutageRiskResult]

    # Aggregated dashboard payload
    dashboard_payload: Optional[dict]

    # Error tracking
    errors: list[str]
    started_at: str
    completed_at: str


# ──────────────────────────────────────────────────────────────────────────────
# Default sensor values (safe fallback when a field is not provided)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: GridPipelineInput = {
    "irradiance_wm2":       700.0,
    "ambient_temp_c":       35.0,
    "panel_capacity_mw":    500.0,
    "hour_of_day":          12,
    "cloud_cover_pct":      20.0,
    "realtime_demand_mw":   420.0,
    "thermal_capacity_mw":  200.0,
    "load_profile":         "mixed",
    "battery_soc_pct":      60.0,
    "battery_capacity_mwh": 500.0,
    "transformer_temp_c":   75.0,
    "voltage_drop_pct":     3.5,
    "line_loading_pct":     72.0,
    "weather_alert":        0,
    "asset_id":             "GRID-TX-01",
}


def _merge_defaults(inp: GridPipelineInput) -> GridPipelineInput:
    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in inp.items() if v is not None})
    return merged  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────────────────────────────────────

def solar_node(state: GridState) -> GridState:
    """Node 1: Solar Generation Forecasting Agent."""
    inp = state["pipeline_input"]
    logger.info("orchestrator | solar_node")
    try:
        solar_inp = SolarInput(
            irradiance_wm2   = inp["irradiance_wm2"],
            ambient_temp_c   = inp["ambient_temp_c"],
            panel_capacity_mw= inp["panel_capacity_mw"],
            hour_of_day      = inp["hour_of_day"],
            cloud_cover_pct  = inp["cloud_cover_pct"],
        )
        state["solar_result"] = run_solar_agent(solar_inp)
    except Exception as exc:
        logger.error("solar_node failed", error=str(exc))
        state.setdefault("errors", []).append(f"solar_node: {exc}")
    return state


def demand_node(state: GridState) -> GridState:
    """Node 2: Demand-Supply Balancing Agent."""
    inp          = state["pipeline_input"]
    solar_result = state.get("solar_result")
    solar_mw     = solar_result.current_output_mw if solar_result else 0.0

    # Battery dispatchable power (10% of capacity at current SoC as quick estimate)
    batt_available = inp["battery_capacity_mwh"] * (inp["battery_soc_pct"] / 100) * 0.5

    logger.info("orchestrator | demand_node", solar_mw=solar_mw)
    try:
        demand_inp = DemandInput(
            realtime_demand_mw  = inp["realtime_demand_mw"],
            solar_output_mw     = solar_mw,
            battery_available_mw= batt_available,
            thermal_capacity_mw = inp["thermal_capacity_mw"],
            hour_of_day         = inp["hour_of_day"],
            load_profile        = inp["load_profile"],
        )
        state["demand_result"] = run_demand_agent(demand_inp)
    except Exception as exc:
        logger.error("demand_node failed", error=str(exc))
        state.setdefault("errors", []).append(f"demand_node: {exc}")
    return state


def battery_node(state: GridState) -> GridState:
    """Node 3: Battery Storage Optimization Agent."""
    inp           = state["pipeline_input"]
    demand_result = state.get("demand_result")
    gap_mw        = demand_result.supply_demand_gap_mw if demand_result else 0.0

    logger.info("orchestrator | battery_node", gap_mw=gap_mw)
    try:
        batt_inp = BatteryInput(
            current_soc_pct       = inp["battery_soc_pct"],
            capacity_mwh          = inp["battery_capacity_mwh"],
            supply_demand_gap_mw  = gap_mw,
            hour_of_day           = inp["hour_of_day"],
        )
        state["battery_result"] = run_battery_agent(batt_inp)
    except Exception as exc:
        logger.error("battery_node failed", error=str(exc))
        state.setdefault("errors", []).append(f"battery_node: {exc}")
    return state


def outage_node(state: GridState) -> GridState:
    """Node 4: Outage Prediction Agent."""
    inp = state["pipeline_input"]
    logger.info("orchestrator | outage_node", asset=inp.get("asset_id"))
    try:
        outage_inp = OutageInput(
            transformer_temp_c = inp["transformer_temp_c"],
            voltage_drop_pct   = inp["voltage_drop_pct"],
            line_loading_pct   = inp["line_loading_pct"],
            weather_alert      = inp["weather_alert"],
            ambient_temp_c     = inp["ambient_temp_c"],
            asset_id           = inp["asset_id"],
        )
        state["outage_result"] = run_outage_agent(outage_inp)
    except Exception as exc:
        logger.error("outage_node failed", error=str(exc))
        state.setdefault("errors", []).append(f"outage_node: {exc}")
    return state


def dashboard_node(state: GridState) -> GridState:
    """
    Node 5: Grid Performance Dashboard Agent.
    Aggregates all agent outputs into a single REST-ready payload.
    """
    logger.info("orchestrator | dashboard_node – aggregating results")

    solar   = state.get("solar_result")
    demand  = state.get("demand_result")
    battery = state.get("battery_result")
    outage  = state.get("outage_result")

    # Helper: safely serialise dataclass / enum fields
    def _safe(obj: Any) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: _safe(v) for k, v in asdict(obj).items()
            }
        if isinstance(obj, list):
            return [_safe(i) for i in obj]
        if hasattr(obj, "value"):          # Enum
            return obj.value
        return obj

    state["dashboard_payload"] = {
        "meta": {
            "platform":     "Smart Energy Management – Gujarat Grid",
            "version":      "1.0.0",
            "pipeline_run": state.get("started_at", ""),
            "errors":       state.get("errors", []),
        },
        "solar_forecast":  _safe(solar),
        "demand_balance":  _safe(demand),
        "battery_status":  _safe(battery),
        "outage_risk":     _safe(outage),
        # Convenience KPI surface (used directly by Streamlit gauges)
        "kpis": {
            "solar_output_mw":      solar.current_output_mw      if solar   else None,
            "demand_mw":            demand.realtime_demand_mw     if demand  else None,
            "supply_mw":            demand.total_supply_mw        if demand  else None,
            "gap_mw":               demand.supply_demand_gap_mw   if demand  else None,
            "grid_status":          demand.grid_status.value      if demand  else "UNKNOWN",
            "battery_soc_pct":      state["pipeline_input"]["battery_soc_pct"],
            "battery_action":       battery.action.value          if battery else "HOLD",
            "battery_power_mw":     battery.power_setpoint_mw     if battery else 0.0,
            "outage_risk_score":    outage.risk_score             if outage  else 0.0,
            "outage_risk_level":    outage.risk_level.value       if outage  else "NORMAL",
            "solar_anomaly":        solar.anomaly_detected        if solar   else False,
            "confidence_pct":       solar.confidence_pct          if solar   else 0.0,
        },
    }

    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Build & Compile LangGraph StateGraph
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Construct and compile the agent pipeline graph."""
    graph = StateGraph(GridState)

    # Register nodes
    graph.add_node("solar",    solar_node)
    graph.add_node("demand",   demand_node)
    graph.add_node("battery",  battery_node)
    graph.add_node("outage",   outage_node)
    graph.add_node("dashboard",dashboard_node)

    # Sequential edges
    graph.set_entry_point("solar")
    graph.add_edge("solar",    "demand")
    graph.add_edge("demand",   "battery")
    graph.add_edge("battery",  "outage")
    graph.add_edge("outage",   "dashboard")
    graph.add_edge("dashboard", END)

    return graph.compile()


# Compiled graph singleton (built once on module import)
_PIPELINE = _build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run_grid_pipeline(raw_input: dict | None = None) -> dict:
    """
    Execute the full 5-node agent pipeline.

    Args:
        raw_input: Dict matching GridPipelineInput fields. Missing keys are
                   filled with safe Gujarat-grid defaults.

    Returns:
        The completed dashboard_payload dict (REST-ready JSON).
    """
    inp = _merge_defaults(raw_input or {})

    initial_state: GridState = {
        "pipeline_input":  inp,
        "solar_result":    None,
        "demand_result":   None,
        "battery_result":  None,
        "outage_result":   None,
        "dashboard_payload": None,
        "errors":          [],
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "completed_at":    "",
    }

    logger.info("orchestrator | pipeline starting",
                hour=inp["hour_of_day"], demand=inp["realtime_demand_mw"])

    final_state = _PIPELINE.invoke(initial_state)

    logger.info("orchestrator | pipeline complete",
                errors=len(final_state.get("errors", [])))

    return final_state.get("dashboard_payload") or {}
