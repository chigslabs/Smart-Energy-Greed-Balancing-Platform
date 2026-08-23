"""
agents/demand_agent.py – Demand-Supply Balancing Agent
───────────────────────────────────────────────────────
Inputs  : real-time consumer demand (MW), solar output forecast (MW),
          historical load profile key (residential / industrial / mixed)
Outputs : supply-demand gap (MW), dispatch commands list,
          grid status label, and IBM Granite reasoning.

ReAct pattern:
  1. THINK  – classify load profile and compute gap
  2. ACT    – build ranked dispatch command set
  3. REASON – Granite LLM refines commands into operator narrative
  4. RETURN – structured BalancingResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import (
    DEFAULT_GEN_PARAMS,
    GRANITE_8B,
    GUJARAT_PEAK_HOURS,
    get_settings,
)
from utils.llm_client import get_llm_response
from utils.logger import get_logger

logger = get_logger("demand_agent")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# Gujarat historical load profiles (normalised MW per 1000 MW installed base)
# Indexed by hour (0–23 IST)
# ──────────────────────────────────────────────────────────────────────────────

_LOAD_PROFILES: dict[str, list[float]] = {
    "residential": [
        0.45, 0.40, 0.38, 0.37, 0.38, 0.42,  # 00–05
        0.52, 0.68, 0.75, 0.72, 0.68, 0.65,  # 06–11
        0.62, 0.60, 0.58, 0.60, 0.65, 0.85,  # 12–17
        0.95, 1.00, 0.98, 0.90, 0.75, 0.55,  # 18–23
    ],
    "industrial": [
        0.70, 0.70, 0.70, 0.70, 0.72, 0.75,  # 00–05
        0.85, 0.95, 1.00, 1.00, 1.00, 0.98,  # 06–11
        0.97, 0.98, 1.00, 0.98, 0.95, 0.88,  # 12–17
        0.80, 0.72, 0.68, 0.68, 0.70, 0.70,  # 18–23
    ],
    "mixed": [
        0.58, 0.55, 0.54, 0.54, 0.55, 0.58,  # 00–05
        0.68, 0.82, 0.88, 0.86, 0.84, 0.82,  # 06–11
        0.80, 0.79, 0.79, 0.79, 0.80, 0.87,  # 12–17
        0.88, 0.86, 0.83, 0.79, 0.73, 0.62,  # 18–23
    ],
}


class GridStatus(str, Enum):
    SURPLUS    = "SURPLUS"      # supply > demand by > 5%
    BALANCED   = "BALANCED"     # within ±5%
    DEFICIT    = "DEFICIT"      # demand > supply by > 5%
    CRITICAL   = "CRITICAL"     # demand > supply by > 20%


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DemandInput:
    """Input bundle for the Demand-Supply Balancing Agent."""
    realtime_demand_mw: float          # Measured consumer demand
    solar_output_mw: float             # Current solar generation
    battery_available_mw: float        # Dispatchable battery power
    thermal_capacity_mw: float         # Available thermal headroom
    hour_of_day: int                   # IST hour (0–23)
    load_profile: str = "mixed"        # "residential" | "industrial" | "mixed"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class DispatchCommand:
    """Single actionable dispatch command."""
    priority: int           # 1 = highest
    source: str             # "Solar" | "Battery" | "Thermal" | "Curtail"
    action: str             # Human-readable command string
    mw_adjustment: float    # Positive = increase output, Negative = curtail
    reason: str


@dataclass
class BalancingResult:
    """Output payload from the Demand-Supply Balancing Agent."""
    realtime_demand_mw: float
    total_supply_mw: float
    supply_demand_gap_mw: float         # Positive = surplus, Negative = deficit
    grid_status: GridStatus
    is_peak_hour: bool
    dispatch_commands: list[DispatchCommand]
    agent_reasoning: str
    recommended_action_summary: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gap analysis & dispatch logic
# ──────────────────────────────────────────────────────────────────────────────

def _classify_grid_status(gap_mw: float, demand_mw: float) -> GridStatus:
    pct = (gap_mw / demand_mw) * 100 if demand_mw > 0 else 0
    if pct >= 5:
        return GridStatus.SURPLUS
    if pct <= -20:
        return GridStatus.CRITICAL
    if pct <= -5:
        return GridStatus.DEFICIT
    return GridStatus.BALANCED


def _build_dispatch_commands(
    gap_mw: float,
    inp: DemandInput,
    status: GridStatus,
    is_peak: bool,
) -> list[DispatchCommand]:
    """
    Rule-based dispatch command builder ranked by merit order:
      1. Solar (zero marginal cost)
      2. Battery (fast response)
      3. Thermal (slow, expensive)
      4. Curtailment (last resort)
    """
    commands: list[DispatchCommand] = []
    remaining_gap = abs(gap_mw)

    if status in (GridStatus.DEFICIT, GridStatus.CRITICAL):
        # ── Deficit: need more supply ────────────────────────────────────────

        # Already using all solar; check if battery can cover
        battery_dispatch = min(remaining_gap, inp.battery_available_mw)
        if battery_dispatch > 0:
            commands.append(DispatchCommand(
                priority=1,
                source="Battery",
                action=f"Discharge {battery_dispatch:.1f} MW from battery storage",
                mw_adjustment=battery_dispatch,
                reason=(
                    "Zero-emission fast-response resource; "
                    f"covering {battery_dispatch:.1f} MW of the {remaining_gap:.1f} MW deficit."
                ),
            ))
            remaining_gap -= battery_dispatch

        if remaining_gap > 0.5:
            thermal_needed = min(remaining_gap, inp.thermal_capacity_mw)
            commands.append(DispatchCommand(
                priority=2,
                source="Thermal",
                action=f"Ramp up thermal generation by {thermal_needed:.1f} MW",
                mw_adjustment=thermal_needed,
                reason=(
                    f"Battery insufficient; thermal fill required for {thermal_needed:.1f} MW. "
                    "Issue start-ramp command to thermal unit GRID-TH-01."
                ),
            ))
            remaining_gap -= thermal_needed

        if remaining_gap > 0.5 and status == GridStatus.CRITICAL:
            commands.append(DispatchCommand(
                priority=3,
                source="Demand Response",
                action=f"Initiate load shedding for non-critical industrial feeders ({remaining_gap:.1f} MW)",
                mw_adjustment=-remaining_gap,
                reason="Critical deficit. Controlled load shedding to prevent frequency collapse.",
            ))

    elif status == GridStatus.SURPLUS:
        surplus = abs(gap_mw)

        # First, charge battery with surplus
        battery_charge = min(surplus, inp.battery_available_mw)
        if battery_charge > 0:
            commands.append(DispatchCommand(
                priority=1,
                source="Battery",
                action=f"Charge battery at {battery_charge:.1f} MW from solar surplus",
                mw_adjustment=-battery_charge,
                reason="Absorb excess renewable generation to maximise storage for evening peak.",
            ))
            surplus -= battery_charge

        if surplus > 2.0:
            commands.append(DispatchCommand(
                priority=2,
                source="Solar",
                action=f"Curtail {surplus:.1f} MW solar (route to EV charging stations if available)",
                mw_adjustment=-surplus,
                reason="Battery at capacity; curtailment required to prevent over-frequency.",
            ))

        if is_peak and inp.thermal_capacity_mw > 0:
            commands.append(DispatchCommand(
                priority=3,
                source="Thermal",
                action="Ramp down thermal unit GRID-TH-01 to minimum stable load",
                mw_adjustment=-min(surplus + 10, inp.thermal_capacity_mw),
                reason="Displace thermal with cheap solar surplus during peak hours.",
            ))

    else:  # BALANCED
        commands.append(DispatchCommand(
            priority=1,
            source="Grid",
            action="Maintain current dispatch schedule – grid is balanced",
            mw_adjustment=0.0,
            reason=f"Gap within ±5% tolerance ({gap_mw:+.1f} MW). No action required.",
        ))

    return commands


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Reasoning
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    inp: DemandInput,
    gap_mw: float,
    status: GridStatus,
    commands: list[DispatchCommand],
) -> str:
    cmd_text = "\n".join(
        f"  [{c.priority}] {c.action} ({c.source}, {c.mw_adjustment:+.1f} MW)"
        for c in commands
    )
    return f"""You are a grid operations specialist for Gujarat's State Load Dispatch Centre (SLDC).

### Grid Snapshot
- Real-Time Demand    : {inp.realtime_demand_mw:.1f} MW
- Solar Generation    : {inp.solar_output_mw:.1f} MW
- Battery Available   : {inp.battery_available_mw:.1f} MW
- Thermal Headroom    : {inp.thermal_capacity_mw:.1f} MW
- Supply-Demand Gap   : {gap_mw:+.1f} MW  ({status.value})
- Hour (IST)          : {inp.hour_of_day:02d}:00
- Load Profile        : {inp.load_profile}

### Proposed Dispatch Commands
{cmd_text}

### Task
Validate the dispatch plan and provide a 2–3 sentence operator advisory covering:
1. Immediate action priority
2. Grid stability risk if action is delayed
3. Recommended monitoring metric for the next 30 minutes
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public Agent Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run_demand_agent(inp: DemandInput) -> BalancingResult:
    """
    Execute the Demand-Supply Balancing ReAct cycle.
    """
    logger.info("demand_agent | starting balance computation",
                demand=inp.realtime_demand_mw, solar=inp.solar_output_mw)

    # ── THINK: compute totals and gap ────────────────────────────────────────
    total_supply = inp.solar_output_mw + inp.battery_available_mw
    gap_mw       = total_supply - inp.realtime_demand_mw
    status       = _classify_grid_status(gap_mw, inp.realtime_demand_mw)
    is_peak      = inp.hour_of_day in GUJARAT_PEAK_HOURS

    # ── ACT: build dispatch commands ─────────────────────────────────────────
    commands = _build_dispatch_commands(gap_mw, inp, status, is_peak)

    # ── REASON: IBM Granite advisory ─────────────────────────────────────────
    prompt   = _build_prompt(inp, gap_mw, status, commands)
    reasoning = get_llm_response(
        prompt=prompt,
        model_id=GRANITE_8B,
        params=DEFAULT_GEN_PARAMS,
    )

    summary = (
        f"{status.value}: Gap of {gap_mw:+.1f} MW. "
        f"{len(commands)} dispatch command(s) issued."
    )

    result = BalancingResult(
        realtime_demand_mw=inp.realtime_demand_mw,
        total_supply_mw=total_supply,
        supply_demand_gap_mw=round(gap_mw, 2),
        grid_status=status,
        is_peak_hour=is_peak,
        dispatch_commands=commands,
        agent_reasoning=reasoning,
        recommended_action_summary=summary,
    )
    logger.info("demand_agent | done", status=status.value, gap_mw=gap_mw)
    return result
