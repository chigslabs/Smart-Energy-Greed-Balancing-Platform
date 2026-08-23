"""
agents/battery_agent.py – Battery Storage Optimization Agent
─────────────────────────────────────────────────────────────
Inputs  : current SoC (%), battery capacity (MWh), peak/off-peak tariff (₹/kWh),
          supply-demand gap (MW), hour of day
Outputs : charge/discharge/hold decision, power setpoint (MW), projected SoC,
          ROI estimate, and IBM Granite reasoning.

ReAct pattern:
  1. THINK  – evaluate SoC safety bounds and tariff context
  2. ACT    – state machine: CHARGE | DISCHARGE | HOLD
  3. REASON – Granite LLM validates decision and provides advisory
  4. RETURN – structured BatteryDecision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import (
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    BATTERY_SOC_TARGET,
    DEFAULT_GEN_PARAMS,
    GRANITE_8B,
    GUJARAT_PEAK_HOURS,
    get_settings,
)
from utils.llm_client import get_llm_response
from utils.logger import get_logger

logger = get_logger("battery_agent")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ──────────────────────────────────────────────────────────────────────────────

class BatteryAction(str, Enum):
    CHARGE    = "CHARGE"
    DISCHARGE = "DISCHARGE"
    HOLD      = "HOLD"


class DegradationRisk(str, Enum):
    LOW      = "LOW"     # SoC 25–85%
    MODERATE = "MODERATE"  # SoC 15–24% or 86–90%
    HIGH     = "HIGH"    # SoC < 15% or > 90% (safety cutoff triggered)


# Typical Gujarat tariff schedule
_PEAK_TARIFF_RS_KWH    = 8.50   # ₹/kWh during peak hours
_OFFPEAK_TARIFF_RS_KWH = 3.20   # ₹/kWh off-peak

# Round-trip efficiency of Li-ion battery
_RTE = 0.92   # 92%

# Maximum C-rate (charge/discharge speed relative to capacity)
_MAX_C_RATE = 0.5   # 0.5C = can fully charge/discharge in 2 hours


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BatteryInput:
    """Input bundle for the Battery Optimization Agent."""
    current_soc_pct: float         # State of Charge in %
    capacity_mwh: float            # Nameplate energy capacity (MWh)
    supply_demand_gap_mw: float    # Positive = surplus, Negative = deficit
    hour_of_day: int               # IST hour (0–23)
    peak_tariff: float = _PEAK_TARIFF_RS_KWH
    offpeak_tariff: float = _OFFPEAK_TARIFF_RS_KWH
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class BatteryDecision:
    """Output payload from the Battery Optimization Agent."""
    action: BatteryAction
    power_setpoint_mw: float       # Positive = charging, Negative = discharging
    duration_hours: float          # Estimated action duration
    projected_soc_after_pct: float
    energy_throughput_mwh: float
    degradation_risk: DegradationRisk
    tariff_rs_kwh: float
    estimated_roi_rs: float        # Revenue from arbitrage or cost saving
    safety_override: bool          # True if hard SoC limit forced the decision
    agent_reasoning: str
    action_summary: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Battery State Machine
# ──────────────────────────────────────────────────────────────────────────────

def _max_charge_mw(capacity_mwh: float, current_soc_pct: float) -> float:
    """Max power (MW) at which we can charge without exceeding SoC_MAX or C-rate."""
    available_capacity_mwh = capacity_mwh * (BATTERY_SOC_MAX - current_soc_pct) / 100.0
    c_rate_limit_mw        = capacity_mwh * _MAX_C_RATE
    return round(min(available_capacity_mwh, c_rate_limit_mw), 2)


def _max_discharge_mw(capacity_mwh: float, current_soc_pct: float) -> float:
    """Max power (MW) at which we can discharge without going below SoC_MIN."""
    available_energy_mwh = capacity_mwh * (current_soc_pct - BATTERY_SOC_MIN) / 100.0
    c_rate_limit_mw      = capacity_mwh * _MAX_C_RATE
    return round(min(available_energy_mwh, c_rate_limit_mw), 2)


def _estimate_roi(
    action: BatteryAction,
    power_mw: float,
    duration_h: float,
    tariff: float,
) -> float:
    """Estimate revenue/saving (₹) from the planned operation."""
    energy_mwh = power_mw * duration_h
    energy_kwh = energy_mwh * 1000
    if action == BatteryAction.DISCHARGE:
        # Revenue from selling/supplying during peak
        return round(energy_kwh * tariff * _RTE, 2)
    if action == BatteryAction.CHARGE:
        # Cost saving by charging cheaply now for peak discharge later
        return round(energy_kwh * (_PEAK_TARIFF_RS_KWH - tariff) * _RTE, 2)
    return 0.0


def _degradation_risk(soc_pct: float) -> DegradationRisk:
    if soc_pct < BATTERY_SOC_MIN or soc_pct > BATTERY_SOC_MAX:
        return DegradationRisk.HIGH
    if soc_pct < 25 or soc_pct > 85:
        return DegradationRisk.MODERATE
    return DegradationRisk.LOW


def _run_state_machine(inp: BatteryInput) -> tuple[BatteryAction, float, bool]:
    """
    Core battery state machine.
    Returns (action, power_mw, safety_override).

    Priority rules:
      1. Hard safety bounds → HOLD (override)
      2. Critical deficit   → DISCHARGE (grid emergency)
      3. Surplus + off-peak → CHARGE (energy arbitrage)
      4. Deficit + peak     → DISCHARGE (peak shaving)
      5. Default            → HOLD
    """
    is_peak      = inp.hour_of_day in GUJARAT_PEAK_HOURS
    gap          = inp.supply_demand_gap_mw
    soc          = inp.current_soc_pct

    # ── Rule 1: Safety Hard Stops ─────────────────────────────────────────────
    if soc <= BATTERY_SOC_MIN and gap < 0:
        # Too low to discharge – hold and raise alarm
        logger.warning("battery_agent | SoC at MIN threshold – forcing HOLD",
                       soc=soc, min=BATTERY_SOC_MIN)
        return BatteryAction.HOLD, 0.0, True

    if soc >= BATTERY_SOC_MAX and gap > 0:
        # Battery full – cannot absorb more surplus
        logger.warning("battery_agent | SoC at MAX threshold – forcing HOLD",
                       soc=soc, max=BATTERY_SOC_MAX)
        return BatteryAction.HOLD, 0.0, True

    # ── Rule 2: Critical Deficit → Emergency Discharge ────────────────────────
    if gap < -50 and soc > BATTERY_SOC_MIN + 10:
        max_d = _max_discharge_mw(inp.capacity_mwh, soc)
        power = min(abs(gap), max_d)
        return BatteryAction.DISCHARGE, power, False

    # ── Rule 3: Surplus + Off-Peak → Charge (cheap arbitrage) ────────────────
    if gap > 10 and not is_peak and soc < BATTERY_SOC_MAX - 5:
        max_c = _max_charge_mw(inp.capacity_mwh, soc)
        power = min(gap, max_c)
        return BatteryAction.CHARGE, power, False

    # ── Rule 4: Any Deficit + Peak → Discharge (peak shaving) ────────────────
    if gap < -5 and is_peak and soc > BATTERY_SOC_MIN + 10:
        max_d = _max_discharge_mw(inp.capacity_mwh, soc)
        power = min(abs(gap), max_d)
        return BatteryAction.DISCHARGE, power, False

    # ── Rule 5: Light Surplus → Top-up charge ────────────────────────────────
    if 0 < gap <= 10 and soc < BATTERY_SOC_TARGET:
        max_c = _max_charge_mw(inp.capacity_mwh, soc)
        power = min(gap, max_c)
        return BatteryAction.CHARGE, power, False

    # ── Default: Hold ─────────────────────────────────────────────────────────
    return BatteryAction.HOLD, 0.0, False


def _project_soc(
    current_soc: float,
    capacity_mwh: float,
    action: BatteryAction,
    power_mw: float,
    duration_h: float,
) -> float:
    """Project SoC after the planned operation (clamped to safe range)."""
    if action == BatteryAction.HOLD or power_mw == 0:
        return current_soc

    delta_mwh = power_mw * duration_h
    if action == BatteryAction.CHARGE:
        new_energy_mwh = (current_soc / 100) * capacity_mwh + delta_mwh * _RTE
    else:  # DISCHARGE
        new_energy_mwh = (current_soc / 100) * capacity_mwh - delta_mwh / _RTE

    new_soc = (new_energy_mwh / capacity_mwh) * 100
    return round(max(BATTERY_SOC_MIN, min(BATTERY_SOC_MAX, new_soc)), 1)


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Reasoning
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    inp: BatteryInput,
    action: BatteryAction,
    power_mw: float,
    projected_soc: float,
    roi: float,
    override: bool,
) -> str:
    tariff = inp.peak_tariff if inp.hour_of_day in GUJARAT_PEAK_HOURS else inp.offpeak_tariff
    return f"""You are a battery management system (BMS) expert for a 500 MWh grid-scale BESS in Gujarat.

### Battery State
- Current SoC         : {inp.current_soc_pct:.1f}%  (Safe range: {BATTERY_SOC_MIN}%–{BATTERY_SOC_MAX}%)
- Capacity            : {inp.capacity_mwh:.0f} MWh
- Supply-Demand Gap   : {inp.supply_demand_gap_mw:+.1f} MW
- Hour (IST)          : {inp.hour_of_day:02d}:00
- Tariff              : ₹{tariff:.2f}/kWh

### Decision
- Action              : {action.value}
- Power Setpoint      : {power_mw:.1f} MW
- Projected SoC After : {projected_soc:.1f}%
- Safety Override     : {"YES – hard limit triggered" if override else "No"}
- Estimated ROI       : ₹{roi:,.0f}

### Task
In 2–3 sentences explain:
1. Why this decision is optimal for grid stability and battery longevity.
2. Any risk to watch for in the next operating cycle.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public Agent Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run_battery_agent(inp: BatteryInput) -> BatteryDecision:
    """
    Execute the Battery Optimization ReAct cycle.
    """
    logger.info("battery_agent | evaluating state",
                soc=inp.current_soc_pct, gap=inp.supply_demand_gap_mw)

    # ── ACT: state machine ───────────────────────────────────────────────────
    action, power_mw, safety_override = _run_state_machine(inp)

    # ── ACT: derived metrics ─────────────────────────────────────────────────
    duration_h  = round(power_mw / (inp.capacity_mwh * _MAX_C_RATE), 2) \
        if (power_mw > 0 and inp.capacity_mwh > 0) else 0.0
    duration_h  = min(duration_h, 2.0)  # Cap at 2-hour horizon

    projected_soc = _project_soc(
        inp.current_soc_pct, inp.capacity_mwh, action, power_mw, duration_h
    )
    energy_throughput = round(power_mw * duration_h, 2)
    tariff = (
        inp.peak_tariff if inp.hour_of_day in GUJARAT_PEAK_HOURS else inp.offpeak_tariff
    )
    roi = _estimate_roi(action, power_mw, duration_h, tariff)

    # ── REASON: IBM Granite advisory ─────────────────────────────────────────
    prompt    = _build_prompt(inp, action, power_mw, projected_soc, roi, safety_override)
    reasoning = get_llm_response(
        prompt=prompt,
        model_id=GRANITE_8B,
        params=DEFAULT_GEN_PARAMS,
    )

    degrad_risk = _degradation_risk(projected_soc)
    summary = (
        f"{action.value} at {power_mw:.1f} MW for ~{duration_h:.1f}h. "
        f"Projected SoC: {projected_soc:.1f}%. "
        f"Est. ROI: ₹{roi:,.0f}."
        + (" [SAFETY OVERRIDE ACTIVE]" if safety_override else "")
    )

    result = BatteryDecision(
        action=action,
        power_setpoint_mw=power_mw,
        duration_hours=duration_h,
        projected_soc_after_pct=projected_soc,
        energy_throughput_mwh=energy_throughput,
        degradation_risk=degrad_risk,
        tariff_rs_kwh=tariff,
        estimated_roi_rs=roi,
        safety_override=safety_override,
        agent_reasoning=reasoning,
        action_summary=summary,
    )
    logger.info("battery_agent | decision complete", action=action.value,
                power_mw=power_mw, projected_soc=projected_soc)
    return result
