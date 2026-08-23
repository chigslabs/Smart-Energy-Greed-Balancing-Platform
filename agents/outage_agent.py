"""
agents/outage_agent.py – Outage Prediction Agent
─────────────────────────────────────────────────
Inputs  : transformer temperature (°C), voltage drop (%), line loading (%),
          weather alert severity (0–3), ambient temperature (°C)
Outputs : composite Risk Score (0–100), risk level, predicted failure window,
          preventive dispatch signals, and IBM Granite reasoning.

ReAct pattern:
  1. THINK  – weight sensor signals against Gujarat grid baselines
  2. ACT    – compute composite risk score and failure window
  3. REASON – Granite LLM refines alert text and recommends preventive actions
  4. RETURN – structured OutageRiskResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import (
    DEFAULT_GEN_PARAMS,
    GRANITE_8B,
    OUTAGE_RISK_HIGH,
    OUTAGE_RISK_LOW,
    OUTAGE_RISK_MEDIUM,
    get_settings,
)
from utils.llm_client import get_llm_response
from utils.logger import get_logger

logger = get_logger("outage_agent")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    NORMAL   = "NORMAL"    # 0–30
    WATCH    = "WATCH"     # 31–60
    ALERT    = "ALERT"     # 61–80
    CRITICAL = "CRITICAL"  # 81–100


class WeatherAlert(int, Enum):
    NONE     = 0   # Clear / no alert
    ADVISORY = 1   # Thunderstorm watch or heat advisory
    WARNING  = 2   # Severe weather warning
    EXTREME  = 3   # Cyclone / extreme heat (>47°C) event


# Gujarat-specific transformer baseline thresholds
_TX_TEMP_NOMINAL_C  = 65.0    # Normal oil-top temperature
_TX_TEMP_CAUTION_C  = 90.0    # Caution threshold
_TX_TEMP_CRITICAL_C = 105.0   # Immediate intervention required

_VOLTAGE_DROP_CAUTION_PCT   = 5.0    # > 5% deviation → watch
_VOLTAGE_DROP_CRITICAL_PCT  = 10.0   # > 10% → alert

_LINE_LOADING_CAUTION_PCT   = 80.0   # > 80% rated → watch
_LINE_LOADING_CRITICAL_PCT  = 95.0   # > 95% → alert


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OutageInput:
    """Input bundle for the Outage Prediction Agent."""
    transformer_temp_c: float      # Oil-top temperature in °C
    voltage_drop_pct: float        # Voltage deviation from nominal (%)
    line_loading_pct: float        # Current load as % of rated capacity
    weather_alert: int             # WeatherAlert enum value (0–3)
    ambient_temp_c: float          # Ambient air temperature (°C)
    asset_id: str = "GRID-TX-01"  # Asset identifier
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PreventiveSignal:
    """Single preventive dispatch or alert signal."""
    signal_type: str        # "ALERT" | "DISPATCH" | "INSPECTION"
    target: str             # Asset or operator target
    message: str
    urgency: str            # "IMMEDIATE" | "WITHIN_1H" | "SCHEDULED"


@dataclass
class OutageRiskResult:
    """Output payload from the Outage Prediction Agent."""
    asset_id: str
    risk_score: float               # 0–100
    risk_level: RiskLevel
    predicted_failure_window_h: Optional[float]   # Hours to potential failure
    contributing_factors: list[str]
    preventive_signals: list[PreventiveSignal]
    agent_reasoning: str
    operator_advisory: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Risk Scoring Engine
# ──────────────────────────────────────────────────────────────────────────────

def _score_transformer_temp(temp_c: float) -> tuple[float, str | None]:
    """
    Contributes up to 35 points to total risk.
    Uses exponential sensitivity above the caution threshold.
    """
    if temp_c <= _TX_TEMP_NOMINAL_C:
        return 0.0, None
    if temp_c <= _TX_TEMP_CAUTION_C:
        score = ((temp_c - _TX_TEMP_NOMINAL_C) /
                 (_TX_TEMP_CAUTION_C - _TX_TEMP_NOMINAL_C)) * 15.0
        return round(score, 1), f"Transformer temp {temp_c:.1f}°C approaching caution limit"
    if temp_c <= _TX_TEMP_CRITICAL_C:
        score = 15.0 + ((temp_c - _TX_TEMP_CAUTION_C) /
                        (_TX_TEMP_CRITICAL_C - _TX_TEMP_CAUTION_C)) * 20.0
        return round(score, 1), f"Transformer temp {temp_c:.1f}°C in CAUTION zone"
    # Above critical
    return 35.0, f"CRITICAL: Transformer temp {temp_c:.1f}°C exceeds {_TX_TEMP_CRITICAL_C}°C limit"


def _score_voltage_drop(drop_pct: float) -> tuple[float, str | None]:
    """Contributes up to 25 points."""
    if drop_pct <= _VOLTAGE_DROP_CAUTION_PCT:
        return 0.0, None
    if drop_pct <= _VOLTAGE_DROP_CRITICAL_PCT:
        score = ((drop_pct - _VOLTAGE_DROP_CAUTION_PCT) /
                 (_VOLTAGE_DROP_CRITICAL_PCT - _VOLTAGE_DROP_CAUTION_PCT)) * 15.0
        return round(score, 1), f"Voltage drop {drop_pct:.1f}% – degraded power quality"
    return 25.0, f"CRITICAL voltage sag {drop_pct:.1f}% – equipment protection risk"


def _score_line_loading(loading_pct: float) -> tuple[float, str | None]:
    """Contributes up to 25 points."""
    if loading_pct <= _LINE_LOADING_CAUTION_PCT:
        return 0.0, None
    if loading_pct <= _LINE_LOADING_CRITICAL_PCT:
        score = ((loading_pct - _LINE_LOADING_CAUTION_PCT) /
                 (_LINE_LOADING_CRITICAL_PCT - _LINE_LOADING_CAUTION_PCT)) * 15.0
        return round(score, 1), f"Line loading {loading_pct:.1f}% – thermal overload risk"
    return 25.0, f"CRITICAL: Line at {loading_pct:.1f}% – immediate deloading required"


def _score_weather(alert: int, ambient_c: float) -> tuple[float, str | None]:
    """Contributes up to 15 points. Weather compounds other risks."""
    base = float(alert) * 3.5  # 0, 3.5, 7.0, 10.5 for alert levels 0–3
    heat_bonus = max(0.0, (ambient_c - 40.0) * 0.45) if ambient_c > 40 else 0.0
    score = min(15.0, base + heat_bonus)
    if score == 0:
        return 0.0, None
    weather_names = {0: "None", 1: "Advisory", 2: "Warning", 3: "Extreme"}
    return round(score, 1), (
        f"Weather alert level {alert} ({weather_names.get(alert,'Unknown')}) "
        f"at ambient {ambient_c:.1f}°C"
    )


def _compute_risk(inp: OutageInput) -> tuple[float, list[str]]:
    """Aggregate component scores into a composite 0–100 risk score."""
    s_tx,  f_tx  = _score_transformer_temp(inp.transformer_temp_c)
    s_vol, f_vol = _score_voltage_drop(inp.voltage_drop_pct)
    s_ll,  f_ll  = _score_line_loading(inp.line_loading_pct)
    s_wx,  f_wx  = _score_weather(inp.weather_alert, inp.ambient_temp_c)

    total = min(100.0, s_tx + s_vol + s_ll + s_wx)
    factors = [f for f in [f_tx, f_vol, f_ll, f_wx] if f is not None]
    return round(total, 1), factors


def _estimate_failure_window(risk_score: float) -> Optional[float]:
    """
    Rough time-to-failure estimate in hours based on risk score.
    Above OUTAGE_RISK_MEDIUM we start predicting a window.
    """
    if risk_score <= OUTAGE_RISK_MEDIUM:
        return None
    if risk_score >= 90:
        return 0.5    # 30 minutes
    if risk_score >= OUTAGE_RISK_HIGH:
        return round(2.0 - (risk_score - OUTAGE_RISK_HIGH) / 20 * 1.5, 1)
    # WATCH → ALERT range
    return round(8.0 - (risk_score - OUTAGE_RISK_MEDIUM) / 20 * 4.0, 1)


def _build_preventive_signals(
    risk_score: float,
    risk_level: RiskLevel,
    inp: OutageInput,
) -> list[PreventiveSignal]:
    """Generate preventive dispatch and alert signals based on risk level."""
    signals: list[PreventiveSignal] = []

    if risk_level == RiskLevel.NORMAL:
        return signals

    if risk_level == RiskLevel.WATCH:
        signals.append(PreventiveSignal(
            signal_type="ALERT",
            target="Grid Operations Centre",
            message=f"Asset {inp.asset_id} at WATCH status (score: {risk_score:.0f}). "
                    "Increase monitoring frequency to 15-minute intervals.",
            urgency="WITHIN_1H",
        ))

    elif risk_level == RiskLevel.ALERT:
        signals.append(PreventiveSignal(
            signal_type="DISPATCH",
            target="Field Maintenance Team",
            message=f"Preventive dispatch to {inp.asset_id}: Transformer temp "
                    f"{inp.transformer_temp_c:.1f}°C, Line loading {inp.line_loading_pct:.1f}%. "
                    "Initiate forced cooling and prepare bypass circuit.",
            urgency="WITHIN_1H",
        ))
        signals.append(PreventiveSignal(
            signal_type="ALERT",
            target="Load Dispatch Centre (SLDC)",
            message=f"Reduce load on feeder connected to {inp.asset_id} by 15% immediately.",
            urgency="IMMEDIATE",
        ))

    elif risk_level == RiskLevel.CRITICAL:
        signals.append(PreventiveSignal(
            signal_type="DISPATCH",
            target="Emergency Response Team",
            message=f"CRITICAL RISK on {inp.asset_id} (score: {risk_score:.0f}). "
                    "Immediate isolation and hot-standby transformer activation required.",
            urgency="IMMEDIATE",
        ))
        signals.append(PreventiveSignal(
            signal_type="DISPATCH",
            target="SLDC Operator",
            message="Activate N-1 contingency: reroute load via alternate feeder. "
                    "Prepare controlled load shedding protocol.",
            urgency="IMMEDIATE",
        ))
        if inp.weather_alert >= WeatherAlert.WARNING:
            signals.append(PreventiveSignal(
                signal_type="INSPECTION",
                target="Weather Risk Assessment Team",
                message=f"Severe weather (level {inp.weather_alert}) compounding risk. "
                        "Deploy storm-response protocol for affected substations.",
                urgency="IMMEDIATE",
            ))

    return signals


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Reasoning
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    inp: OutageInput,
    risk_score: float,
    risk_level: RiskLevel,
    factors: list[str],
    failure_window: Optional[float],
) -> str:
    factor_text = "\n".join(f"  - {f}" for f in factors) if factors else "  - None identified"
    window_text = (
        f"~{failure_window:.1f} hours" if failure_window else "No imminent failure predicted"
    )
    return f"""You are a power grid reliability engineer at Gujarat SLDC analyzing equipment telemetry.

### Equipment: {inp.asset_id}
- Transformer Temp    : {inp.transformer_temp_c:.1f}°C  (Critical: >{_TX_TEMP_CRITICAL_C}°C)
- Voltage Drop        : {inp.voltage_drop_pct:.1f}%
- Line Loading        : {inp.line_loading_pct:.1f}% of rated capacity
- Weather Alert Level : {inp.weather_alert} / 3
- Ambient Temp        : {inp.ambient_temp_c:.1f}°C

### Risk Assessment
- Composite Risk Score    : {risk_score:.1f} / 100  → {risk_level.value}
- Predicted Failure Window: {window_text}
- Contributing Factors:
{factor_text}

### Task
Provide a 3–4 sentence operator advisory:
1. Primary risk driver and its immediate consequence if unaddressed.
2. Most effective preventive action with time criticality.
3. Recommended monitoring parameter to confirm risk reduction after action.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public Agent Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run_outage_agent(inp: OutageInput) -> OutageRiskResult:
    """
    Execute the Outage Prediction ReAct cycle.
    """
    logger.info("outage_agent | computing risk", asset=inp.asset_id,
                tx_temp=inp.transformer_temp_c, line_loading=inp.line_loading_pct)

    # ── THINK + ACT: compute composite risk ──────────────────────────────────
    risk_score, factors = _compute_risk(inp)

    if risk_score <= OUTAGE_RISK_LOW:
        risk_level = RiskLevel.NORMAL
    elif risk_score <= OUTAGE_RISK_MEDIUM:
        risk_level = RiskLevel.WATCH
    elif risk_score <= OUTAGE_RISK_HIGH:
        risk_level = RiskLevel.ALERT
    else:
        risk_level = RiskLevel.CRITICAL

    failure_window = _estimate_failure_window(risk_score)
    signals        = _build_preventive_signals(risk_score, risk_level, inp)

    # ── REASON: IBM Granite advisory ─────────────────────────────────────────
    prompt    = _build_prompt(inp, risk_score, risk_level, factors, failure_window)
    reasoning = get_llm_response(
        prompt=prompt,
        model_id=GRANITE_8B,
        params=DEFAULT_GEN_PARAMS,
    )

    advisory = (
        f"Risk {risk_level.value} ({risk_score:.0f}/100) on {inp.asset_id}. "
        + (f"Estimated time to failure: {failure_window:.1f}h. " if failure_window else "")
        + f"{len(signals)} preventive signal(s) issued."
    )

    result = OutageRiskResult(
        asset_id=inp.asset_id,
        risk_score=risk_score,
        risk_level=risk_level,
        predicted_failure_window_h=failure_window,
        contributing_factors=factors,
        preventive_signals=signals,
        agent_reasoning=reasoning,
        operator_advisory=advisory,
    )
    logger.info("outage_agent | done", risk_score=risk_score, risk_level=risk_level.value)
    return result
