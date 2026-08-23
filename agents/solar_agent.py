"""
agents/solar_agent.py – Solar Generation Forecasting Agent
───────────────────────────────────────────────────────────
Inputs  : solar irradiance (W/m²), ambient temperature (°C),
          panel capacity (MW), hour of day (0–23), cloud cover (%)
Outputs : Forecasted solar output (MW), 24-hour profile, anomaly flags,
          and IBM Granite reasoning narrative.

ReAct pattern:
  1. THINK  – validate inputs and detect anomalies
  2. ACT    – run physics-based forecast model
  3. REASON – pass result + anomaly context to Granite for narrative
  4. RETURN – structured SolarForecastResult
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import (
    DEFAULT_GEN_PARAMS,
    GRANITE_8B,
    get_settings,
)
from utils.llm_client import get_llm_response
from utils.logger import get_logger

logger = get_logger("solar_agent")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SolarInput:
    """Validated input bundle for the Solar Forecasting Agent."""
    irradiance_wm2: float          # Solar irradiance in W/m²  (0–1200)
    ambient_temp_c: float          # Ambient temperature in °C
    panel_capacity_mw: float       # Nameplate capacity in MW
    hour_of_day: int               # Current hour in IST (0–23)
    cloud_cover_pct: float         # Cloud cover percentage (0–100)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class SolarForecastResult:
    """Output payload from the Solar Forecasting Agent."""
    current_output_mw: float
    capacity_factor_pct: float
    forecast_24h_mw: list[float]         # Hourly forecast for next 24 hours
    anomaly_detected: bool
    anomaly_description: str
    agent_reasoning: str                  # IBM Granite narrative
    confidence_pct: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Physics-based forecast helpers
# ──────────────────────────────────────────────────────────────────────────────

# Panel efficiency temperature coefficient (%/°C above 25°C STC)
_TEMP_COEFF = -0.004   # –0.4%/°C (typical crystalline silicon)

# Gujarat latitude ≈ 22.5°N – sunrise/sunset approximation
_GUJARAT_SOLAR_HOURS = {
    h: max(0.0, math.sin(math.pi * (h - 6) / 12))
    for h in range(24)
}  # Normalised insolation curve (0 before 6am and after 18pm)


def _temperature_derating(temp_c: float) -> float:
    """Return multiplier accounting for panel temperature loss."""
    delta = max(0.0, temp_c - 25.0)
    return max(0.1, 1.0 + _TEMP_COEFF * delta)


def _cloud_derating(cloud_cover_pct: float) -> float:
    """Return multiplier for cloud cover attenuation."""
    # Diffuse irradiance still contributes ~20% even under full overcast
    return max(0.20, 1.0 - (cloud_cover_pct / 100.0) * 0.80)


def _compute_output_mw(
    irradiance_wm2: float,
    temp_c: float,
    capacity_mw: float,
    hour: int,
    cloud_pct: float,
) -> float:
    """
    Physics-based instantaneous output estimation.
    Uses the simplified PV production formula:
        P = G/G_stc × Pmax × η_temp × η_cloud × η_diurnal
    where G_stc = 1000 W/m².
    """
    g_ratio    = min(irradiance_wm2 / 1000.0, 1.0)
    eta_temp   = _temperature_derating(temp_c)
    eta_cloud  = _cloud_derating(cloud_pct)
    eta_diurnal = _GUJARAT_SOLAR_HOURS.get(hour, 0.0)

    output = capacity_mw * g_ratio * eta_temp * eta_cloud * eta_diurnal
    return round(max(0.0, output), 2)


def _generate_24h_profile(
    capacity_mw: float,
    base_temp_c: float,
    base_cloud_pct: float,
    current_hour: int,
) -> list[float]:
    """
    Generate a 24-hour rolling forecast starting from (current_hour + 1).
    Cloud cover is stochastically evolved hour-by-hour.
    """
    profile: list[float] = []
    cloud = base_cloud_pct

    for offset in range(1, 25):
        future_hour = (current_hour + offset) % 24
        # Random walk on cloud cover (clamped to 0–100)
        cloud = float(max(0.0, min(100.0, cloud + random.gauss(0, 5))))
        temp  = base_temp_c + random.gauss(0, 2)
        # Nominal irradiance derived from diurnal curve
        g     = _GUJARAT_SOLAR_HOURS[future_hour] * 950.0
        output = _compute_output_mw(g, temp, capacity_mw, future_hour, cloud)
        profile.append(output)

    return profile


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_anomaly(inp: SolarInput, computed_mw: float) -> tuple[bool, str]:
    """
    Detect grid-relevant anomalies and return (flag, description).
    Rules:
      - Sudden cloud burst: cloud > 70% during peak solar hours (9–16)
      - Irradiance crash:   irradiance < 100 W/m² during daytime
      - Overheat:           ambient temp > 45°C
      - Night generation:   output > 0 when hour outside 6–18
    """
    anomalies: list[str] = []

    if 9 <= inp.hour_of_day <= 16 and inp.cloud_cover_pct > 70:
        anomalies.append(
            f"Sudden cloud burst detected: {inp.cloud_cover_pct:.0f}% cloud cover "
            f"during peak solar window (hour {inp.hour_of_day} IST)."
        )
    if 6 <= inp.hour_of_day <= 18 and inp.irradiance_wm2 < 100:
        anomalies.append(
            f"Abnormally low irradiance ({inp.irradiance_wm2:.1f} W/m²) during daytime."
        )
    if inp.ambient_temp_c > 45:
        anomalies.append(
            f"High ambient temperature ({inp.ambient_temp_c:.1f}°C) – "
            "panel efficiency significantly degraded."
        )
    if (inp.hour_of_day < 6 or inp.hour_of_day > 18) and computed_mw > 0.5:
        anomalies.append(
            f"Unexpected generation ({computed_mw:.1f} MW) outside solar window."
        )

    return bool(anomalies), " | ".join(anomalies) if anomalies else "None"


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Reasoning
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(inp: SolarInput, output_mw: float, anomaly: str) -> str:
    return f"""You are an expert solar energy analyst for Gujarat's power grid.

### Current Sensor Data
- Solar Irradiance  : {inp.irradiance_wm2:.1f} W/m²
- Ambient Temp      : {inp.ambient_temp_c:.1f} °C
- Panel Capacity    : {inp.panel_capacity_mw:.1f} MW
- Hour of Day (IST) : {inp.hour_of_day:02d}:00
- Cloud Cover       : {inp.cloud_cover_pct:.1f}%
- Computed Output   : {output_mw:.2f} MW
- Anomaly Flag      : {anomaly}

### Task
1. Briefly validate the forecast reasonableness.
2. If an anomaly exists, explain its grid impact and recommend an immediate corrective action.
3. Provide a one-sentence operator advisory.

Respond in 3–5 concise sentences.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public Agent Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run_solar_agent(inp: SolarInput) -> SolarForecastResult:
    """
    Execute the Solar Forecasting ReAct cycle:
      THINK → ACT (physics model) → REASON (Granite LLM) → RETURN
    """
    logger.info("solar_agent | starting forecast", hour=inp.hour_of_day,
                irradiance=inp.irradiance_wm2, cloud=inp.cloud_cover_pct)

    # ── ACT: compute current output ──────────────────────────────────────────
    current_mw = _compute_output_mw(
        inp.irradiance_wm2,
        inp.ambient_temp_c,
        inp.panel_capacity_mw,
        inp.hour_of_day,
        inp.cloud_cover_pct,
    )
    capacity_factor = round((current_mw / inp.panel_capacity_mw) * 100, 2) \
        if inp.panel_capacity_mw > 0 else 0.0

    # ── THINK: anomaly detection ─────────────────────────────────────────────
    anomaly_flag, anomaly_desc = _detect_anomaly(inp, current_mw)

    # ── ACT: 24-hour profile ─────────────────────────────────────────────────
    forecast_24h = _generate_24h_profile(
        inp.panel_capacity_mw,
        inp.ambient_temp_c,
        inp.cloud_cover_pct,
        inp.hour_of_day,
    )

    # ── REASON: IBM Granite narrative ────────────────────────────────────────
    prompt = _build_prompt(inp, current_mw, anomaly_desc)
    reasoning = get_llm_response(
        prompt=prompt,
        model_id=GRANITE_8B,
        params=DEFAULT_GEN_PARAMS,
    )

    # Confidence degrades with cloud cover and anomalies
    confidence = round(
        max(40.0, 95.0 - inp.cloud_cover_pct * 0.4 - (10.0 if anomaly_flag else 0.0)),
        1,
    )

    result = SolarForecastResult(
        current_output_mw=current_mw,
        capacity_factor_pct=capacity_factor,
        forecast_24h_mw=forecast_24h,
        anomaly_detected=anomaly_flag,
        anomaly_description=anomaly_desc,
        agent_reasoning=reasoning,
        confidence_pct=confidence,
    )
    logger.info("solar_agent | forecast complete",
                output_mw=current_mw, anomaly=anomaly_flag, confidence=confidence)
    return result
