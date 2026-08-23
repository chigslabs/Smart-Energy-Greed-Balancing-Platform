"""
main.py – FastAPI server for the Smart Energy Management Platform
──────────────────────────────────────────────────────────────────
Endpoints:
  GET  /                            Health check
  GET  /api/v1/grid-status          Full pipeline run with default values
  POST /api/v1/forecast             Solar-only forecast (lightweight)
  POST /api/v1/agent-dispatch       Full pipeline with custom sensor input
  GET  /api/v1/simulate             Auto-generate a realistic Gujarat scenario

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agents.solar_agent import SolarInput, run_solar_agent
from config import get_settings
from orchestrator import GridPipelineInput, run_grid_pipeline
from utils.logger import get_logger

logger   = get_logger("main")
settings = get_settings()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Energy Management – Gujarat Grid API",
    description=(
        "Agentic AI platform for real-time solar forecasting, demand-supply balancing, "
        "battery storage optimization, and outage prediction using IBM Granite LLMs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas (Pydantic v2)
# ──────────────────────────────────────────────────────────────────────────────

class SolarForecastRequest(BaseModel):
    irradiance_wm2:    float = Field(700.0,  ge=0,    le=1200, description="W/m²")
    ambient_temp_c:    float = Field(35.0,   ge=-10,  le=55,   description="°C")
    panel_capacity_mw: float = Field(500.0,  gt=0,             description="MW")
    hour_of_day:       int   = Field(12,     ge=0,    le=23,   description="IST hour")
    cloud_cover_pct:   float = Field(20.0,   ge=0,    le=100,  description="%")


class DispatchRequest(BaseModel):
    """Full sensor payload for the complete agent pipeline."""
    # Solar
    irradiance_wm2:      float = Field(700.0,  ge=0,   le=1200)
    ambient_temp_c:      float = Field(35.0,   ge=-10, le=55)
    panel_capacity_mw:   float = Field(500.0,  gt=0)
    hour_of_day:         int   = Field(12,     ge=0,   le=23)
    cloud_cover_pct:     float = Field(20.0,   ge=0,   le=100)
    # Demand
    realtime_demand_mw:  float = Field(420.0,  gt=0)
    thermal_capacity_mw: float = Field(200.0,  ge=0)
    load_profile:        str   = Field("mixed")
    # Battery
    battery_soc_pct:     float = Field(60.0,   ge=0,   le=100)
    battery_capacity_mwh:float = Field(500.0,  gt=0)
    # Outage
    transformer_temp_c:  float = Field(75.0,   ge=0,   le=200)
    voltage_drop_pct:    float = Field(3.5,    ge=0,   le=50)
    line_loading_pct:    float = Field(72.0,   ge=0,   le=150)
    weather_alert:       int   = Field(0,      ge=0,   le=3)
    asset_id:            str   = Field("GRID-TX-01")

    @field_validator("load_profile")
    @classmethod
    def valid_profile(cls, v: str) -> str:
        allowed = {"residential", "industrial", "mixed"}
        if v not in allowed:
            raise ValueError(f"load_profile must be one of {allowed}")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Simulation helper – realistic Gujarat day-cycle scenario
# ──────────────────────────────────────────────────────────────────────────────

def _generate_simulation_input(hour: Optional[int] = None) -> dict:
    """
    Generate a plausible sensor snapshot for a given IST hour.
    Mirrors Gujarat's typical solar generation and demand patterns.
    """
    h = hour if hour is not None else datetime.now().hour

    # Solar irradiance follows a bell curve peaking at noon
    import math
    base_irr = max(0.0, 950 * math.sin(math.pi * max(0, h - 6) / 12))
    cloud    = random.uniform(5, 60)
    irr      = base_irr * (1 - cloud / 200)

    # Demand peaks mornings and evenings
    demand_factor = 0.6 + 0.4 * abs(math.sin(math.pi * (h - 5) / 13))
    demand        = round(1800 * demand_factor + random.gauss(0, 30), 1)

    return {
        "irradiance_wm2":      round(irr, 1),
        "ambient_temp_c":      round(30 + 10 * math.sin(math.pi * (h - 6) / 12) + random.gauss(0, 1.5), 1),
        "panel_capacity_mw":   8500.0,       # ~8.5 GW Gujarat solar fleet
        "hour_of_day":         h,
        "cloud_cover_pct":     round(cloud, 1),
        "realtime_demand_mw":  demand,
        "thermal_capacity_mw": round(random.uniform(400, 800), 1),
        "load_profile":        random.choice(["residential", "industrial", "mixed"]),
        "battery_soc_pct":     round(random.uniform(25, 85), 1),
        "battery_capacity_mwh":500.0,
        "transformer_temp_c":  round(random.uniform(60, 100), 1),
        "voltage_drop_pct":    round(random.uniform(1, 12), 2),
        "line_loading_pct":    round(random.uniform(50, 110), 1),
        "weather_alert":       random.choices([0, 1, 2, 3], weights=[70, 20, 7, 3])[0],
        "asset_id":            random.choice(["GRID-TX-01", "GRID-TX-02", "GRID-TX-03"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check and platform info."""
    return {
        "service": "Smart Energy Management – Gujarat Grid",
        "status":  "online",
        "version": "1.0.0",
        "simulation_mode": settings.simulation_mode,
        "model":   settings.default_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/grid-status", tags=["Grid"], summary="Full pipeline – default sensor values")
async def grid_status():
    """
    Run the complete 5-agent pipeline with the current simulated Gujarat scenario.
    Returns the aggregated dashboard payload including all KPIs and agent reasoning.
    """
    try:
        sim_input = _generate_simulation_input()
        payload   = run_grid_pipeline(sim_input)
        return payload
    except Exception as exc:
        logger.error("grid_status endpoint error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        )


@app.post("/api/v1/forecast", tags=["Solar"], summary="Solar-only forecast (lightweight)")
async def solar_forecast(req: SolarForecastRequest):
    """
    Run only the Solar Forecasting Agent.
    Useful for lightweight forecast polling without triggering the full pipeline.
    """
    try:
        inp = SolarInput(
            irradiance_wm2    = req.irradiance_wm2,
            ambient_temp_c    = req.ambient_temp_c,
            panel_capacity_mw = req.panel_capacity_mw,
            hour_of_day       = req.hour_of_day,
            cloud_cover_pct   = req.cloud_cover_pct,
        )
        result = run_solar_agent(inp)
        return {
            "current_output_mw":    result.current_output_mw,
            "capacity_factor_pct":  result.capacity_factor_pct,
            "forecast_24h_mw":      result.forecast_24h_mw,
            "anomaly_detected":     result.anomaly_detected,
            "anomaly_description":  result.anomaly_description,
            "agent_reasoning":      result.agent_reasoning,
            "confidence_pct":       result.confidence_pct,
            "timestamp":            result.timestamp,
        }
    except Exception as exc:
        logger.error("forecast endpoint error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solar forecast failed: {exc}",
        )


@app.post("/api/v1/agent-dispatch", tags=["Grid"], summary="Full pipeline – custom sensor input")
async def agent_dispatch(req: DispatchRequest):
    """
    Run the complete 5-agent pipeline with user-supplied sensor readings.
    Returns full agent reasoning chains and dispatch commands.
    """
    try:
        pipeline_input: GridPipelineInput = {
            "irradiance_wm2":       req.irradiance_wm2,
            "ambient_temp_c":       req.ambient_temp_c,
            "panel_capacity_mw":    req.panel_capacity_mw,
            "hour_of_day":          req.hour_of_day,
            "cloud_cover_pct":      req.cloud_cover_pct,
            "realtime_demand_mw":   req.realtime_demand_mw,
            "thermal_capacity_mw":  req.thermal_capacity_mw,
            "load_profile":         req.load_profile,
            "battery_soc_pct":      req.battery_soc_pct,
            "battery_capacity_mwh": req.battery_capacity_mwh,
            "transformer_temp_c":   req.transformer_temp_c,
            "voltage_drop_pct":     req.voltage_drop_pct,
            "line_loading_pct":     req.line_loading_pct,
            "weather_alert":        req.weather_alert,
            "asset_id":             req.asset_id,
        }
        payload = run_grid_pipeline(pipeline_input)
        return payload
    except Exception as exc:
        logger.error("agent_dispatch endpoint error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent dispatch failed: {exc}",
        )


@app.get("/api/v1/simulate", tags=["Simulation"],
         summary="Auto-generate a realistic Gujarat scenario at any hour")
async def simulate(hour: Optional[int] = None):
    """
    Generate and run a realistic Gujarat grid scenario.
    Optionally pass `?hour=14` (IST 0–23) to simulate a specific time of day.
    """
    if hour is not None and not (0 <= hour <= 23):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hour must be in range 0–23",
        )
    try:
        sim_input = _generate_simulation_input(hour)
        payload   = run_grid_pipeline(sim_input)
        payload["simulation_input"] = sim_input   # Echo back for transparency
        return payload
    except Exception as exc:
        logger.error("simulate endpoint error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
