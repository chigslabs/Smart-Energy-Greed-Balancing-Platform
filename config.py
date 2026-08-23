"""
config.py – IBM Watsonx credentials, model constants, and application settings.

All sensitive values are loaded from environment variables or a .env file.
Copy `.env.example` to `.env` and fill in your credentials before running.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Model Constants
# ──────────────────────────────────────────────────────────────────────────────

GRANITE_8B  = "ibm/granite-3-8b-instruct"
GRANITE_20B = "ibm/granite-3-20b-instruct"

# Default generation parameters used across all agents
DEFAULT_GEN_PARAMS: dict = {
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "temperature": 0.2,        # Low temp for deterministic grid decisions
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# Gujarat grid constants
GUJARAT_GRID_FREQ_HZ: float = 50.0          # Nominal grid frequency
GUJARAT_PEAK_HOURS: list[int] = [9, 10, 11, 18, 19, 20, 21]  # IST hours
GUJARAT_SOLAR_CAPACITY_MW: float = 10_000.0  # ~10 GW installed solar capacity

# Battery safety thresholds
BATTERY_SOC_MIN: float = 15.0   # % – below this: force-stop discharge
BATTERY_SOC_MAX: float = 90.0   # % – above this: force-stop charging
BATTERY_SOC_TARGET: float = 60.0  # % – ideal resting state

# Risk score thresholds for outage agent
OUTAGE_RISK_LOW    = 30   # 0–30   → Normal
OUTAGE_RISK_MEDIUM = 60   # 31–60  → Watch
OUTAGE_RISK_HIGH   = 80   # 61–80  → Alert
                          # 81–100 → Critical


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Settings (auto-loaded from .env)
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Application-wide settings resolved from environment variables.
    All Watsonx credentials must be set before starting the server.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── IBM Watsonx Core ──────────────────────────────────────────────────────
    watsonx_api_key: str = Field(
        default="",
        validation_alias="WATSONX_API_KEY",
        description="IBM Cloud IAM API Key for Watsonx authentication.",
    )
    watsonx_project_id: str = Field(
        default="",
        validation_alias="WATSONX_PROJECT_ID",
        description="Watsonx.ai Project ID (found in project settings).",
    )
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        validation_alias="WATSONX_URL",
        description="Regional Watsonx.ai endpoint URL.",
    )

    # ── Model Selection ───────────────────────────────────────────────────────
    default_model: str = Field(
        default=GRANITE_8B,
        validation_alias="WATSONX_DEFAULT_MODEL",
        description="Default IBM Granite model ID.",
    )
    orchestrator_model: str = Field(
        default=GRANITE_20B,
        validation_alias="WATSONX_ORCHESTRATOR_MODEL",
        description="Heavier model used by the orchestration layer.",
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000,      validation_alias="API_PORT")
    api_debug: bool = Field(default=False,   validation_alias="API_DEBUG")
    cors_origins: list[str] = Field(
        default=["*"],
        validation_alias="CORS_ORIGINS",
    )

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_api_base: str = Field(
        default="http://localhost:8000",
        validation_alias="STREAMLIT_API_BASE",
        description="FastAPI base URL consumed by the Streamlit dashboard.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    # ── Simulation mode (no real Watsonx calls) ───────────────────────────────
    simulation_mode: bool = Field(
        default=True,
        validation_alias="SIMULATION_MODE",
        description=(
            "When True the platform runs with deterministic mock LLM responses. "
            "Set to False only when valid Watsonx credentials are provided."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("watsonx_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def watsonx_credentials(self) -> dict:
        """Return a credentials dict compatible with ibm-watsonx-ai client."""
        return {
            "url": self.watsonx_url,
            "apikey": self.watsonx_api_key,
        }

    @property
    def is_watsonx_configured(self) -> bool:
        """True only if real credentials are present."""
        return bool(self.watsonx_api_key and self.watsonx_project_id)


# ──────────────────────────────────────────────────────────────────────────────
# Singleton accessor (cached after first call)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide settings singleton."""
    return Settings()


# ──────────────────────────────────────────────────────────────────────────────
# .env.example generator (run this file directly to create the template)
# ──────────────────────────────────────────────────────────────────────────────

_ENV_EXAMPLE = """\
# ── IBM Watsonx Credentials ──────────────────────────────────────────────────
WATSONX_API_KEY=your_ibm_cloud_iam_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# ── Model Selection (optional – defaults shown) ───────────────────────────────
WATSONX_DEFAULT_MODEL=ibm/granite-3-8b-instruct
WATSONX_ORCHESTRATOR_MODEL=ibm/granite-3-20b-instruct

# ── FastAPI Server ────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
CORS_ORIGINS=["*"]

# ── Streamlit Dashboard ───────────────────────────────────────────────────────
STREAMLIT_API_BASE=http://localhost:8000

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO

# ── Simulation Mode ───────────────────────────────────────────────────────────
# Set to false only when real Watsonx credentials are provided above
SIMULATION_MODE=true
"""

if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), ".env.example")
    with open(env_path, "w") as fh:
        fh.write(_ENV_EXAMPLE)
    print(f".env.example written to {env_path}")
    s = get_settings()
    print(f"Settings loaded → simulation_mode={s.simulation_mode}, model={s.default_model}")
