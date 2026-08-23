"""
utils/llm_client.py – Unified IBM Watsonx / Simulation LLM call wrapper
─────────────────────────────────────────────────────────────────────────
All agents call `get_llm_response(prompt, model_id, params)`.
  • If SIMULATION_MODE=True  → returns deterministic mock responses instantly.
  • If SIMULATION_MODE=False → calls ibm-watsonx-ai SDK (credentials required).
"""

from __future__ import annotations

import hashlib
import textwrap
from functools import lru_cache
from typing import Any

from config import get_settings
from utils.logger import get_logger

logger   = get_logger("llm_client")
settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# Simulation mock bank (deterministic for demo / CI)
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_TEMPLATES = [
    (
        "solar",
        "The current solar generation of {value:.1f} MW aligns with expected insolation for "
        "this hour. Gujarat's high irradiance levels are being effectively captured; no "
        "immediate grid intervention is required. Operators should monitor cloud-cover "
        "forecasts and keep battery reserves available for rapid ramp-down compensation.",
    ),
    (
        "battery",
        "The {action} decision at {value:.1f} MW is optimal given current SoC and tariff "
        "conditions, balancing grid stability with battery longevity. Maintaining SoC within "
        "the 15–90% window prevents accelerated cycle degradation. Monitor State of Health "
        "(SoH) telemetry in the next cycle to confirm no capacity drift.",
    ),
    (
        "demand",
        "The proposed dispatch plan addresses the supply-demand gap efficiently using "
        "least-cost merit order. Priority given to solar and battery reduces thermal "
        "dispatch emissions. Recommend monitoring frequency deviation every 5 minutes "
        "to confirm grid stabilisation post-dispatch.",
    ),
    (
        "outage",
        "The primary risk driver is elevated transformer temperature compounded by high "
        "line loading, which accelerates insulation degradation. Immediate forced cooling "
        "and 15% load reduction on the affected feeder will reduce thermal stress within "
        "30 minutes. Track oil-top temperature as the key confirmation metric post-action.",
    ),
]


def _mock_response(prompt: str) -> str:
    """
    Pick a mock response template based on keyword presence in the prompt.
    Falls back to a generic grid advisory if no keyword matches.
    """
    lower = prompt.lower()
    for keyword, template in _MOCK_TEMPLATES:
        if keyword in lower:
            # Fill in numeric placeholders if detectable
            import re
            numbers = re.findall(r"\b\d+\.?\d*\b", prompt)
            value = float(numbers[0]) if numbers else 100.0
            action = "DISCHARGE" if "discharge" in lower else "CHARGE"
            try:
                return template.format(value=value, action=action)
            except (KeyError, ValueError):
                return template.replace("{value:.1f}", str(value)) \
                               .replace("{action}", action)

    return (
        "Grid telemetry analysis complete. All monitored parameters are within acceptable "
        "operating bounds for Gujarat's transmission network. Recommend maintaining current "
        "dispatch schedule and scheduling routine preventive maintenance."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Watsonx real call
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_watsonx_client():
    """Lazy-initialise Watsonx ModelInference client (cached singleton)."""
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        creds = Credentials(
            url=settings.watsonx_url,
            api_key=settings.watsonx_api_key,
        )
        return ModelInference
    except ImportError as exc:
        logger.error("ibm-watsonx-ai SDK not installed", error=str(exc))
        raise


def _call_watsonx(
    prompt: str,
    model_id: str,
    params: dict[str, Any],
) -> str:
    """Call the real IBM Watsonx API and extract generated text."""
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(
            url=settings.watsonx_url,
            api_key=settings.watsonx_api_key,
        )
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=settings.watsonx_project_id,
            params=params,
        )
        response = model.generate_text(prompt=prompt)
        return response.strip()

    except Exception as exc:
        logger.error("Watsonx API call failed", model=model_id, error=str(exc))
        return f"[LLM ERROR] {exc} — falling back to rule-based advisory."


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def get_llm_response(
    prompt: str,
    model_id: str,
    params: dict[str, Any] | None = None,
) -> str:
    """
    Unified entry point for all agent LLM calls.

    Args:
        prompt   : Full formatted prompt string.
        model_id : IBM Granite model identifier.
        params   : Generation hyperparameters (uses DEFAULT_GEN_PARAMS if None).

    Returns:
        Generated text string from LLM or simulation mock.
    """
    from config import DEFAULT_GEN_PARAMS
    params = params or DEFAULT_GEN_PARAMS

    if settings.simulation_mode or not settings.is_watsonx_configured:
        logger.debug("llm_client | simulation mode – returning mock response")
        return _mock_response(prompt)

    logger.debug("llm_client | calling Watsonx", model=model_id)
    return _call_watsonx(prompt, model_id, params)
