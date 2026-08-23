"""
utils/logger.py – Structured logger factory using structlog.
All agents and services call get_logger(__name__) to obtain a bound logger.
"""

from __future__ import annotations

import logging
import sys

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog bound logger for the given module name.
    First call configures structlog globally (idempotent afterwards).
    """
    _configure_once()
    return structlog.get_logger(name)


_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return

    from config import get_settings
    settings  = get_settings()
    log_level = getattr(logging, settings.log_level, logging.INFO)

    # Standard-library logging handler
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True
