"""Observability - Structured Logging."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from graphrag.core.config import get_settings


def setup_logging() -> None:
    """Configure structured logging with structlog."""
    settings = get_settings()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_production:
        # JSON output for production
        shared_processors.append(structlog.processors.format_exc_info)
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger.

    Args:
        name: Logger name.

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


def add_trace_context(trace_id: str, span_id: str | None = None) -> None:
    """Add trace context to logs.

    Args:
        trace_id: Trace ID.
        span_id: Optional span ID.
    """
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id,
        span_id=span_id,
    )


def clear_trace_context() -> None:
    """Clear trace context from logs."""
    structlog.contextvars.unbind_contextvars("trace_id", "span_id")
