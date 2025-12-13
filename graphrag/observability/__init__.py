"""Observability Package Exports."""

from graphrag.observability.logging import (
    setup_logging,
    get_logger,
    add_trace_context,
    clear_trace_context,
)
from graphrag.observability.metrics import (
    setup_metrics,
    record_retrieval_metrics,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    VECTOR_RECALL_COUNT,
    LLM_TOKENS_TOTAL,
    ANSWER_CONFIDENCE,
    SAFETY_ESCALATIONS,
)
from graphrag.observability.tracing import (
    setup_tracing,
    get_tracer,
    instrument_fastapi,
    traced,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "add_trace_context",
    "clear_trace_context",
    # Metrics
    "setup_metrics",
    "record_retrieval_metrics",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "VECTOR_RECALL_COUNT",
    "LLM_TOKENS_TOTAL",
    "ANSWER_CONFIDENCE",
    "SAFETY_ESCALATIONS",
    # Tracing
    "setup_tracing",
    "get_tracer",
    "instrument_fastapi",
    "traced",
]
