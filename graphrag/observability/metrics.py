"""Observability - Prometheus Metrics."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response


# ============================================================================
# Request Metrics
# ============================================================================

REQUEST_COUNT = Counter(
    "graphrag_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "graphrag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
)


# ============================================================================
# Retrieval Metrics
# ============================================================================

VECTOR_RECALL_COUNT = Histogram(
    "graphrag_vector_recall_count",
    "Number of vector results retrieved",
    buckets=[1, 5, 10, 25, 50, 100],
)

GRAPH_HOPS_COUNT = Histogram(
    "graphrag_graph_hops",
    "Number of graph traversal hops",
    ["view"],
    buckets=[1, 2, 3, 4, 5],
)

RETRIEVAL_STAGE_LATENCY = Histogram(
    "graphrag_retrieval_stage_latency_seconds",
    "Latency per retrieval stage",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)


# ============================================================================
# LLM Metrics
# ============================================================================

LLM_TOKENS_TOTAL = Counter(
    "graphrag_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "model", "type"],  # type: input/output
)

LLM_REQUESTS_TOTAL = Counter(
    "graphrag_llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)

LLM_LATENCY = Histogram(
    "graphrag_llm_latency_seconds",
    "LLM request latency",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


# ============================================================================
# Quality Metrics
# ============================================================================

ANSWER_CONFIDENCE = Histogram(
    "graphrag_answer_confidence",
    "Answer confidence score",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CITATION_COUNT = Histogram(
    "graphrag_citation_count",
    "Number of citations per answer",
    buckets=[0, 1, 2, 3, 5, 10],
)

SAFETY_ESCALATIONS = Counter(
    "graphrag_safety_escalations_total",
    "Total safety escalations triggered",
)


# ============================================================================
# System Metrics
# ============================================================================

ACTIVE_CONNECTIONS = Gauge(
    "graphrag_active_connections",
    "Number of active connections",
    ["database"],
)

INGESTION_QUEUE_SIZE = Gauge(
    "graphrag_ingestion_queue_size",
    "Number of documents in ingestion queue",
)


# ============================================================================
# Setup
# ============================================================================


def setup_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics endpoint.

    Args:
        app: FastAPI application.
    """

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


def record_retrieval_metrics(
    vector_count: int,
    graph_hops: int,
    stage_timings: dict[str, float],
    confidence: float,
    citation_count: int,
) -> None:
    """Record metrics for a retrieval operation.

    Args:
        vector_count: Number of vector results.
        graph_hops: Number of graph facts.
        stage_timings: Timing per stage.
        confidence: Answer confidence.
        citation_count: Number of citations.
    """
    VECTOR_RECALL_COUNT.observe(vector_count)
    ANSWER_CONFIDENCE.observe(confidence)
    CITATION_COUNT.observe(citation_count)

    # Record stage latencies
    stages = [
        "query_normalizer",
        "intent_classifier",
        "vector_recall",
        "entity_extractor",
        "graph_expander",
        "reranker",
        "evidence_assembler",
        "skip_fusion",
        "answer_synthesizer",
        "citation_generator",
        "guardrails",
    ]

    prev_time = stage_timings.get("start", 0)
    for stage in stages:
        if stage in stage_timings:
            latency = stage_timings[stage] - prev_time
            RETRIEVAL_STAGE_LATENCY.labels(stage=stage).observe(latency)
            prev_time = stage_timings[stage]
