"""FastAPI Application - GraphRAG Maintenance Chatbot."""

from contextlib import asynccontextmanager
from typing import Any
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import structlog

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from graphrag.core.config import get_settings
from graphrag.core.models import QueryRequest, QueryResponse
from graphrag.storage import get_neo4j_client, get_qdrant_client, get_postgres_client
from graphrag.observability.logging import setup_logging
from graphrag.observability.metrics import setup_metrics, REQUEST_LATENCY, REQUEST_COUNT
from graphrag.api import chat
from graphrag.api import full_pipeline

# Configure logging - suppress noisy libraries
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# Setup structured logging
setup_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()

    # Startup
    logger.info("Starting GraphRAG API", env=settings.app_env)

    # Try to initialize database connections (don't fail if unavailable)
    try:
        neo4j = await get_neo4j_client()
        logger.info("Neo4j connected")
    except Exception as e:
        logger.warning("Neo4j not available", error=str(e))

    try:
        qdrant = await get_qdrant_client()
        logger.info("Qdrant connected")
    except Exception as e:
        logger.warning("Qdrant not available", error=str(e))

    logger.info("GraphRAG API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down GraphRAG API")


# Create FastAPI app
app = FastAPI(
    title="GraphRAG Maintenance Chatbot",
    description="Production-grade GraphRAG for oil & gas maintenance operations",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(full_pipeline.router)

# Setup Prometheus metrics
setup_metrics(app)


# ============================================================================
# Health Endpoints
# ============================================================================


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    """Readiness probe - checks all dependencies."""
    checks = {}

    try:
        neo4j = await get_neo4j_client()
        checks["neo4j"] = await neo4j.health_check()
    except Exception:
        checks["neo4j"] = False

    try:
        qdrant = await get_qdrant_client()
        checks["qdrant"] = await qdrant.health_check()
    except Exception:
        checks["qdrant"] = False

    try:
        postgres = await get_postgres_client()
        checks["postgres"] = await postgres.health_check()
    except Exception:
        checks["postgres"] = False

    all_healthy = all(checks.values())

    if not all_healthy:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "checks": checks})

    return {"status": "ready", "checks": checks}


# ============================================================================
# Query Endpoints
# ============================================================================


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a maintenance query.

    Args:
        request: Query request with question and optional filters.

    Returns:
        Response with answer, citations, and metadata.
    """
    import time

    start_time = time.time()

    logger.info(
        "Processing query",
        query=request.query[:100],
        session_id=request.session_id,
    )

    try:
        # Run retrieval pipeline
        result = await run_retrieval(
            query=request.query,
            session_id=request.session_id,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Log audit
        postgres = await get_postgres_client()
        await postgres.log_query(
            user_id="anonymous",  # TODO: Get from auth
            query_text=request.query,
            intent=result.intent.value if result.intent else None,
            vector_results_count=len(result.vector_candidates),
            graph_hops=len(result.graph_facts),
            answer_text=result.answer,
            confidence=result.confidence,
            latency_ms=latency_ms,
            trace_id=result.trace_id,
        )

        # Record metrics
        REQUEST_COUNT.labels(
            endpoint="/api/v1/query",
            status="success",
        ).inc()
        REQUEST_LATENCY.labels(
            endpoint="/api/v1/query",
        ).observe(latency_ms / 1000)

        return QueryResponse(
            query=request.query,
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            intent=result.intent,
            safety_escalation=result.safety_escalation,
            trace_id=result.trace_id if request.include_trace else None,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        REQUEST_COUNT.labels(
            endpoint="/api/v1/query",
            status="error",
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/query/{trace_id}/trace")
async def get_query_trace(trace_id: str) -> dict[str, Any]:
    """Get detailed trace for a query (debugging).

    Args:
        trace_id: Trace ID from query response.

    Returns:
        Detailed trace information.
    """
    # TODO: Implement trace retrieval from audit logs
    return {"trace_id": trace_id, "stages": [], "message": "Not implemented"}


# ============================================================================
# Document Endpoints
# ============================================================================


@app.post("/api/v1/ingest")
async def ingest_document(
    # file: UploadFile,
    # metadata: DocumentMetadata,
) -> dict[str, Any]:
    """Ingest a new document.

    Returns:
        Ingestion status.
    """
    # TODO: Implement document ingestion
    return {"status": "not_implemented", "message": "Use ingestion worker"}


@app.get("/api/v1/documents")
async def list_documents(
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List ingested documents.

    Args:
        limit: Max results.
        offset: Pagination offset.

    Returns:
        Document list.
    """
    # TODO: Implement document listing
    return {"documents": [], "total": 0, "limit": limit, "offset": offset}


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the application."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "graphrag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
