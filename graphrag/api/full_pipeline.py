"""Full pipeline chat endpoint using U-shaped retrieval with LangGraph."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

from graphrag.retrieval.pipeline_langgraph import run_full_retrieval
from graphrag.observability.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ANSWER_CONFIDENCE,
    CITATION_COUNT,
    VECTOR_RECALL_COUNT,
)

router = APIRouter(prefix="/api/v1/full", tags=["full-pipeline"])


class FullQueryRequest(BaseModel):
    """Full pipeline query request."""
    message: str
    session_id: Optional[str] = None


class FullQueryResponse(BaseModel):
    """Full pipeline query response."""
    message: str
    sources: List[Dict[str, Any]]
    retrieval_steps: List[Dict[str, Any]]
    intent: str
    confidence: float
    graph_facts: List[str]
    session_id: str


@router.post("/", response_model=FullQueryResponse)
async def query_full_pipeline(request: FullQueryRequest) -> FullQueryResponse:
    """Process query using complete U-shaped retrieval pipeline.
    
    Pipeline stages:
    1. Intent Classification
    2. Vector Search
    3. Graph Expansion (multi-view)
    4. Reranking
    5. Context Fusion
    6. LLM Generation
    """
    
    import uuid
    session_id = request.session_id or str(uuid.uuid4())
    
    # Start timing
    start_time = time.time()
    
    try:
        result = await run_full_retrieval(
            query=request.message,
            session_id=session_id
        )
        
        # Record metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/api/v1/full/").observe(latency)
        REQUEST_COUNT.labels(endpoint="/api/v1/full/", status="200").inc()
        ANSWER_CONFIDENCE.observe(result.confidence)
        CITATION_COUNT.observe(len(result.sources))
        
        # Record vector recall count from retrieval steps
        for step in result.retrieval_steps:
            if step.stage == "vector_search" and "count" in step.data:
                VECTOR_RECALL_COUNT.observe(step.data["count"])
        
        return FullQueryResponse(
            message=result.answer,
            sources=result.sources,
            retrieval_steps=[step.dict() for step in result.retrieval_steps],
            intent=result.intent,
            confidence=result.confidence,
            graph_facts=result.graph_facts,
            session_id=session_id
        )
        
    except Exception as e:
        # Record error
        REQUEST_COUNT.labels(endpoint="/api/v1/full/", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for full pipeline."""
    return {"status": "healthy", "pipeline": "full-u-shaped"}
