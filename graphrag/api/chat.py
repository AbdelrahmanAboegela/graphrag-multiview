"""Simplified chat endpoint that works without full pipeline dependencies."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    session_id: Optional[str] = None


class RetrievalStep(BaseModel):
    """A step in the retrieval process for visualization."""
    stage: str
    description: str
    data: Dict[str, Any]
    duration_ms: int


class ChatResponse(BaseModel):
    """Chat response with retrieval visualization."""
    message: str
    sources: List[Dict[str, Any]]
    retrieval_steps: List[RetrievalStep]
    session_id: str


# Store for retrieval visualization
retrieval_traces: Dict[str, List[RetrievalStep]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message with retrieval visualization."""
    import time
    from sentence_transformers import SentenceTransformer
    from groq import AsyncGroq
    import os
    
    session_id = request.session_id or str(uuid.uuid4())
    retrieval_steps = []
    start_time = time.time()
    
    try:
        # Step 1: Query Understanding
        step_start = time.time()
        retrieval_steps.append(RetrievalStep(
            stage="Query Understanding",
            description=f"Received query: '{request.message}'",
            data={"query": request.message, "tokens": len(request.message.split())},
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 2: Generate Embedding
        step_start = time.time()
        model = SentenceTransformer('intfloat/e5-large-v2')
        query_embedding = model.encode(f"query: {request.message}").tolist()
        retrieval_steps.append(RetrievalStep(
            stage="Embedding Generation",
            description="Generated E5 query embedding (1024 dims)",
            data={"model": "e5-large-v2", "dimensions": 1024},
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 3: Vector Search
        step_start = time.time()
        import httpx
        
        # Use direct HTTP API to avoid client version incompatibility
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:6333/collections/graphrag_chunks/points/search",
                json={
                    "vector": query_embedding,
                    "limit": 5,
                    "with_payload": True
                }
            )
            response.raise_for_status()
            search_data = response.json()
        
        sources = []
        context_chunks = []
        
        for point in search_data.get("result", []):
            payload = point.get("payload", {})
            text = payload.get("text", payload.get("content", ""))
            doc_id = payload.get("document_id", payload.get("doc_id", "unknown"))
            sources.append({
                "chunk_id": str(point.get("id")),
                "document_id": doc_id,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "score": point.get("score", 0.0)
            })
            context_chunks.append(text)
        
        retrieval_steps.append(RetrievalStep(
            stage="Vector Search",
            description=f"Found {len(sources)} relevant chunks from Qdrant",
            data={
                "collection": "graphrag_chunks",
                "top_k": 5,
                "results": len(sources),
                "top_scores": [s["score"] for s in sources[:3]]
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 4: Context Assembly
        step_start = time.time()
        context = "\n\n---\n\n".join(context_chunks)
        retrieval_steps.append(RetrievalStep(
            stage="Context Assembly",
            description=f"Assembled {len(context_chunks)} chunks into context",
            data={"chunks": len(context_chunks), "total_chars": len(context)},
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Step 5: LLM Generation
        step_start = time.time()
        
        groq_client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        system_prompt = """You are a helpful oil & gas maintenance assistant. 
Answer questions based on the provided context. Be specific and cite the source when possible.
If the context doesn't contain relevant information, say so clearly."""

        response = await groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.message}"}
            ],
            max_tokens=1024,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        retrieval_steps.append(RetrievalStep(
            stage="LLM Generation",
            description=f"Generated answer using Groq {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')}",
            data={
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # Store trace for visualization
        retrieval_traces[session_id] = retrieval_steps
        
        return ChatResponse(
            message=answer,
            sources=sources,
            retrieval_steps=retrieval_steps,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trace/{session_id}")
async def get_trace(session_id: str) -> List[RetrievalStep]:
    """Get retrieval trace for a session."""
    if session_id not in retrieval_traces:
        raise HTTPException(status_code=404, detail="Trace not found")
    return retrieval_traces[session_id]
