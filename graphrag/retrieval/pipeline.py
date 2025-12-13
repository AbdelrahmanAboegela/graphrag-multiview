"""Complete U-shaped retrieval pipeline."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time
import uuid
import httpx
from sentence_transformers import SentenceTransformer

from graphrag.retrieval.intent_classifier import IntentClassifier, QueryIntent
from graphrag.retrieval.graph_expander import GraphExpander
from graphrag.retrieval.reranker import Reranker
from groq import AsyncGroq
import os


class RetrievalStep(BaseModel):
    """A step in the retrieval pipeline."""
    stage: str
    description: str
    data: Dict[str, Any]
    duration_ms: int


class RetrievalResult(BaseModel):
    """Complete retrieval result."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_steps: List[RetrievalStep]
    intent: QueryIntent
    confidence: float
    graph_facts: List[str]  # Changed to List[str] to match actual return value


async def run_full_retrieval(query: str, session_id: str) -> RetrievalResult:
    """Execute complete U-shaped retrieval pipeline.
    
    Pipeline stages:
    1. Intent Classification
    2. Vector Search (Qdrant)
    3. Graph Expansion (Neo4j multi-view)
    4. Reranking (LLM-based)
    5. Fact Extraction
    6. Context Fusion (skip connections)
    7. LLM Generation
    
    Args:
        query: User query
        session_id: Session identifier
        
    Returns:
        RetrievalResult with answer, sources, and pipeline trace
    """
    
    retrieval_steps = []
    start_time = time.time()
    
    # Initialize components
    intent_classifier = IntentClassifier()
    graph_expander = GraphExpander()
    reranker = Reranker()
    embedding_model = SentenceTransformer('intfloat/e5-large-v2')
    groq_client = AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    try:
        # ========================================================================
        # STAGE 1: Intent Classification
        # ========================================================================
        step_start = time.time()
        
        intent_result = await intent_classifier.classify(query)
        intent = intent_result.intent
        
        retrieval_steps.append(RetrievalStep(
            stage="1. Intent Classification",
            description=f"Classified as '{intent}' ({intent_result.reasoning})",
            data={
                "intent": intent,
                "confidence": intent_result.confidence,
                "reasoning": intent_result.reasoning
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # STAGE 2: Vector Search (Broad Recall)
        # ========================================================================
        step_start = time.time()
        
        query_embedding = embedding_model.encode(f"query: {query}").tolist()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:6333/collections/graphrag_chunks/points/search",
                json={
                    "vector": query_embedding,
                    "limit": 20,  # Broad recall
                    "with_payload": True
                }
            )
            response.raise_for_status()
            search_data = response.json()
        
        vector_candidates = []
        for point in search_data.get("result", []):
            payload = point.get("payload", {})
            vector_candidates.append({
                "chunk_id": str(point.get("id")),
                "content": payload.get("text", payload.get("content", "")),
                "score": point.get("score", 0.0),
                "document_id": payload.get("document_id", payload.get("doc_id", ""))
            })
        
        # Save skip connection
        skip_vector = vector_candidates[:5]
        
        retrieval_steps.append(RetrievalStep(
            stage="2. Vector Search",
            description=f"Retrieved {len(vector_candidates)} candidates from Qdrant",
            data={
                "candidates": len(vector_candidates),
                "top_scores": [c["score"] for c in vector_candidates[:3]],
                "embedding_model": "e5-large-v2"
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # STAGE 3: Graph Expansion (Multi-View)
        # ========================================================================
        step_start = time.time()
        
        chunk_ids = [c["chunk_id"] for c in vector_candidates[:10]]
        graph_context = await graph_expander.expand(
            seed_chunks=chunk_ids,
            intent=intent,
            max_hops=2
        )
        
        # Save skip connection
        skip_graph = graph_context.facts[:10]
        
        retrieval_steps.append(RetrievalStep(
            stage="3. Graph Expansion",
            description=f"Expanded to {len(graph_context.facts)} facts across {graph_context.total_hops} hops",
            data={
                "facts_count": len(graph_context.facts),
                "nodes_count": len(graph_context.nodes),
                "paths_count": len(graph_context.paths),
                "max_hops": graph_context.total_hops,
                "sample_facts": [f["fact"] for f in graph_context.facts[:3]]
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # STAGE 4: Reranking
        # ========================================================================
        step_start = time.time()
        
        reranked = await reranker.rerank(
            query=query,
            candidates=vector_candidates,
            top_k=10
        )
        
        # Save skip connection
        skip_rerank = [
            {
                "content": r.content,
                "score": r.score,
                "chunk_id": r.chunk_id
            }
            for r in reranked[:5]
        ]
        
        retrieval_steps.append(RetrievalStep(
            stage="4. Reranking",
            description=f"Reranked to top {len(reranked)} most relevant chunks",
            data={
                "reranked_count": len(reranked),
                "top_scores": [r.score for r in reranked[:3]],
                "method": "LLM-based relevance scoring"
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # STAGE 5: Context Fusion (Skip Connections)
        # ========================================================================
        step_start = time.time()
        
        # Fuse evidence from skip connections
        fused_evidence = []
        seen_content = set()
        
        # Add graph facts (highest weight)
        for fact in skip_graph:
            content = fact["fact"]
            if content not in seen_content:
                fused_evidence.append({
                    "content": content,
                    "source": "graph",
                    "weight": 0.4
                })
                seen_content.add(content)
        
        # Add reranked chunks
        for item in skip_rerank:
            content = item["content"][:300]  # Truncate
            if content not in seen_content:
                fused_evidence.append({
                    "content": content,
                    "source": "rerank",
                    "weight": 0.35
                })
                seen_content.add(content)
        
        # Add vector results
        for item in skip_vector:
            content = item["content"][:300]
            if content not in seen_content:
                fused_evidence.append({
                    "content": content,
                    "source": "vector",
                    "weight": 0.25
                })
                seen_content.add(content)
        
        retrieval_steps.append(RetrievalStep(
            stage="5. Context Fusion",
            description=f"Fused {len(fused_evidence)} evidence pieces from skip connections",
            data={
                "total_evidence": len(fused_evidence),
                "from_graph": len([e for e in fused_evidence if e["source"] == "graph"]),
                "from_rerank": len([e for e in fused_evidence if e["source"] == "rerank"]),
                "from_vector": len([e for e in fused_evidence if e["source"] == "vector"])
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # STAGE 6: LLM Generation
        # ========================================================================
        step_start = time.time()
        
        # Build context from fused evidence
        context_parts = []
        for i, evidence in enumerate(fused_evidence[:15], 1):
            context_parts.append(f"[{i}] {evidence['content']}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = f"""You are a helpful oil & gas maintenance assistant with access to a multi-view knowledge graph.

The user's query has been classified as: {intent}

Answer based on the provided context which includes:
- Document chunks (from maintenance manuals)
- Graph facts (from equipment, people, and maintenance records)
- Reranked evidence (most relevant information)

**Important guidelines:**
- Match your answer's detail level to the question's complexity
- For simple factual questions (who/what/where), provide direct, concise answers
- For complex questions (how-to/troubleshooting), provide comprehensive step-by-step guidance
- Only include context that directly answers the question - avoid tangential information
- Cite sources by number [1], [2], etc.
- If the context doesn't contain sufficient information, say so clearly

Think: Does this question need a paragraph or just a sentence?"""

        response = await groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=1024,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        retrieval_steps.append(RetrievalStep(
            stage="6. LLM Generation",
            description=f"Generated answer using Groq {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')}",
            data={
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "context_pieces": len(fused_evidence)
            },
            duration_ms=int((time.time() - step_start) * 1000)
        ))
        
        # ========================================================================
        # Prepare sources for citation
        # ========================================================================
        sources = []
        for item in reranked[:5]:
            sources.append({
                "chunk_id": item.chunk_id,
                "text": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                "score": item.score
            })
        
        # Calculate overall confidence
        avg_vector_score = sum(c["score"] for c in vector_candidates[:5]) / 5 if vector_candidates else 0
        avg_rerank_score = sum(r.score for r in reranked[:5]) / 5 if reranked else 0
        graph_coverage = min(len(graph_context.facts) / 10, 1.0)
        
        confidence = (0.3 * avg_vector_score + 0.4 * avg_rerank_score + 0.3 * graph_coverage)
        
        return RetrievalResult(
            answer=answer,
            sources=sources,
            retrieval_steps=retrieval_steps,
            intent=intent,
            confidence=confidence,
            graph_facts=[f["fact"] for f in graph_context.facts[:10]]
        )
        
    finally:
        await graph_expander.close()
