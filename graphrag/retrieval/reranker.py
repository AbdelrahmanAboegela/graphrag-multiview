"""Reranking module for evidence refinement."""
from typing import List, Dict, Any
from pydantic import BaseModel
from groq import AsyncGroq
import os


class RankedChunk(BaseModel):
    """Reranked chunk with score."""
    chunk_id: str
    content: str
    score: float
    original_score: float


class Reranker:
    """Rerank candidates using LLM-based relevance scoring."""
    
    def __init__(self):
        self.client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[RankedChunk]:
        """Rerank candidates using LLM relevance scoring.
        
        Args:
            query: User query
            candidates: List of candidate chunks with content and scores
            top_k: Number of top results to return
            
        Returns:
            List of reranked chunks
        """
        
        if not candidates:
            return []
        
        # Batch reranking using LLM
        reranked = []
        
        for candidate in candidates[:20]:  # Limit to top 20 for reranking
            content = candidate.get("content", candidate.get("text", ""))
            original_score = candidate.get("score", 0.0)
            chunk_id = candidate.get("chunk_id", candidate.get("id", ""))
            
            # Score relevance using LLM
            relevance_score = await self._score_relevance(query, content)
            
            # Combine original vector score with LLM relevance
            final_score = 0.4 * original_score + 0.6 * relevance_score
            
            reranked.append(RankedChunk(
                chunk_id=str(chunk_id),
                content=content,
                score=final_score,
                original_score=original_score
            ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:top_k]
    
    async def _score_relevance(self, query: str, content: str) -> float:
        """Score relevance of content to query using LLM.
        
        Args:
            query: User query
            content: Content to score
            
        Returns:
            Relevance score 0.0-1.0
        """
        
        system_prompt = """You are a relevance scorer for maintenance documentation.

Score how relevant the given content is to answering the user's query.

Return a JSON object with:
{
  "score": 0.0-1.0,
  "reasoning": "brief explanation"
}

Scoring guidelines:
- 1.0: Directly answers the question with specific details
- 0.7-0.9: Highly relevant, contains key information
- 0.4-0.6: Somewhat relevant, provides context
- 0.1-0.3: Tangentially related
- 0.0: Not relevant"""

        try:
            response = await self.client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContent: {content[:500]}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=100
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return float(result.get("score", 0.5))
            
        except Exception as e:
            # Fallback to simple keyword matching
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            return min(overlap / max(len(query_words), 1), 1.0)
