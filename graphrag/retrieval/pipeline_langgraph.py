"""Session-based retrieval pipeline with conversation memory."""
from typing import List, Dict, Any
from pydantic import BaseModel
import uuid

# Import the working pipeline
from graphrag.retrieval.pipeline import run_full_retrieval as _run_pipeline_internal
from graphrag.retrieval.pipeline import RetrievalResult


# Simple in-memory session store
_session_store: Dict[str, List[Dict[str, str]]] = {}


async def run_full_retrieval(query: str, session_id: str) -> RetrievalResult:
    """Execute retrieval pipeline with session-based conversation memory.
    
    Args:
        query: User query
        session_id: Session identifier for conversation memory
        
    Returns:
        RetrievalResult with answer and metadata
    """
    # Initialize session if new
    if session_id not in _session_store:
        _session_store[session_id] = []
    
    # Get conversation history
    history = _session_store[session_id]
    
    # Enhance query with context from recent history (last 2 exchanges)
    enhanced_query = query
    if history:
        # Add context from previous exchange if query seems to reference it
        reference_words = ["he", "she", "it", "they", "his", "her", "their", "this", "that", "these", "those"]
        query_lower = query.lower()
        
        if any(word in query_lower.split() for word in reference_words):
            # Query likely references previous context
            recent = history[-2:] if len(history) >= 2 else history
            context = "\n".join([f"Previous: {ex['query']} -> {ex['answer'][:100]}" for ex in recent])
            enhanced_query = f"Context: {context}\n\nCurrent question: {query}"
    
    # Run the pipeline with enhanced query
    result = await _run_pipeline_internal(enhanced_query, session_id)
    
    # Store in session history
    _session_store[session_id].append({
        "query": query,
        "answer": result.answer,
        "intent": result.intent,
    })
    
    # Keep only last 10 exchanges per session
    if len(_session_store[session_id]) > 10:
        _session_store[session_id] = _session_store[session_id][-10:]
    
    return result


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of conversation exchanges
    """
    return _session_store.get(session_id, [])


def clear_session(session_id: str) -> None:
    """Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
    """
    if session_id in _session_store:
        del _session_store[session_id]

