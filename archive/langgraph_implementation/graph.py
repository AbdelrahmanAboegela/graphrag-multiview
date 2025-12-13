"""U-Shaped Retrieval Pipeline - LangGraph Orchestration."""

import time
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graphrag.core.models import (
    Citation,
    ClassificationLevel,
    EvidencePack,
    ExtractedEntity,
    GraphFact,
    QueryIntent,
    RetrievalState,
    SafetyEscalation,
    VectorSearchResult,
)
from graphrag.retrieval.nodes import (
    answer_synthesizer_node,
    citation_generator_node,
    entity_extractor_node,
    evidence_assembler_node,
    graph_expander_node,
    guardrails_node,
    intent_classifier_node,
    query_normalizer_node,
    reranker_node,
    skip_fusion_node,
    vector_recall_node,
)


def evidence_sufficiency_check(
    state: RetrievalState,
) -> Literal["sufficient", "insufficient", "safety_critical"]:
    """Check if evidence is sufficient for answer synthesis.

    Args:
        state: Current retrieval state.

    Returns:
        Routing decision.
    """
    evidence = state.evidence_pack

    if evidence is None:
        return "insufficient"

    # Check for safety-critical content
    if evidence.safety_rules:
        # Check if query involves hazardous operations
        safety_keywords = ["bypass", "override", "shortcut", "skip", "disable"]
        query_lower = state.query.lower()
        if any(kw in query_lower for kw in safety_keywords):
            return "safety_critical"

    # Check evidence sufficiency
    total_evidence = len(evidence.passages) + len(evidence.graph_facts)

    if total_evidence < 2:
        return "insufficient"

    # Check citation coverage
    if not evidence.citations:
        return "insufficient"

    return "sufficient"


def guardrail_check(
    state: RetrievalState,
) -> Literal["pass", "escalate", "retry"]:
    """Check guardrails on generated answer.

    Args:
        state: Current retrieval state.

    Returns:
        Routing decision.
    """
    # Check for safety escalation trigger
    if state.safety_escalation:
        return "escalate"

    # Check confidence threshold
    if state.confidence < 0.3:
        return "retry"

    # Check citation coverage
    if not state.citations:
        return "retry"

    return "pass"


def create_retrieval_graph() -> StateGraph:
    """Create the U-shaped retrieval LangGraph.

    Returns:
        Compiled LangGraph for retrieval.
    """
    # Initialize graph with state schema
    graph = StateGraph(RetrievalState)

    # ========================================================================
    # DOWN PATH NODES (Broad → Narrow)
    # ========================================================================

    # Stage 1: Query normalization and intent classification
    graph.add_node("query_normalizer", query_normalizer_node)
    graph.add_node("intent_classifier", intent_classifier_node)

    # Stage 2: Broad vector recall
    graph.add_node("vector_recall", vector_recall_node)

    # Stage 3: Entity extraction from query + top results
    graph.add_node("entity_extractor", entity_extractor_node)

    # Stage 4: Multi-view graph expansion
    graph.add_node("graph_expander", graph_expander_node)

    # Stage 5: Reranking with hybrid scoring
    graph.add_node("reranker", reranker_node)

    # Stage 6: Evidence pack assembly
    graph.add_node("evidence_assembler", evidence_assembler_node)

    # ========================================================================
    # UP PATH NODES (Evidence Fusion → Answer)
    # ========================================================================

    # Stage 7: Skip connection fusion
    graph.add_node("skip_fusion", skip_fusion_node)

    # Stage 8: Answer synthesis with grounding
    graph.add_node("answer_synthesizer", answer_synthesizer_node)

    # Stage 9: Citation generation
    graph.add_node("citation_generator", citation_generator_node)

    # Stage 10: Guardrails check
    graph.add_node("guardrails", guardrails_node)

    # ========================================================================
    # CONDITIONAL / TERMINAL NODES
    # ========================================================================

    graph.add_node("insufficient_evidence", insufficient_evidence_node)
    graph.add_node("safety_escalation", safety_escalation_node)

    # ========================================================================
    # EDGES - Down Path
    # ========================================================================

    graph.add_edge("query_normalizer", "intent_classifier")
    graph.add_edge("intent_classifier", "vector_recall")
    graph.add_edge("vector_recall", "entity_extractor")
    graph.add_edge("entity_extractor", "graph_expander")
    graph.add_edge("graph_expander", "reranker")
    graph.add_edge("reranker", "evidence_assembler")

    # Evidence assembly → conditional routing
    graph.add_conditional_edges(
        "evidence_assembler",
        evidence_sufficiency_check,
        {
            "sufficient": "skip_fusion",
            "insufficient": "insufficient_evidence",
            "safety_critical": "safety_escalation",
        },
    )

    # ========================================================================
    # EDGES - Up Path
    # ========================================================================

    graph.add_edge("skip_fusion", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", "citation_generator")
    graph.add_edge("citation_generator", "guardrails")

    # Guardrails → conditional routing
    graph.add_conditional_edges(
        "guardrails",
        guardrail_check,
        {
            "pass": END,
            "escalate": "safety_escalation",
            "retry": "answer_synthesizer",  # Retry with more context
        },
    )

    # Terminal nodes
    graph.add_edge("insufficient_evidence", END)
    graph.add_edge("safety_escalation", END)

    # Set entry point
    graph.set_entry_point("query_normalizer")

    return graph


def insufficient_evidence_node(state: RetrievalState) -> dict[str, Any]:
    """Handle insufficient evidence case.

    Args:
        state: Current retrieval state.

    Returns:
        State update with fallback answer.
    """
    return {
        "answer": (
            "I don't have enough information in the available documentation to "
            "answer this question accurately. Please try:\n"
            "1. Rephrasing your question with more specific terms\n"
            "2. Specifying the asset type or equipment name\n"
            "3. Asking about a specific procedure or maintenance task\n\n"
            "If this is urgent, please contact your maintenance supervisor or SME."
        ),
        "confidence": 0.0,
        "citations": [],
    }


def safety_escalation_node(state: RetrievalState) -> dict[str, Any]:
    """Handle safety-critical escalation.

    Args:
        state: Current retrieval state.

    Returns:
        State update with safety escalation.
    """
    return {
        "answer": (
            "⚠️ **SAFETY NOTICE**\n\n"
            "Your query involves potentially hazardous operations that require "
            "human expert review. This system cannot provide guidance for:\n"
            "- Bypassing safety interlocks\n"
            "- Overriding protective systems\n"
            "- Shortcuts to standard procedures\n\n"
            "Please contact your HSE representative or maintenance supervisor "
            "for guidance on this matter."
        ),
        "confidence": 0.0,
        "safety_escalation": SafetyEscalation(
            reason="Query involves potentially hazardous operations",
            severity="HIGH",
            recommended_action="Contact HSE representative",
            contact_roles=["HSE Representative", "Maintenance Supervisor"],
        ),
    }


def compile_retrieval_graph():
    """Compile the retrieval graph with checkpointing.

    Returns:
        Compiled graph ready for execution.
    """
    graph = create_retrieval_graph()
    return graph.compile(checkpointer=MemorySaver())


# Singleton compiled graph
_retrieval_graph = None


def get_retrieval_graph():
    """Get or create retrieval graph singleton."""
    global _retrieval_graph
    if _retrieval_graph is None:
        _retrieval_graph = compile_retrieval_graph()
    return _retrieval_graph


async def run_retrieval(
    query: str,
    user_id: str = "anonymous",
    session_id: str | None = None,
    access_level: ClassificationLevel = ClassificationLevel.INTERNAL,
) -> RetrievalState:
    """Run the retrieval pipeline.

    Args:
        query: User query.
        user_id: User ID for access control.
        session_id: Optional session ID.
        access_level: User's access level.

    Returns:
        Final retrieval state with answer.
    """
    graph = get_retrieval_graph()

    # Initialize state
    initial_state = RetrievalState(
        query=query,
        user_id=user_id,
        session_id=session_id or str(uuid4()),
        access_level=access_level,
        trace_id=str(uuid4()),
        stage_timings={"start": time.time()},
    )

    # Run graph
    config = {"configurable": {"thread_id": initial_state.session_id}}
    final_state = await graph.ainvoke(initial_state, config)

    # Calculate total latency
    final_state["stage_timings"]["end"] = time.time()

    return RetrievalState(**final_state)
