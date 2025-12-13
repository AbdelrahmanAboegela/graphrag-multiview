"""U-Shaped Retrieval Pipeline Node Implementations."""

import hashlib
import time
from typing import Any

from graphrag.core.models import (
    Citation,
    EntityType,
    EvidencePack,
    ExtractedEntity,
    GraphFact,
    QueryIntent,
    RetrievalState,
    VectorSearchResult,
)
from graphrag.llm import get_llm_provider
from graphrag.retrieval.embeddings import get_embedding_service
from graphrag.storage import get_neo4j_client, get_qdrant_client


# ============================================================================
# DOWN PATH NODES
# ============================================================================


async def query_normalizer_node(state: RetrievalState) -> dict[str, Any]:
    """Normalize and clean the query.

    - Lowercase
    - Fix common typos
    - Expand abbreviations
    """
    query = state.query.strip()

    # Basic normalization
    normalized = query

    # Expand common abbreviations
    abbreviations = {
        "pm": "preventive maintenance",
        "cm": "corrective maintenance",
        "rca": "root cause analysis",
        "mtbf": "mean time between failures",
        "mttr": "mean time to repair",
        "sop": "standard operating procedure",
        "ppe": "personal protective equipment",
    }

    words = normalized.lower().split()
    expanded_words = [abbreviations.get(w, w) for w in words]
    normalized = " ".join(expanded_words)

    return {
        "normalized_query": normalized,
        "stage_timings": {
            **state.stage_timings,
            "query_normalizer": time.time(),
        },
    }


async def intent_classifier_node(state: RetrievalState) -> dict[str, Any]:
    """Classify query intent."""
    query = state.normalized_query.lower()

    # Rule-based intent classification
    intent = QueryIntent.PROCEDURE  # Default

    # Check for troubleshooting patterns
    troubleshooting_patterns = [
        "troubleshoot",
        "diagnose",
        "problem",
        "issue",
        "failure",
        "not working",
        "vibration",
        "noise",
        "leak",
        "overheating",
        "alarm",
        "fault",
        "error",
    ]
    if any(p in query for p in troubleshooting_patterns):
        intent = QueryIntent.TROUBLESHOOTING

    # Check for definition patterns
    definition_patterns = [
        "what is",
        "what are",
        "define",
        "meaning of",
        "explain",
        "definition",
    ]
    if any(p in query for p in definition_patterns):
        intent = QueryIntent.DEFINITION

    # Check for asset info patterns
    asset_patterns = [
        "specification",
        "specs",
        "rating",
        "capacity",
        "model",
        "manufacturer",
        "serial",
        "installed",
        "location",
    ]
    if any(p in query for p in asset_patterns):
        intent = QueryIntent.ASSET_INFO

    # Check for safety patterns
    safety_patterns = [
        "safety",
        "hazard",
        "ppe",
        "lockout",
        "tagout",
        "permit",
        "risk",
        "danger",
    ]
    if any(p in query for p in safety_patterns):
        intent = QueryIntent.SAFETY

    # Build metadata filters based on intent
    metadata_filters = {}
    if intent == QueryIntent.PROCEDURE:
        metadata_filters["doc_type"] = ["SOP", "OEM_MANUAL", "CHECKLIST"]
    elif intent == QueryIntent.TROUBLESHOOTING:
        metadata_filters["doc_type"] = ["OEM_MANUAL", "RCA", "BULLETIN"]
    elif intent == QueryIntent.DEFINITION:
        metadata_filters["doc_type"] = ["STANDARD", "OEM_MANUAL"]
    elif intent == QueryIntent.SAFETY:
        metadata_filters["doc_type"] = ["SOP", "STANDARD"]

    return {
        "intent": intent,
        "metadata_filters": metadata_filters,
        "stage_timings": {
            **state.stage_timings,
            "intent_classifier": time.time(),
        },
    }


async def vector_recall_node(state: RetrievalState) -> dict[str, Any]:
    """Broad vector retrieval using E5 embeddings."""
    embedding_service = get_embedding_service()
    qdrant = await get_qdrant_client()

    # Embed query with E5 formatting
    query_embedding = embedding_service.embed_query(state.normalized_query)

    # Build filters with access control
    filters = {**state.metadata_filters}
    filters["classification"] = [
        level.value
        for level in [
            state.access_level,
            *[l for l in list(ClassificationLevel) if l.value < state.access_level.value]
        ]
    ] if hasattr(state, 'access_level') else None

    # Remove None filters
    filters = {k: v for k, v in filters.items() if v is not None}

    # Search with broad recall (top-50)
    results = await qdrant.search(
        query_embedding=query_embedding,
        top_k=50,
        filters=filters if filters else None,
    )

    # Save skip connection (top-10 for later fusion)
    skip_vector = results[:10]

    return {
        "vector_candidates": results,
        "skip_vector": skip_vector,
        "stage_timings": {
            **state.stage_timings,
            "vector_recall": time.time(),
        },
    }


async def entity_extractor_node(state: RetrievalState) -> dict[str, Any]:
    """Extract entities from query and top results."""
    # Combine query with top passages for context
    context = state.normalized_query + "\n\n"
    for result in state.vector_candidates[:5]:
        context += result.content[:500] + "\n\n"

    # Use LLM for entity extraction
    llm = get_llm_provider()

    # Simple extraction prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Extract entities from the text. Return JSON with entities array. "
                "Each entity has: name, type (ASSET, COMPONENT, FAILURE_MODE, "
                "SYMPTOM, PROCEDURE, TOOL, SAFETY_RULE, TERM), confidence (0-1)."
            ),
        },
        {"role": "user", "content": context[:2000]},
    ]

    try:
        from graphrag.core.models import ExtractionOutput

        extraction = await llm.structured_output(
            messages=messages,
            schema=ExtractionOutput,
            temperature=0,
        )
        entities = extraction.entities
    except Exception:
        # Fallback to simple keyword extraction
        entities = []

        # Extract asset tags (pattern: letter(s)-number(s))
        import re

        asset_pattern = r"\b[A-Z]{1,3}-\d{2,4}[A-Z]?\b"
        for match in re.finditer(asset_pattern, context):
            entities.append(
                ExtractedEntity(
                    name=match.group(),
                    type=EntityType.ASSET,
                    confidence=0.9,
                )
            )

    return {
        "extracted_entities": entities,
        "stage_timings": {
            **state.stage_timings,
            "entity_extractor": time.time(),
        },
    }


async def graph_expander_node(state: RetrievalState) -> dict[str, Any]:
    """Expand context using multi-view graph traversal."""
    neo4j = await get_neo4j_client()

    graph_facts = []
    expanded_context = []

    # Define traversal config based on intent
    traversal_configs = {
        QueryIntent.PROCEDURE: {
            "edge_types": ["HAS_PROCEDURE", "HAS_STEP", "APPLIES_TO", "REQUIRES_TOOL"],
            "max_hops": 2,
        },
        QueryIntent.TROUBLESHOOTING: {
            "edge_types": ["INDICATES", "CAUSED_BY", "AFFECTS", "MITIGATES"],
            "max_hops": 3,
        },
        QueryIntent.DEFINITION: {
            "edge_types": ["DEFINED_AS", "SYNONYM_OF", "IS_A", "PART_OF"],
            "max_hops": 2,
        },
        QueryIntent.ASSET_INFO: {
            "edge_types": ["HAS_SUBSYSTEM", "HAS_COMPONENT", "INSTALLED_AT", "MADE_BY"],
            "max_hops": 2,
        },
        QueryIntent.SAFETY: {
            "edge_types": ["HAS_SAFETY_RULE", "MITIGATES", "REQUIRES"],
            "max_hops": 2,
        },
    }

    config = traversal_configs.get(
        state.intent,
        traversal_configs[QueryIntent.PROCEDURE],
    )

    # Traverse from each extracted entity
    for entity in state.extracted_entities[:10]:  # Limit entities
        try:
            # Find node by name
            node = await neo4j.get_node_by_id(entity.name)
            if not node:
                continue

            # Traverse from node
            paths = await neo4j.traverse_from_node(
                start_node_id=entity.name,
                edge_types=config["edge_types"],
                max_hops=config["max_hops"],
                limit=20,
            )

            for path in paths:
                # Extract fact from path
                nodes = path.get("nodes", [])
                rels = path.get("relationships", [])

                if len(nodes) >= 2:
                    fact_text = f"{nodes[0].get('name', nodes[0].get('id', ''))} "
                    if rels:
                        fact_text += f"{rels[0].replace('_', ' ').lower()} "
                    fact_text += f"{nodes[-1].get('name', nodes[-1].get('id', ''))}"

                    graph_facts.append(
                        GraphFact(
                            fact=fact_text,
                            source_nodes=[n.get("id", "") for n in nodes],
                            path=path,
                            confidence=0.9,
                        )
                    )
                    expanded_context.append(nodes[-1])

        except Exception:
            continue  # Skip failed traversals

    # Save skip connection (top-15 facts)
    skip_graph = graph_facts[:15]

    return {
        "graph_facts": graph_facts,
        "expanded_context": expanded_context,
        "skip_graph": skip_graph,
        "stage_timings": {
            **state.stage_timings,
            "graph_expander": time.time(),
        },
    }


async def reranker_node(state: RetrievalState) -> dict[str, Any]:
    """Rerank candidates using hybrid scoring."""

    def compute_hybrid_score(
        result: VectorSearchResult,
        graph_facts: list[GraphFact],
    ) -> float:
        """Compute hybrid score for a result."""
        # Vector score (already normalized 0-1)
        vector_score = result.score

        # Graph relevance: boost if entities in result appear in graph facts
        graph_relevance = 0.0
        result_content_lower = result.content.lower()
        for fact in graph_facts:
            for node_id in fact.source_nodes:
                if node_id.lower() in result_content_lower:
                    graph_relevance += 0.1
        graph_relevance = min(graph_relevance, 1.0)

        # Recency score (placeholder - would use metadata)
        recency_score = 0.5

        # Document authority based on type
        authority_map = {
            "OEM_MANUAL": 1.0,
            "STANDARD": 0.95,
            "SOP": 0.9,
            "BULLETIN": 0.85,
            "RCA": 0.8,
        }
        doc_type = result.metadata.get("doc_type", "")
        authority_score = authority_map.get(doc_type, 0.7)

        # Weighted combination
        hybrid = (
            0.35 * vector_score
            + 0.30 * graph_relevance
            + 0.15 * recency_score
            + 0.20 * authority_score
        )

        return hybrid

    # Score all candidates
    scored_results = []
    for result in state.vector_candidates:
        score = compute_hybrid_score(result, state.graph_facts)
        # Create new result with updated score
        scored_result = VectorSearchResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            content=result.content,
            score=score,
            metadata=result.metadata,
        )
        scored_results.append(scored_result)

    # Sort by hybrid score
    scored_results.sort(key=lambda x: x.score, reverse=True)

    # Take top-k
    reranked = scored_results[:10]

    # Save skip connection
    skip_rerank = reranked[:5]

    return {
        "reranked_evidence": reranked,
        "skip_rerank": skip_rerank,
        "stage_timings": {
            **state.stage_timings,
            "reranker": time.time(),
        },
    }


async def evidence_assembler_node(state: RetrievalState) -> dict[str, Any]:
    """Assemble evidence pack with citations and safety rules."""
    passages = state.reranked_evidence
    graph_facts_list = state.graph_facts

    # Build citations
    citations = []
    seen_docs = set()

    for passage in passages:
        if passage.doc_id not in seen_docs:
            citations.append(
                Citation(
                    doc_id=passage.doc_id,
                    doc_title=passage.metadata.get("title"),
                    section=passage.metadata.get("section"),
                    page=passage.metadata.get("page"),
                    excerpt=passage.content[:200] + "..."
                    if len(passage.content) > 200
                    else passage.content,
                    relevance_score=passage.score,
                )
            )
            seen_docs.add(passage.doc_id)

    # Extract safety rules (look for safety-related content)
    safety_rules = []
    safety_keywords = ["warning", "caution", "danger", "ppe", "lockout", "hazard"]

    for passage in passages:
        content_lower = passage.content.lower()
        if any(kw in content_lower for kw in safety_keywords):
            safety_rules.append(
                {
                    "content": passage.content,
                    "source": passage.doc_id,
                    "type": "extracted",
                }
            )

    # Detect conflicts (simplified - same topic, different content)
    conflicts = []

    evidence_pack = EvidencePack(
        passages=passages,
        graph_facts=graph_facts_list,
        safety_rules=safety_rules,
        citations=citations,
        conflicts=conflicts,
    )

    return {
        "evidence_pack": evidence_pack,
        "stage_timings": {
            **state.stage_timings,
            "evidence_assembler": time.time(),
        },
    }


# ============================================================================
# UP PATH NODES
# ============================================================================


async def skip_fusion_node(state: RetrievalState) -> dict[str, Any]:
    """Fuse skip connections from multiple stages."""
    fused = []
    seen_hashes = set()

    def content_hash(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:16]

    # Weight contributions from each skip connection
    weights = {
        "vector": 0.3,
        "graph": 0.4,
        "rerank": 0.3,
    }

    # Add vector skip connection (top passages from stage 2)
    for item in state.skip_vector:
        h = content_hash(item.content)
        if h not in seen_hashes:
            fused.append(
                {
                    "content": item.content,
                    "score": item.score * weights["vector"],
                    "source": "vector",
                    "doc_id": item.doc_id,
                }
            )
            seen_hashes.add(h)

    # Add graph skip connection (validated facts from stage 4)
    for fact in state.skip_graph:
        h = content_hash(fact.fact)
        if h not in seen_hashes:
            fused.append(
                {
                    "content": fact.fact,
                    "score": fact.confidence * weights["graph"],
                    "source": "graph",
                    "nodes": fact.source_nodes,
                }
            )
            seen_hashes.add(h)

    # Add rerank skip connection (top reranked from stage 5)
    for item in state.skip_rerank:
        h = content_hash(item.content)
        if h not in seen_hashes:
            fused.append(
                {
                    "content": item.content,
                    "score": item.score * weights["rerank"],
                    "source": "rerank",
                    "doc_id": item.doc_id,
                }
            )
            seen_hashes.add(h)

    # Sort by weighted score
    fused.sort(key=lambda x: x["score"], reverse=True)

    return {
        "fused_evidence": fused[:20],
        "stage_timings": {
            **state.stage_timings,
            "skip_fusion": time.time(),
        },
    }


async def answer_synthesizer_node(state: RetrievalState) -> dict[str, Any]:
    """Synthesize answer with grounding."""
    llm = get_llm_provider()

    # Build evidence context
    evidence_text = ""
    for i, item in enumerate(state.fused_evidence[:10], 1):
        source = item.get("doc_id", item.get("nodes", ["graph"])[0])
        evidence_text += f"[{i}] Source: {source}\n{item['content']}\n\n"

    # Build prompt
    system_prompt = """You are a maintenance assistant for oil & gas operations.
Generate a response based ONLY on the provided evidence.

RULES:
1. Only state facts that appear in the evidence
2. Cite every claim with [source number]
3. If evidence is insufficient, say "I don't have enough information"
4. Include all relevant safety warnings from evidence
5. For procedures, provide step-by-step guidance matching the source

INTENT: {intent}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                intent=state.intent.value if state.intent else "general"
            ),
        },
        {
            "role": "user",
            "content": f"QUERY: {state.query}\n\nEVIDENCE:\n{evidence_text}",
        },
    ]

    response = await llm.chat_completion(
        messages=messages,
        temperature=0.3,  # Low temperature for factual responses
    )

    # Estimate confidence based on evidence quality
    confidence = min(0.95, len(state.fused_evidence) * 0.1)

    return {
        "answer": response.content,
        "confidence": confidence,
        "stage_timings": {
            **state.stage_timings,
            "answer_synthesizer": time.time(),
        },
    }


async def citation_generator_node(state: RetrievalState) -> dict[str, Any]:
    """Generate structured citations."""
    # Use citations from evidence pack
    citations = []

    if state.evidence_pack:
        citations = state.evidence_pack.citations

    return {
        "citations": citations,
        "stage_timings": {
            **state.stage_timings,
            "citation_generator": time.time(),
        },
    }


async def guardrails_node(state: RetrievalState) -> dict[str, Any]:
    """Apply guardrails to the answer."""
    answer = state.answer

    # Check for potential hallucination indicators
    hallucination_phrases = [
        "i think",
        "probably",
        "might be",
        "i assume",
        "generally speaking",
        "in my experience",
    ]

    answer_lower = answer.lower()
    for phrase in hallucination_phrases:
        if phrase in answer_lower:
            # Add disclaimer
            answer = (
                answer
                + "\n\n*Note: Please verify this information with official documentation.*"
            )
            break

    # Check for safety-critical content without warnings
    if state.evidence_pack and state.evidence_pack.safety_rules:
        safety_mentioned = any(
            "warning" in answer_lower or "caution" in answer_lower
            or "safety" in answer_lower
        )
        if not safety_mentioned:
            # Inject safety reminder
            answer += (
                "\n\n⚠️ **Safety Reminder**: Always follow proper safety "
                "procedures and use required PPE when performing maintenance tasks."
            )

    return {
        "answer": answer,
        "stage_timings": {
            **state.stage_timings,
            "guardrails": time.time(),
        },
    }


# Import ClassificationLevel for vector_recall_node
from graphrag.core.models import ClassificationLevel
