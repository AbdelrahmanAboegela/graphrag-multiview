"""GraphRAG Core Domain Models and Schemas."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class DocumentType(str, Enum):
    """Document classification types."""

    OEM_MANUAL = "OEM_MANUAL"
    SOP = "SOP"
    CHECKLIST = "CHECKLIST"
    RCA = "RCA"
    BULLETIN = "BULLETIN"
    STANDARD = "STANDARD"
    INSPECTION_LOG = "INSPECTION_LOG"


class ClassificationLevel(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class QueryIntent(str, Enum):
    """Query intent classification."""

    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    DEFINITION = "definition"
    ASSET_INFO = "asset_info"
    SAFETY = "safety"


class EntityType(str, Enum):
    """Entity types for extraction."""

    ASSET = "ASSET"
    COMPONENT = "COMPONENT"
    FAILURE_MODE = "FAILURE_MODE"
    SYMPTOM = "SYMPTOM"
    PROCEDURE = "PROCEDURE"
    TOOL = "TOOL"
    MATERIAL = "MATERIAL"
    SAFETY_RULE = "SAFETY_RULE"
    TERM = "TERM"
    STANDARD = "STANDARD"
    LOCATION = "LOCATION"
    MANUFACTURER = "MANUFACTURER"


class RelationType(str, Enum):
    """Relation types for extraction."""

    APPLIES_TO = "APPLIES_TO"
    REQUIRES = "REQUIRES"
    INDICATES = "INDICATES"
    CAUSED_BY = "CAUSED_BY"
    MITIGATES = "MITIGATES"
    PART_OF = "PART_OF"
    LOCATED_AT = "LOCATED_AT"
    MADE_BY = "MADE_BY"
    HAS_STEP = "HAS_STEP"
    REFERENCES = "REFERENCES"


# ============================================================================
# Document Models
# ============================================================================


class DocumentMetadata(BaseModel):
    """Document metadata for ingestion."""

    doc_type: DocumentType
    title: str
    version: str | None = None
    effective_date: datetime | None = None
    supersedes: str | None = None
    classification: ClassificationLevel = ClassificationLevel.INTERNAL
    source: str | None = None


class Chunk(BaseModel):
    """Document chunk with embedding."""

    id: str
    doc_id: str
    content: str
    sequence: int
    chunk_type: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Entity & Relation Models
# ============================================================================


class ExtractedEntity(BaseModel):
    """Entity extracted from text."""

    name: str
    type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    context: str = ""
    source_chunk_id: str | None = None


class ExtractedRelation(BaseModel):
    """Relation extracted from text."""

    subject: str
    predicate: RelationType
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str | None = None


class ExtractionOutput(BaseModel):
    """Combined extraction output."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class ResolvedEntity(BaseModel):
    """Entity after resolution/canonicalization."""

    id: str
    canonical_name: str
    original_names: list[str] = Field(default_factory=list)
    type: EntityType
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Retrieval Models
# ============================================================================


class VectorSearchResult(BaseModel):
    """Result from vector search."""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphFact(BaseModel):
    """Fact retrieved from graph traversal."""

    fact: str
    source_nodes: list[str] = Field(default_factory=list)
    path: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0


class Citation(BaseModel):
    """Citation for answer grounding."""

    doc_id: str
    doc_title: str | None = None
    section: str | None = None
    page: int | None = None
    excerpt: str
    relevance_score: float = 1.0


class EvidencePack(BaseModel):
    """Assembled evidence for answer synthesis."""

    passages: list[VectorSearchResult] = Field(default_factory=list)
    graph_facts: list[GraphFact] = Field(default_factory=list)
    safety_rules: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


class SafetyEscalation(BaseModel):
    """Safety escalation trigger."""

    reason: str
    severity: str
    recommended_action: str
    contact_roles: list[str] = Field(default_factory=list)


# ============================================================================
# Query & Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Incoming query request."""

    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    include_trace: bool = False


class QueryResponse(BaseModel):
    """Query response with answer and citations."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    intent: QueryIntent | None = None
    safety_escalation: SafetyEscalation | None = None
    trace_id: str | None = None
    latency_ms: int | None = None


# ============================================================================
# Retrieval State (LangGraph)
# ============================================================================


class RetrievalState(BaseModel):
    """State object for LangGraph retrieval pipeline."""

    # Input
    query: str
    user_id: str = "anonymous"
    session_id: str = ""
    access_level: ClassificationLevel = ClassificationLevel.INTERNAL
    trace_id: str = ""

    # Down Path State
    normalized_query: str = ""
    intent: QueryIntent | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)

    # Stage 2: Vector Recall
    vector_candidates: list[VectorSearchResult] = Field(default_factory=list)

    # Stage 3: Entity Extraction
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)

    # Stage 4: Graph Expansion
    graph_facts: list[GraphFact] = Field(default_factory=list)
    expanded_context: list[dict[str, Any]] = Field(default_factory=list)

    # Stage 5: Reranking
    reranked_evidence: list[VectorSearchResult] = Field(default_factory=list)

    # Stage 6: Evidence Pack
    evidence_pack: EvidencePack | None = None

    # Up Path State (Skip Connections)
    skip_vector: list[VectorSearchResult] = Field(default_factory=list)
    skip_graph: list[GraphFact] = Field(default_factory=list)
    skip_rerank: list[VectorSearchResult] = Field(default_factory=list)

    # Final Output
    fused_evidence: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    safety_escalation: SafetyEscalation | None = None

    # Debugging
    stage_timings: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
