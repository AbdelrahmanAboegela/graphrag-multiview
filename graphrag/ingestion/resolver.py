"""Document Ingestion Pipeline - Entity Resolution."""

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from graphrag.core.models import EntityType, ExtractedEntity, ResolvedEntity
from graphrag.retrieval.embeddings import get_embedding_service
from graphrag.storage import get_neo4j_client


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    action: str  # CREATE, MERGE, LINK
    entity_id: str
    canonical: ResolvedEntity | None
    link_to: str | None = None
    link_type: str | None = None
    confidence: float = 1.0
    reasoning: str = ""


class EntityResolver:
    """Entity resolution engine with normalization and similarity matching.

    Implements the entity resolution algorithm:
    1. Normalize entity names
    2. Find candidates via exact/fuzzy/embedding match
    3. Score candidates using context
    4. Decide: CREATE, MERGE, or LINK
    """

    def __init__(
        self,
        merge_threshold: float = 0.85,
        link_threshold: float = 0.70,
        embedding_weight: float = 0.6,
        context_weight: float = 0.4,
    ):
        """Initialize resolver.

        Args:
            merge_threshold: Score above which to merge entities.
            link_threshold: Score above which to create RELATED_TO link.
            embedding_weight: Weight for embedding similarity.
            context_weight: Weight for context similarity.
        """
        self.merge_threshold = merge_threshold
        self.link_threshold = link_threshold
        self.embedding_weight = embedding_weight
        self.context_weight = context_weight

        # Common abbreviations in oil & gas
        self.abbreviation_map = {
            "pm": "preventive maintenance",
            "cm": "corrective maintenance",
            "rca": "root cause analysis",
            "mtbf": "mean time between failures",
            "mttr": "mean time to repair",
            "sop": "standard operating procedure",
            "ppe": "personal protective equipment",
            "loto": "lockout tagout",
            "p&id": "piping and instrumentation diagram",
            "api": "american petroleum institute",
            "asme": "american society of mechanical engineers",
        }

    def normalize(self, entity: ExtractedEntity) -> dict[str, Any]:
        """Normalize entity name.

        Args:
            entity: Extracted entity.

        Returns:
            Normalized entity data.
        """
        name = entity.name.lower().strip()

        # Expand abbreviations
        words = name.split()
        expanded = [self.abbreviation_map.get(w, w) for w in words]
        name = " ".join(expanded)

        # Remove special characters (keep alphanumeric, space, hyphen)
        name = re.sub(r"[^\w\s\-]", "", name)

        # Normalize whitespace
        name = re.sub(r"\s+", " ", name).strip()

        # Type-specific normalization
        if entity.type == EntityType.ASSET:
            # Standardize asset tag format (e.g., P-101A)
            match = re.match(r"([a-z]+)\s*[-_]?\s*(\d+)\s*([a-z]?)", name)
            if match:
                prefix, number, suffix = match.groups()
                name = f"{prefix.upper()}-{number}{suffix.upper()}"

        # Get embedding for the normalized name
        embedding_service = get_embedding_service()
        embedding = embedding_service.embed_passage(name)

        return {
            "canonical_name": name,
            "original_name": entity.name,
            "type": entity.type,
            "embedding": embedding,
        }

    async def find_candidates(
        self,
        normalized: dict[str, Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find candidate entities for matching.

        Args:
            normalized: Normalized entity data.
            limit: Max candidates to return.

        Returns:
            List of candidate entities with scores.
        """
        neo4j = await get_neo4j_client()
        candidates = []

        # 1. Exact match
        query = """
        MATCH (n {canonical_name: $name})
        WHERE labels(n)[0] IN ['Asset', 'Component', 'FailureMode', 'Symptom', 
                               'Procedure', 'Tool', 'Material', 'SafetyRule', 
                               'Term', 'Standard', 'Location', 'Manufacturer']
        RETURN n.id AS id, n.canonical_name AS name, labels(n)[0] AS type,
               n.embedding AS embedding
        LIMIT 1
        """
        results = await neo4j.execute_query(
            query, {"name": normalized["canonical_name"]}
        )
        for r in results:
            candidates.append({**r, "match_type": "exact", "score": 1.0})

        if candidates:
            return candidates[:limit]

        # 2. Fuzzy match (Levenshtein distance < 2)
        query = """
        MATCH (n)
        WHERE labels(n)[0] IN ['Asset', 'Component', 'FailureMode', 'Symptom', 
                               'Procedure', 'Tool', 'Material', 'SafetyRule', 
                               'Term', 'Standard', 'Location', 'Manufacturer']
        AND apoc.text.levenshteinDistance(n.canonical_name, $name) < 3
        RETURN n.id AS id, n.canonical_name AS name, labels(n)[0] AS type,
               n.embedding AS embedding,
               apoc.text.levenshteinDistance(n.canonical_name, $name) AS distance
        ORDER BY distance
        LIMIT $limit
        """
        try:
            results = await neo4j.execute_query(
                query, {"name": normalized["canonical_name"], "limit": limit}
            )
            for r in results:
                score = 1.0 - (r["distance"] / max(len(normalized["canonical_name"]), 1))
                candidates.append({**r, "match_type": "fuzzy", "score": score})
        except Exception:
            pass  # APOC may not be available

        # 3. Embedding similarity (would require vector index in Neo4j or separate query)
        # For now, this is handled in the scoring step

        return candidates[:limit]

    def compute_embedding_similarity(
        self,
        embedding1: list[float] | None,
        embedding2: list[float] | None,
    ) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Similarity score (0-1).
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        embedding_service = get_embedding_service()
        return embedding_service.cosine_similarity(embedding1, embedding2)

    async def compute_context_similarity(
        self,
        normalized: dict[str, Any],
        candidate: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute context-based similarity.

        Args:
            normalized: Normalized entity.
            candidate: Candidate entity.
            context: Optional extraction context.

        Returns:
            Context similarity score (0-1).
        """
        score = 0.0

        # Type consistency check
        if normalized["type"].value == candidate.get("type", ""):
            score += 0.5

        # Co-occurrence check (would need full context)
        if context:
            # Check if candidate appears in same documents
            pass

        return min(score, 1.0)

    async def resolve(
        self,
        entity: ExtractedEntity,
        context: dict[str, Any] | None = None,
    ) -> ResolutionResult:
        """Resolve an extracted entity.

        Args:
            entity: Extracted entity.
            context: Optional extraction context.

        Returns:
            Resolution result with action and entity ID.
        """
        # Step 1: Normalize
        normalized = self.normalize(entity)

        # Step 2: Find candidates
        candidates = await self.find_candidates(normalized)

        if not candidates:
            # No candidates - create new entity
            entity_id = str(uuid4())
            return ResolutionResult(
                action="CREATE",
                entity_id=entity_id,
                canonical=ResolvedEntity(
                    id=entity_id,
                    canonical_name=normalized["canonical_name"],
                    original_names=[entity.name],
                    type=entity.type,
                    embedding=normalized["embedding"],
                ),
            )

        # Step 3: Score candidates
        scored = []
        for candidate in candidates:
            emb_score = self.compute_embedding_similarity(
                normalized["embedding"],
                candidate.get("embedding"),
            )
            ctx_score = await self.compute_context_similarity(
                normalized, candidate, context
            )

            final_score = (
                self.embedding_weight * emb_score
                + self.context_weight * ctx_score
            )

            scored.append((candidate, final_score, emb_score, ctx_score))

        # Step 4: Decision
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0]
        candidate, score, emb_score, ctx_score = best

        if score >= self.merge_threshold:
            return ResolutionResult(
                action="MERGE",
                entity_id=candidate["id"],
                canonical=None,  # Use existing
                confidence=score,
                reasoning=f"emb={emb_score:.2f}, ctx={ctx_score:.2f}",
            )
        elif score >= self.link_threshold:
            new_id = str(uuid4())
            return ResolutionResult(
                action="LINK",
                entity_id=new_id,
                canonical=ResolvedEntity(
                    id=new_id,
                    canonical_name=normalized["canonical_name"],
                    original_names=[entity.name],
                    type=entity.type,
                    embedding=normalized["embedding"],
                ),
                link_to=candidate["id"],
                link_type="RELATED_TO",
                confidence=score,
            )
        else:
            new_id = str(uuid4())
            return ResolutionResult(
                action="CREATE",
                entity_id=new_id,
                canonical=ResolvedEntity(
                    id=new_id,
                    canonical_name=normalized["canonical_name"],
                    original_names=[entity.name],
                    type=entity.type,
                    embedding=normalized["embedding"],
                ),
            )

    async def batch_resolve(
        self,
        entities: list[ExtractedEntity],
        context: dict[str, Any] | None = None,
    ) -> list[ResolutionResult]:
        """Resolve multiple entities.

        Args:
            entities: List of extracted entities.
            context: Optional extraction context.

        Returns:
            List of resolution results.
        """
        results = []
        for entity in entities:
            result = await self.resolve(entity, context)
            results.append(result)
        return results


# Singleton instance
_entity_resolver: EntityResolver | None = None


def get_entity_resolver() -> EntityResolver:
    """Get or create entity resolver singleton."""
    global _entity_resolver
    if _entity_resolver is None:
        _entity_resolver = EntityResolver()
    return _entity_resolver
