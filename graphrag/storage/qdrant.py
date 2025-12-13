"""Qdrant Vector Database Client."""

from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from graphrag.core.config import get_settings
from graphrag.core.models import VectorSearchResult


class QdrantClient:
    """Async Qdrant client for vector operations."""

    COLLECTION_NAME = "graphrag_chunks"

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
    ):
        """Initialize Qdrant client.

        Args:
            host: Qdrant host.
            port: Qdrant port.
            api_key: Optional API key.
        """
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        api_key = api_key or settings.qdrant_api_key.get_secret_value()

        self.client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=api_key if api_key else None,
        )
        self.dimension = settings.embedding_dimension

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.COLLECTION_NAME not in collection_names:
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
                on_disk_payload=True,
            )

            # Create payload indexes
            await self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="classification",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a chunk with its embedding.

        Args:
            chunk_id: Unique chunk ID.
            embedding: Vector embedding.
            doc_id: Parent document ID.
            content: Chunk text content.
            metadata: Additional metadata.
        """
        payload = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "content": content,
            **(metadata or {}),
        }

        await self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    async def upsert_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Batch upsert multiple chunks.

        Args:
            chunks: List of chunk dicts with id, embedding, doc_id, content, metadata.
        """
        points = [
            PointStruct(
                id=chunk["id"],
                vector=chunk["embedding"],
                payload={
                    "chunk_id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "content": chunk["content"],
                    **chunk.get("metadata", {}),
                },
            )
            for chunk in chunks
        ]

        await self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            filters: Metadata filters.
            score_threshold: Minimum score threshold.

        Returns:
            List of search results.
        """
        # Build filter conditions
        filter_conditions = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            if conditions:
                filter_conditions = models.Filter(must=conditions)

        # Use query method (newer API) instead of search
        results = await self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            VectorSearchResult(
                chunk_id=str(r.id),
                doc_id=r.payload.get("doc_id", r.payload.get("document_id", "")),
                content=r.payload.get("content", r.payload.get("text", "")),
                score=r.score,
                metadata={
                    k: v
                    for k, v in r.payload.items()
                    if k not in ("chunk_id", "doc_id", "content", "document_id", "text")
                },
            )
            for r in results.points
        ]

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID.
        """
        await self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Collection info dict.
        """
        info = await self.client.get_collection(self.COLLECTION_NAME)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }

    async def health_check(self) -> bool:
        """Check Qdrant connectivity.

        Returns:
            True if healthy.
        """
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False


# Singleton instance
_qdrant_client: QdrantClient | None = None


async def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient()
        await _qdrant_client.ensure_collection()
    return _qdrant_client
