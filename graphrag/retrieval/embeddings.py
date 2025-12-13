"""E5 Embedding Service for Vector Retrieval."""

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.core.config import get_settings


class E5EmbeddingService:
    """E5-large-v2 embedding service with query/passage formatting.
    
    E5 models require specific prefix formatting:
    - Queries: "query: {text}"
    - Passages: "passage: {text}"
    """

    def __init__(self, model_name: str | None = None):
        """Initialize embedding service.

        Args:
            model_name: Model name (default from settings).
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model with GPU support."""
        if self._model is None:
            import torch
            # Use GPU if available, fallback to CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model.max_seq_length = 512
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with E5 formatting.

        Args:
            query: Query text.

        Returns:
            Normalized embedding vector.
        """
        formatted = f"query: {query}"
        embedding = self.model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_passage(self, passage: str) -> list[float]:
        """Embed a document passage with E5 formatting.

        Args:
            passage: Passage text.

        Returns:
            Normalized embedding vector.
        """
        formatted = f"passage: {passage}"
        embedding = self.model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_queries_batch(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Batch embed multiple queries.

        Args:
            queries: List of query texts.
            batch_size: Batch size for encoding.

        Returns:
            List of embedding vectors.
        """
        formatted = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_passages_batch(
        self,
        passages: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Batch embed multiple passages for ingestion.

        Args:
            passages: List of passage texts.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            List of embedding vectors.
        """
        formatted = [f"passage: {p}" for p in passages]
        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()

    def cosine_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity score.
        """
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Singleton instance
_embedding_service: E5EmbeddingService | None = None


def get_embedding_service() -> E5EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = E5EmbeddingService()
    return _embedding_service
