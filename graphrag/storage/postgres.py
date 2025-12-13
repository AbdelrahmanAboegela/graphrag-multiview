"""PostgreSQL Database Client for Metadata and Audit."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from graphrag.core.config import get_settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


# ============================================================================
# ORM Models
# ============================================================================


class DocumentRecord(Base):
    """Document metadata record."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    doc_type: Mapped[str] = mapped_column(String(50))
    version: Mapped[str | None] = mapped_column(String(50))
    classification: Mapped[str] = mapped_column(String(20), default="internal")
    source_hash: Mapped[str] = mapped_column(String(64), unique=True)
    source_path: Mapped[str | None] = mapped_column(String(1000))
    supersedes: Mapped[str | None] = mapped_column(String(36))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    effective_date: Mapped[datetime | None] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), server_default=func.now()
    )
    metadata_json: Mapped[dict | None] = mapped_column(JSON)


class AuditQuery(Base):
    """Audit log for queries."""

    __tablename__ = "audit_queries"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), server_default=func.now()
    )
    user_id: Mapped[str] = mapped_column(String(100))
    session_id: Mapped[str | None] = mapped_column(String(100))
    query_text: Mapped[str] = mapped_column(Text)
    query_hash: Mapped[str | None] = mapped_column(String(64))
    intent: Mapped[str | None] = mapped_column(String(50))

    # Retrieval details
    vector_results_count: Mapped[int | None] = mapped_column(Integer)
    graph_hops: Mapped[int | None] = mapped_column(Integer)
    retrieved_doc_ids: Mapped[list[str] | None] = mapped_column(ARRAY(String))

    # Response details
    answer_text: Mapped[str | None] = mapped_column(Text)
    citations_json: Mapped[list | None] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)
    safety_escalation: Mapped[bool] = mapped_column(default=False)

    # Provider details
    llm_provider: Mapped[str | None] = mapped_column(String(50))
    llm_model: Mapped[str | None] = mapped_column(String(100))
    prompt_template_version: Mapped[str | None] = mapped_column(String(50))

    # Performance
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_usage: Mapped[dict | None] = mapped_column(JSON)

    # Tracing
    trace_id: Mapped[str | None] = mapped_column(String(100))


class AuditDocumentAccess(Base):
    """Audit log for document access."""

    __tablename__ = "audit_document_access"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), server_default=func.now()
    )
    user_id: Mapped[str] = mapped_column(String(100))
    document_id: Mapped[str] = mapped_column(String(100))
    section_ids: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    access_type: Mapped[str] = mapped_column(String(20))  # retrieval, view, download
    access_granted: Mapped[bool] = mapped_column(default=True)
    denial_reason: Mapped[str | None] = mapped_column(String(200))


# ============================================================================
# Database Client
# ============================================================================


class PostgresClient:
    """Async PostgreSQL client for metadata and audit."""

    def __init__(self, database_url: str | None = None):
        """Initialize PostgreSQL client.

        Args:
            database_url: Database connection URL.
        """
        settings = get_settings()
        url = database_url or settings.database_url.get_secret_value()

        self.engine = create_async_engine(
            url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get a database session.

        Yields:
            AsyncSession.
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ========================================================================
    # Document Operations
    # ========================================================================

    async def create_document(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        source_hash: str,
        **kwargs: Any,
    ) -> DocumentRecord:
        """Create a document record.

        Args:
            doc_id: Document ID.
            title: Document title.
            doc_type: Document type.
            source_hash: Content hash.
            **kwargs: Additional fields.

        Returns:
            Created document record.
        """
        doc = DocumentRecord(
            id=doc_id,
            title=title,
            doc_type=doc_type,
            source_hash=source_hash,
            **kwargs,
        )

        async with self.session() as session:
            session.add(doc)
            await session.flush()
            return doc

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Get document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document record or None.
        """
        async with self.session() as session:
            return await session.get(DocumentRecord, doc_id)

    async def document_exists_by_hash(self, source_hash: str) -> str | None:
        """Check if document exists by hash.

        Args:
            source_hash: Content hash.

        Returns:
            Document ID if exists, None otherwise.
        """
        from sqlalchemy import select

        async with self.session() as session:
            result = await session.execute(
                select(DocumentRecord.id).where(
                    DocumentRecord.source_hash == source_hash
                )
            )
            row = result.scalar_one_or_none()
            return row

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        """Update document status.

        Args:
            doc_id: Document ID.
            status: New status.
            chunk_count: Optional chunk count.
        """
        from sqlalchemy import update

        async with self.session() as session:
            stmt = (
                update(DocumentRecord)
                .where(DocumentRecord.id == doc_id)
                .values(
                    status=status,
                    **({"chunk_count": chunk_count} if chunk_count else {}),
                )
            )
            await session.execute(stmt)

    # ========================================================================
    # Audit Operations
    # ========================================================================

    async def log_query(
        self,
        user_id: str,
        query_text: str,
        **kwargs: Any,
    ) -> str:
        """Log a query for audit.

        Args:
            user_id: User ID.
            query_text: Query text.
            **kwargs: Additional audit fields.

        Returns:
            Audit record ID.
        """
        import hashlib

        audit = AuditQuery(
            user_id=user_id,
            query_text=query_text,
            query_hash=hashlib.sha256(query_text.encode()).hexdigest()[:16],
            **kwargs,
        )

        async with self.session() as session:
            session.add(audit)
            await session.flush()
            return audit.id

    async def log_document_access(
        self,
        user_id: str,
        document_id: str,
        access_type: str,
        access_granted: bool = True,
        **kwargs: Any,
    ) -> str:
        """Log document access.

        Args:
            user_id: User ID.
            document_id: Document ID.
            access_type: Access type.
            access_granted: Whether access was granted.
            **kwargs: Additional fields.

        Returns:
            Audit record ID.
        """
        audit = AuditDocumentAccess(
            user_id=user_id,
            document_id=document_id,
            access_type=access_type,
            access_granted=access_granted,
            **kwargs,
        )

        async with self.session() as session:
            session.add(audit)
            await session.flush()
            return audit.id

    async def health_check(self) -> bool:
        """Check database connectivity.

        Returns:
            True if healthy.
        """
        try:
            async with self.session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception:
            return False


# Singleton instance
_postgres_client: PostgresClient | None = None


async def get_postgres_client() -> PostgresClient:
    """Get or create PostgreSQL client singleton."""
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient()
        await _postgres_client.create_tables()
    return _postgres_client
