"""GraphRAG Core Configuration Module."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "graphrag-maintenance"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # LLM Provider
    llm_provider: Literal["gemini", "groq", "openrouter", "ollama"] = "gemini"

    # Gemini
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-1.5-pro"

    # Groq
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.1-70b-versatile"

    # OpenRouter
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:70b"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: SecretStr = SecretStr("")

    # PostgreSQL
    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://graphrag:password@localhost:5432/graphrag"
    )

    # Redis
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # S3/MinIO
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: SecretStr = SecretStr("")
    s3_secret_key: SecretStr = SecretStr("")
    s3_bucket: str = "documents"

    # Embedding
    embedding_model: str = "intfloat/e5-large-v2"
    embedding_dimension: int = 1024

    # Security
    secret_key: SecretStr = SecretStr("change-me-in-production")
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "graphrag-api"

    # Prefect
    prefect_api_url: str = "http://localhost:4200/api"

    # Retrieval Settings
    vector_top_k: int = 50
    rerank_top_k: int = 10
    graph_max_hops: int = 3
    entity_merge_threshold: float = 0.85
    entity_link_threshold: float = 0.70

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
