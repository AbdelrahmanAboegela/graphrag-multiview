"""LLM Provider Factory and Package Exports."""

from typing import Any

from pydantic import SecretStr

from graphrag.core.config import get_settings
from graphrag.llm.base import LLMConfig, LLMProvider, LLMResponse
from graphrag.llm.gemini import GeminiConfig, GeminiProvider
from graphrag.llm.groq import GroqConfig, GroqProvider
from graphrag.llm.ollama import OllamaConfig, OllamaProvider
from graphrag.llm.openrouter import OpenRouterConfig, OpenRouterProvider


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: dict[str, tuple[type[LLMProvider], type[LLMConfig]]] = {
        "gemini": (GeminiProvider, GeminiConfig),
        "groq": (GroqProvider, GroqConfig),
        "openrouter": (OpenRouterProvider, OpenRouterConfig),
        "ollama": (OllamaProvider, OllamaConfig),
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create an LLM provider instance.

        Args:
            provider_name: Name of the provider (gemini, groq, openrouter, ollama).
            config: Optional pre-built config object.
            **kwargs: Config parameters if config not provided.

        Returns:
            Configured LLM provider instance.

        Raises:
            ValueError: If provider name is unknown.
        """
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )

        provider_class, config_class = cls._providers[provider_name]

        if config is None:
            config = config_class(**kwargs)

        return provider_class(config)

    @classmethod
    def from_settings(cls) -> LLMProvider:
        """Create provider from application settings.

        Returns:
            LLM provider configured from environment variables.
        """
        settings = get_settings()
        provider = settings.llm_provider

        if provider == "gemini":
            return cls.create(
                "gemini",
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model,
            )
        elif provider == "groq":
            return cls.create(
                "groq",
                api_key=settings.groq_api_key,
                model_name=settings.groq_model,
            )
        elif provider == "openrouter":
            return cls.create(
                "openrouter",
                api_key=settings.openrouter_api_key,
                model_name=settings.openrouter_model,
            )
        elif provider == "ollama":
            return cls.create(
                "ollama",
                host=settings.ollama_host,
                model_name=settings.ollama_model,
            )
        else:
            raise ValueError(f"Unknown provider in settings: {provider}")


def get_llm_provider() -> LLMProvider:
    """Get LLM provider from settings (dependency injection helper).

    Returns:
        Configured LLM provider.
    """
    return LLMProviderFactory.from_settings()


__all__ = [
    # Base
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    # Providers
    "GeminiProvider",
    "GeminiConfig",
    "GroqProvider",
    "GroqConfig",
    "OpenRouterProvider",
    "OpenRouterConfig",
    "OllamaProvider",
    "OllamaConfig",
    # Factory
    "LLMProviderFactory",
    "get_llm_provider",
]
