"""Abstract LLM Provider Interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from uuid import uuid4

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    finish_reason: str | None = None


class LLMConfig(BaseModel):
    """Base LLM configuration."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3


class LLMProvider(ABC):
    """Abstract LLM provider interface.
    
    All LLM providers must implement this interface to ensure
    consistent behavior across Gemini, Groq, OpenRouter, and Ollama.
    """

    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration."""
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name identifier."""
        pass

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether provider supports function/tool calling."""
        pass

    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether provider supports JSON schema output."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            **kwargs: Provider-specific parameters.
            
        Returns:
            Standardized LLMResponse.
        """
        pass

    @abstractmethod
    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output matching a Pydantic schema.
        
        Args:
            messages: List of message dicts.
            schema: Pydantic model class for expected output.
            temperature: Temperature (default 0 for determinism).
            **kwargs: Provider-specific parameters.
            
        Returns:
            Instance of the schema model.
        """
        pass

    @abstractmethod
    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.
        
        Args:
            messages: List of message dicts.
            **kwargs: Provider-specific parameters.
            
        Yields:
            Token strings as they are generated.
        """
        pass

    async def health_check(self) -> bool:
        """Check if provider is available.
        
        Returns:
            True if provider is healthy.
        """
        try:
            response = await self.chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception:
            return False
