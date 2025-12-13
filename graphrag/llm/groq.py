"""Groq LLM Provider."""

import json
from typing import Any, AsyncIterator

from groq import AsyncGroq
from pydantic import BaseModel, SecretStr

from graphrag.llm.base import LLMConfig, LLMProvider, LLMResponse


class GroqConfig(LLMConfig):
    """Groq-specific configuration."""

    api_key: SecretStr
    model_name: str = "llama-3.1-70b-versatile"


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation."""

    def __init__(self, config: GroqConfig):
        super().__init__(config)
        self._config = config
        self.client = AsyncGroq(api_key=config.api_key.get_secret_value())

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True  # Via JSON mode

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate chat completion using Groq."""
        response = await self.client.chat.completions.create(
            model=self._config.model_name,
            messages=messages,  # type: ignore
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            request_id=response.id,
            finish_reason=choice.finish_reason,
        )

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output with JSON mode."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_messages = messages.copy()

        schema_instruction = f"""
Respond with valid JSON matching this schema:
{schema_json}

Output ONLY the JSON object, no markdown or explanation.
"""

        # Add schema to system message
        system_idx = next(
            (i for i, m in enumerate(enhanced_messages) if m["role"] == "system"),
            None,
        )

        if system_idx is not None:
            enhanced_messages[system_idx]["content"] += "\n\n" + schema_instruction
        else:
            enhanced_messages.insert(
                0, {"role": "system", "content": schema_instruction}
            )

        response = await self.client.chat.completions.create(
            model=self._config.model_name,
            messages=enhanced_messages,  # type: ignore
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return schema.model_validate_json(content)

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        stream = await self.client.chat.completions.create(
            model=self._config.model_name,
            messages=messages,  # type: ignore
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
