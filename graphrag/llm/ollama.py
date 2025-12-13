"""Ollama Local LLM Provider."""

import json
from typing import Any, AsyncIterator

from ollama import AsyncClient
from pydantic import BaseModel

from graphrag.llm.base import LLMConfig, LLMProvider, LLMResponse


class OllamaConfig(LLMConfig):
    """Ollama-specific configuration."""

    host: str = "http://localhost:11434"
    model_name: str = "llama3.1:70b"
    timeout: float = 120.0  # Longer timeout for local inference


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider implementation."""

    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self._config = config
        self.client = AsyncClient(host=config.host)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def supports_tools(self) -> bool:
        return False  # Limited tool support

    @property
    def supports_structured_output(self) -> bool:
        return True  # Via JSON mode

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Convert messages to Ollama format."""
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate chat completion using Ollama."""
        ollama_messages = self._convert_messages(messages)

        options = {
            "temperature": temperature or self.config.temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        response = await self.client.chat(
            model=self._config.model_name,
            messages=ollama_messages,
            options=options,
            **kwargs,
        )

        usage = {}
        if "prompt_eval_count" in response:
            usage = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
            }

        return LLMResponse(
            content=response["message"]["content"],
            model=response.get("model", self._config.model_name),
            provider=self.provider_name,
            usage=usage,
            finish_reason="stop" if response.get("done") else None,
        )

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output with JSON format."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_messages = messages.copy()

        schema_instruction = f"""
Respond with valid JSON matching this schema:
{schema_json}

Output ONLY the JSON object, no markdown or explanation.
"""

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

        ollama_messages = self._convert_messages(enhanced_messages)

        response = await self.client.chat(
            model=self._config.model_name,
            messages=ollama_messages,
            format="json",
            options={"temperature": temperature},
        )

        content = response["message"]["content"]
        return schema.model_validate_json(content)

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        ollama_messages = self._convert_messages(messages)

        stream = await self.client.chat(
            model=self._config.model_name,
            messages=ollama_messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]
