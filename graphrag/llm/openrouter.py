"""OpenRouter LLM Provider."""

import json
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel, SecretStr

from graphrag.llm.base import LLMConfig, LLMProvider, LLMResponse


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration."""

    api_key: SecretStr
    model_name: str = "anthropic/claude-3.5-sonnet"
    base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = ""
    app_name: str = "graphrag-maintenance"


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider implementation."""

    def __init__(self, config: OpenRouterConfig):
        super().__init__(config)
        self._config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key.get_secret_value()}",
                "HTTP-Referer": config.site_url,
                "X-Title": config.app_name,
            },
            timeout=config.timeout,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def supports_tools(self) -> bool:
        # Depends on underlying model
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate chat completion via OpenRouter."""
        payload = {
            "model": self._config.model_name,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs,
        }

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._config.model_name),
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            request_id=data.get("id", ""),
            finish_reason=choice.get("finish_reason"),
        )

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output."""
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

        response = await self.chat_completion(
            enhanced_messages,
            temperature=temperature,
            **kwargs,
        )

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        return schema.model_validate_json(content)

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        payload = {
            "model": self._config.model_name,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with self.client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"] and chunk["choices"][0]["delta"].get(
                            "content"
                        ):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
