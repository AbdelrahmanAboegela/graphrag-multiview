"""Google Gemini LLM Provider."""

import json
from typing import Any, AsyncIterator

import google.generativeai as genai
from pydantic import BaseModel, SecretStr

from graphrag.llm.base import LLMConfig, LLMProvider, LLMResponse


class GeminiConfig(LLMConfig):
    """Gemini-specific configuration."""

    api_key: SecretStr
    model_name: str = "gemini-1.5-pro"


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self._config = config
        genai.configure(api_key=config.api_key.get_secret_value())
        self.client = genai.GenerativeModel(config.model_name)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format."""
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        return gemini_messages, system_instruction

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate chat completion using Gemini."""
        gemini_messages, system_instruction = self._convert_messages(messages)

        generation_config = genai.GenerationConfig(
            temperature=temperature or self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
        )

        # Create model with system instruction if present
        model = self.client
        if system_instruction:
            model = genai.GenerativeModel(
                self._config.model_name,
                system_instruction=system_instruction,
            )

        response = await model.generate_content_async(
            gemini_messages,
            generation_config=generation_config,
        )

        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            content=response.text,
            model=self._config.model_name,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.candidates[0].finish_reason.name
            if response.candidates
            else None,
        )

    async def structured_output(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float = 0,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output with JSON schema."""
        # Add schema instruction to messages
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_messages = messages.copy()

        # Find or create system message
        system_idx = next(
            (i for i, m in enumerate(enhanced_messages) if m["role"] == "system"),
            None,
        )

        schema_instruction = f"""
Respond with valid JSON matching this schema:
{schema_json}

Output ONLY the JSON object, no markdown or explanation.
"""

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

        # Parse and validate response
        content = response.content.strip()
        if content.startswith("```"):
            # Remove markdown code blocks
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
        gemini_messages, system_instruction = self._convert_messages(messages)

        model = self.client
        if system_instruction:
            model = genai.GenerativeModel(
                self._config.model_name,
                system_instruction=system_instruction,
            )

        response = await model.generate_content_async(
            gemini_messages,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text
