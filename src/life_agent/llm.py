"""Async LLM client abstraction supporting structured JSON, multimodal vision JSON, and function calling."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import LLMConfig, ModelEndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw: Any = None


def extract_json_object(text: str) -> dict[str, Any]:
    """Robustly extract a JSON object from LLM text (handles markdown code fences)."""
    cleaned = text.strip()
    if not cleaned:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse JSON from LLM output: %s", text[:300])
    return {}


class LLMClient:
    """Multi-role OpenAI-compatible async LLM client."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._clients: dict[tuple[str, str, float], AsyncOpenAI] = {}

    def _get_client(self, ep: ModelEndpointConfig) -> AsyncOpenAI:
        key = (ep.base_url, ep.api_key, ep.timeout_seconds)
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                base_url=ep.base_url,
                api_key=ep.api_key or "EMPTY",
                timeout=ep.timeout_seconds,
            )
        return self._clients[key]

    async def complete_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Request a structured JSON object response for Structurer / Gateway / RTIR / Importer."""
        ep = self.config.for_role(role)
        client = self._get_client(ep)
        kwargs: dict[str, Any] = {
            "model": ep.model,
            "temperature": ep.temperature,
            "max_tokens": ep.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = await client.chat.completions.create(**kwargs)

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return extract_json_object(text)

    async def complete_vision_json(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        """Send an image + text prompt to a multimodal OpenAI-compatible endpoint and parse JSON."""
        ep = self.config.for_role(role)
        client = self._get_client(ep)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"

        kwargs: dict[str, Any] = {
            "model": ep.model,
            "temperature": ep.temperature,
            "max_tokens": max(ep.max_tokens, 4096),
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = await client.chat.completions.create(**kwargs)

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return extract_json_object(text)

    async def chat(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Chat completion with optional OpenAI function calling tools."""
        ep = self.config.for_role(role)
        client = self._get_client(ep)
        kwargs: dict[str, Any] = {
            "model": ep.model,
            "temperature": ep.temperature,
            "max_tokens": ep.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await client.chat.completions.create(**kwargs)
        if not resp.choices:
            return LLMResponse(content="", raw=resp)

        msg = resp.choices[0].message
        tool_calls: list[ToolCallRequest] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            raw=resp,
        )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
