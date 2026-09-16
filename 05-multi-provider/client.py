"""Thin Chat Completions client. Groq/OpenAI/Ollama share one POST; Anthropic is translated."""

from __future__ import annotations

from typing import Any

import httpx

from anthropic_adapter import from_anthropic_response, to_anthropic_body
from profiles import Profile


class ChatClient:
    def __init__(self, profile: Profile, timeout: float = 90.0):
        self.profile = profile
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        model = model or self.profile.default_model
        if self.profile.kind == "anthropic":
            return self._anthropic(messages, model, temperature, max_tokens)
        return self._openai(messages, model, temperature, max_tokens, extra or {})

    def _openai(self, messages, model, temperature, max_tokens, extra: dict) -> dict:
        url = f"{self.profile.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.profile.api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        data = response.json()
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code} {url}: {data}")
        return data

    def _anthropic(self, messages, model, temperature, max_tokens) -> dict:
        url = f"{self.profile.base_url.rstrip('/')}/messages"
        payload = to_anthropic_body(messages, model, max_tokens, temperature)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                headers={
                    "x-api-key": self.profile.api_key(),
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        data = response.json()
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code} {url}: {data}")
        return from_anthropic_response(data)
