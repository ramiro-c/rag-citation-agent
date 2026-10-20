"""OpenAI-compatible chat-completions client with forced JSON responses.

Provider-agnostic: point base_url at OpenRouter, OpenAI, or any
compatible endpoint. Which one to use is decided by config.from_env().
"""

import json
import re
from typing import Protocol

import httpx


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


def parse_json_response(content: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating fences and prose."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    embedded = re.search(r"\{.*\}", content, re.DOTALL)
    if embedded:
        return json.loads(embedded.group(0))
    raise ValueError(f"LLM response is not valid JSON: {content[:200]!r}")


class OpenAICompatClient:
    """Minimal chat-completions client that forces a JSON response."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("API key required (set LLM_API_KEY or {PROVIDER}_API_KEY)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def complete_json(self, system: str, user: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_json_response(content)
