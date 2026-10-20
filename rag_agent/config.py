"""Env-based configuration: provider preset/overrides and agent knobs.

Environment variables:
    LLM_PROVIDER              provider preset: openrouter (default) | openai
    LLM_BASE_URL              overrides the preset base URL (any OpenAI-compatible API)
    LLM_MODEL                 overrides the preset model
    LLM_API_KEY               API key (fallback: {PROVIDER}_API_KEY, e.g. OPENAI_API_KEY)
    AGENT_CONFIDENCE_THRESHOLD  min confidence to accept an answer (default 0.6)
    AGENT_MAX_ATTEMPTS          retrieve+answer attempts before giving up (default 2)
    AGENT_RETRIEVAL_K           base number of sources per attempt (default 3)
"""

import os
from dataclasses import dataclass

from rag_agent.llm import OpenAICompatClient, LLMClient

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemma-4-31b-it:free",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

DEFAULT_PROVIDER = "openrouter"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        preset = PROVIDER_PRESETS.get(provider, {})

        base_url = os.environ.get("LLM_BASE_URL", "").strip() or preset.get("base_url")
        if not base_url:
            raise ValueError(
                f"unknown provider '{provider}': set LLM_BASE_URL to an "
                "OpenAI-compatible endpoint"
            )

        model = os.environ.get("LLM_MODEL", "").strip() or preset.get("model", "")
        api_key = (
            os.environ.get("LLM_API_KEY", "").strip()
            or os.environ.get(f"{provider.upper()}_API_KEY", "").strip()
        )
        return cls(provider=provider, base_url=base_url, model=model, api_key=api_key)


def build_llm(cfg: LLMConfig, timeout: float = 30.0) -> LLMClient:
    return OpenAICompatClient(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        timeout=timeout,
    )


@dataclass(frozen=True)
class AgentConfig:
    confidence_threshold: float
    max_attempts: int
    retrieval_k: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            confidence_threshold=float(
                os.environ.get("AGENT_CONFIDENCE_THRESHOLD", "0.6")
            ),
            max_attempts=int(os.environ.get("AGENT_MAX_ATTEMPTS", "2")),
            retrieval_k=int(os.environ.get("AGENT_RETRIEVAL_K", "3")),
        )
