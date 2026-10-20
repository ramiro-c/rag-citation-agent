"""Tests for env-based configuration of provider and agent knobs."""

import pytest

from rag_agent.agent import RagCitationAgent
from rag_agent.config import AgentConfig, LLMConfig, build_llm
from rag_agent.retrieval import TfidfRetriever


def test_openrouter_preset_by_default(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")

    cfg = LLMConfig.from_env()

    assert cfg.provider == "openrouter"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model == "google/gemma-4-31b-it:free"
    assert cfg.api_key == "sk-or-test"


def test_openai_preset_uses_provider_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

    cfg = LLMConfig.from_env()

    assert cfg.provider == "openai"
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key == "sk-openai-test"


def test_env_overrides_preset(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4")
    monkeypatch.setenv("LLM_BASE_URL", "https://my-proxy.internal/v1")

    cfg = LLMConfig.from_env()

    assert cfg.model == "anthropic/claude-sonnet-4"
    assert cfg.base_url == "https://my-proxy.internal/v1"


def test_unknown_provider_without_base_url_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        LLMConfig.from_env()


def test_build_llm_returns_working_client(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")

    client = build_llm(LLMConfig.from_env())

    assert client.base_url == "https://openrouter.ai/api/v1"


def test_agent_config_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("AGENT_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("AGENT_RETRIEVAL_K", raising=False)

    cfg = AgentConfig.from_env()

    assert cfg.confidence_threshold == 0.6
    assert cfg.max_attempts == 2
    assert cfg.retrieval_k == 3


def test_agent_config_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AGENT_RETRIEVAL_K", "5")

    cfg = AgentConfig.from_env()

    assert cfg.confidence_threshold == 0.8
    assert cfg.max_attempts == 3
    assert cfg.retrieval_k == 5


def test_agent_from_env_wires_everything(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("AGENT_CONFIDENCE_THRESHOLD", "0.7")

    agent = RagCitationAgent.from_env(retriever=TfidfRetriever())

    assert isinstance(agent, RagCitationAgent)
    assert agent.confidence_threshold == 0.7
    assert agent.llm.base_url == "https://openrouter.ai/api/v1"
