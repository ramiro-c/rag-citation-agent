"""Tests for RagCitationAgent: citations (S2), confidence gate (S3), fallback (S4)."""

import pytest

from rag_agent.agent import RagCitationAgent
from rag_agent.retrieval import TfidfRetriever

DOCS = {
    "d1": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
    "d2": "The Great Wall of China stretches over 21,000 kilometers.",
    "d3": "Photosynthesis converts sunlight, water and CO2 into glucose.",
}


class FakeLLM:
    """Scriptable LLM client: returns queued JSON payloads in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append({"system": system, "user": user})
        return self.responses.pop(0)


def make_agent(responses, k=3):
    retriever = TfidfRetriever()
    for doc_id, text in DOCS.items():
        retriever.add(doc_id, text)
    llm = FakeLLM(responses)
    return RagCitationAgent(retriever=retriever, llm=llm), llm


GOOD = {
    "answer": "The Eiffel Tower is in Paris, France.",
    "citations": [{"doc_id": "d1", "quote": "The Eiffel Tower is located in Paris, France."}],
    "confidence": 0.9,
}

LOW = {
    "answer": "It is somewhere in Europe.",
    "citations": [{"doc_id": "d1", "quote": "The Eiffel Tower is located in Paris, France."}],
    "confidence": 0.3,
}

HALLUCINATED = {
    "answer": "The Eiffel Tower is in Rome.",
    "citations": [{"doc_id": "dX", "quote": "The Eiffel Tower is in Rome."}],
    "confidence": 0.95,
}


def test_answer_includes_valid_citation_and_sources():
    agent, _ = make_agent([GOOD])

    result = agent.answer("Where is the Eiffel Tower?")

    assert result.answer == "The Eiffel Tower is in Paris, France."
    assert result.citations[0].doc_id == "d1"
    assert result.sources, "answer must always carry the sources it was grounded on"
    assert result.low_confidence is False


def test_citation_with_unknown_doc_id_is_rejected():
    agent, _ = make_agent([HALLUCINATED, GOOD])

    result = agent.answer("Where is the Eiffel Tower?")

    # First attempt cited an unknown source -> fallback rescued the answer.
    assert result.citations[0].doc_id == "d1"
    assert result.low_confidence is False


def test_low_confidence_answer_is_flagged_after_failed_fallback():
    agent, _ = make_agent([LOW, LOW])

    result = agent.answer("Where is the Eiffel Tower?")

    assert result.low_confidence is True


def test_insufficient_evidence_when_no_valid_citation_twice():
    agent, _ = make_agent([HALLUCINATED, HALLUCINATED])

    result = agent.answer("Where is the Eiffel Tower?")

    assert result.insufficient_evidence is True
    assert result.low_confidence is True
    # S2: even a refusal carries the sources it searched.
    assert result.sources


def test_fallback_sends_broader_context_on_second_call():
    agent, llm = make_agent([LOW, GOOD])

    agent.answer("Where is the Eiffel Tower?")

    assert len(llm.calls) == 2
    first_ctx = llm.calls[0]["user"]
    second_ctx = llm.calls[1]["user"]
    assert second_ctx != first_ctx, "fallback must change the retrieval context"


def test_llm_receives_question_and_context():
    agent, llm = make_agent([GOOD])

    agent.answer("Where is the Eiffel Tower?")

    assert "Where is the Eiffel Tower?" in llm.calls[0]["user"]
    assert "d1" in llm.calls[0]["user"]
