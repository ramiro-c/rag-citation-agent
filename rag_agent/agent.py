"""RagCitationAgent: retrieve, answer with citations, flag low confidence, fall back."""

import re

from rag_agent.config import AgentConfig, LLMConfig, build_llm
from rag_agent.llm import LLMClient
from rag_agent.retrieval import Retriever
from rag_agent.schemas import CitedAnswer, Source

SYSTEM_PROMPT = (
    "You are a question-answering agent. You MUST answer using ONLY the provided "
    "sources. Every claim must cite the doc_id of the source that supports it, "
    "quoting the exact supporting sentence from that source. If the sources do "
    "not contain enough evidence, set confidence below 0.5 and say so in the "
    "answer. Respond in JSON with keys: answer (str), citations "
    "(list of {doc_id, quote}), confidence (float 0..1)."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class RagCitationAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        confidence_threshold: float = 0.6,
        max_attempts: int = 2,
        retrieval_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts
        self.retrieval_k = retrieval_k

    @classmethod
    def from_env(cls, retriever: Retriever) -> "RagCitationAgent":
        agent_cfg = AgentConfig.from_env()
        return cls(
            retriever=retriever,
            llm=build_llm(LLMConfig.from_env()),
            confidence_threshold=agent_cfg.confidence_threshold,
            max_attempts=agent_cfg.max_attempts,
            retrieval_k=agent_cfg.retrieval_k,
        )

    def answer(self, question: str, k: int | None = None) -> CitedAnswer:
        k = k if k is not None else self.retrieval_k
        sources: list[Source] = []
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            new_sources = self.retriever.search(question, k=k * attempt)
            # Broader retrieval each attempt; merge without duplicates.
            seen = {s.doc_id for s in sources}
            sources.extend(s for s in new_sources if s.doc_id not in seen)

            retry_note = ""
            if attempt > 1:
                retry_note = (
                    "\n\nNote: the previous attempt lacked sufficiently grounded "
                    "evidence. Ground the answer better using the sources; if "
                    "impossible, set confidence below 0.5."
                )
            raw = self.llm.complete_json(
                SYSTEM_PROMPT, self._user_prompt(question, sources) + retry_note
            )
            cited = self._validate_citations(raw, sources)
            confidence = float(raw.get("confidence", 0.0))

            if cited and confidence >= self.confidence_threshold:
                return CitedAnswer(
                    answer=raw["answer"],
                    citations=cited,
                    sources=sources,
                    confidence=confidence,
                    low_confidence=False,
                )
            # Low confidence or unsupported citations: retry with broader context.

        last_confidence = float(raw.get("confidence", 0.0))
        return CitedAnswer(
            answer=raw.get("answer", ""),
            citations=self._validate_citations(raw, sources),
            sources=sources,
            confidence=last_confidence,
            low_confidence=True,
            insufficient_evidence=True,
        )

    def _user_prompt(self, question: str, sources: list[Source]) -> str:
        context = "\n\n".join(f"[{s.doc_id}] {s.text}" for s in sources)
        return f"Sources:\n{context}\n\nQuestion: {question}"

    def _validate_citations(self, raw: dict, sources: list[Source]) -> list:
        """Keep only citations pointing at a known source whose text contains the quote."""
        by_id = {s.doc_id: s for s in sources}
        valid = []
        for cit in raw.get("citations", []):
            source = by_id.get(cit.get("doc_id"))
            if source is None:
                continue
            if _normalize(cit.get("quote", "")) not in _normalize(source.text):
                continue
            valid.append(cit)
        return valid
