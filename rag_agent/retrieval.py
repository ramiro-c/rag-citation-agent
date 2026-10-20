"""In-memory TF-IDF vector store with cosine similarity.

Deliberately hand-rolled: the retrieval mechanism is part of what this
project demonstrates. Implements the Retriever interface so it can be
swapped for a real embedding-based store later.
"""

import math
import re
from typing import Protocol

from rag_agent.schemas import Source

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    def add(self, doc_id: str, text: str) -> None: ...

    def search(self, query: str, k: int = 3) -> list[Source]: ...


class TfidfRetriever:
    """Sparse TF-IDF vectors, cosine similarity, no external service."""

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}
        self._texts: dict[str, str] = {}
        self._df: dict[str, int] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        self._texts[doc_id] = text
        for term in set(tokens):
            self._df[term] = self._df.get(term, 0) + 1

    def search(self, query: str, k: int = 3) -> list[Source]:
        if not self._docs:
            return []
        idf = {
            term: math.log((1 + len(self._docs)) / (1 + df)) + 1
            for term, df in self._df.items()
        }
        doc_vectors = {doc_id: self._tfidf(tokens, idf) for doc_id, tokens in self._docs.items()}
        query_vector = self._tfidf(tokenize(query), idf)
        query_norm = self._norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[Source] = []
        for doc_id, vector in doc_vectors.items():
            dot = sum(weight * query_vector.get(term, 0.0) for term, weight in vector.items())
            score = dot / (self._norm(vector) * query_norm) if self._norm(vector) else 0.0
            if score > 0:
                scored.append(
                    Source(
                        doc_id=doc_id,
                        text=self._texts[doc_id],
                        score=round(score, 4),
                    )
                )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    @staticmethod
    def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        return {term: count * idf.get(term, 0.0) for term, count in tf.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(w * w for w in vector.values()))
