"""Tests for corpus loading and retrieval wiring (demo pipeline)."""

import json

import pytest

from rag_agent.corpus import CORPUS_PATH, build_retriever, load_corpus
from rag_agent.retrieval import TfidfRetriever


def test_load_corpus_reads_jsonl(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"doc_id": "d1", "text": "Eiffel Tower is in Paris."})
        + "\n"
        + json.dumps({"doc_id": "d2", "text": "Great Wall is in China."})
        + "\n"
    )

    docs = load_corpus(path)

    assert [d["doc_id"] for d in docs] == ["d1", "d2"]
    assert "Great Wall" in docs[1]["text"]


def test_load_corpus_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path_anchor() / "nope.jsonl")


def tmp_path_anchor():
    import pathlib

    return pathlib.Path("/tmp")


def test_load_corpus_skips_blank_lines(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"doc_id": "d1", "text": "A."}) + "\n\n" + json.dumps({"doc_id": "d2", "text": "B."}) + "\n"
    )

    docs = load_corpus(path)

    assert len(docs) == 2


def test_corpus_path_is_local_and_cached():
    assert CORPUS_PATH == __import__("pathlib").Path("data/corpus.jsonl")


def test_build_retriever_indexes_all_docs():
    docs = [
        {"doc_id": "d1", "text": "The Eiffel Tower is in Paris."},
        {"doc_id": "d2", "text": "The Great Wall is in China."},
        {"doc_id": "d3", "text": "Photosynthesis converts sunlight into glucose."},
    ]

    retriever = build_retriever(docs)

    assert isinstance(retriever, TfidfRetriever)
    results = retriever.search("Eiffel Paris", k=1)
    assert results[0].doc_id == "d1"


def test_build_retriever_rejects_docs_without_required_keys():
    with pytest.raises(ValueError, match="doc_id"):
        build_retriever([{"text": "missing id"}])
