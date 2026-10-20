"""Tests for the TF-IDF retriever (S1: retrieve relevant context from the index)."""

from rag_agent.retrieval import TfidfRetriever

DOCS = {
    "d1": "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
    "d2": "The Great Wall of China stretches over 21,000 kilometers.",
    "d3": "Photosynthesis converts sunlight, water and CO2 into glucose.",
}


def test_ranks_most_relevant_doc_first():
    r = TfidfRetriever()
    for doc_id, text in DOCS.items():
        r.add(doc_id, text)

    results = r.search("Where is the Eiffel Tower?", k=2)

    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[0].score >= results[1].score


def test_returns_at_most_k_results():
    r = TfidfRetriever()
    for doc_id, text in DOCS.items():
        r.add(doc_id, text)

    results = r.search("glucose sunlight", k=1)

    assert len(results) == 1


def test_no_results_for_out_of_domain_query():
    r = TfidfRetriever()
    for doc_id, text in DOCS.items():
        r.add(doc_id, text)

    results = r.search("quantum blockchain cryptocurrency", k=3)

    assert results == []


def test_source_carries_text_and_score():
    r = TfidfRetriever()
    for doc_id, text in DOCS.items():
        r.add(doc_id, text)

    results = r.search("Great Wall kilometers", k=1)

    assert results[0].doc_id == "d2"
    assert "21,000" in results[0].text
    assert 0.0 < results[0].score <= 1.0
