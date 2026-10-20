"""Local corpus management for the demo: download, cache, load, index.

The demo corpus is fetched once from Wikipedia (REST page summaries) and
cached at data/corpus.jsonl so the demo runs fully offline afterwards.
"""

import json
from pathlib import Path

import httpx

from rag_agent.retrieval import TfidfRetriever

CORPUS_PATH = Path("data/corpus.jsonl")

TOPICS = [
    "Artificial_intelligence",
    "Machine_learning",
    "Large_language_model",
    "Retrieval-augmented_generation",
    "Python_(programming_language)",
    "Eiffel_Tower",
    "Great_Wall_of_China",
    "Photosynthesis",
    "European_Union",
    "Apollo_program",
    "Theory_of_relativity",
    "Penicillin",
    "Coffee",
    "Tango",
    "Buenos_Aires",
    "Moon",
    "Black_hole",
    "Internet",
    "Association_football",
    "Napoleon",
]

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    """Read the JSONL corpus. Each line: {"doc_id": str, "text": str}."""
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path}: run 'uv run python demo.py --fetch' first"
        )
    docs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        if "doc_id" not in doc or "text" not in doc:
            raise ValueError(f"corpus line missing 'doc_id' or 'text': {line[:80]!r}")
        docs.append(doc)
    return docs


def build_retriever(docs: list[dict]) -> TfidfRetriever:
    """Index the corpus into the in-memory TF-IDF retriever."""
    retriever = TfidfRetriever()
    for doc in docs:
        if "doc_id" not in doc:
            raise ValueError("every corpus doc needs a 'doc_id'")
        retriever.add(doc["doc_id"], doc.get("text", ""))
    return retriever


def fetch_summary(title: str, retries: int = 2) -> dict | None:
    """Fetch one Wikipedia page summary as a corpus doc, or None on failure."""
    import time

    for attempt in range(retries + 1):
        try:
            response = httpx.get(
                WIKIPEDIA_SUMMARY_URL.format(title=title),
                headers={"User-Agent": "rag-citation-agent-demo/0.1 (personal portfolio)"},
                timeout=15.0,
            )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < retries:
                time.sleep(1.5)
                continue
            return None
        except httpx.HTTPError:
            return None
    data = response.json()
    extract = data.get("extract", "").strip()
    if not extract:
        return None
    return {"doc_id": f"wiki:{title}", "text": extract}


def fetch_corpus(topics: list[str] | None = None, path: Path = CORPUS_PATH) -> list[dict]:
    """Fetch all topic summaries, persist to JSONL, return the docs."""
    import time

    docs = []
    for topic in topics or TOPICS:
        doc = fetch_summary(topic)
        if doc:
            docs.append(doc)
        print(f"  {'ok ' if doc else 'skip'} {topic}")
        time.sleep(0.4)
    if not docs:
        raise RuntimeError("fetch failed for every topic: check your connection")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d) + "\n" for d in docs))
    return docs
