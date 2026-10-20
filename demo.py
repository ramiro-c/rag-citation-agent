#!/usr/bin/env python
"""Interactive demo: ask questions against the local cached corpus.

Usage:
    uv run python demo.py --fetch           # download corpus (needs internet, once)
    uv run python demo.py                   # interactive question loop (offline)
    uv run python demo.py --ask "Question"  # single question, no loop

Requires LLM_API_KEY in the environment or in a .env file at the repo root.
"""

import sys
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from rag_agent.agent import RagCitationAgent
from rag_agent.corpus import CORPUS_PATH, build_retriever, fetch_corpus, load_corpus


def build_agent():
    try:
        docs = load_corpus(CORPUS_PATH)
    except FileNotFoundError as exc:
        print(f"error: {exc.args[0]}")
        sys.exit(1)

    print(f"corpus: {len(docs)} docs cached in {CORPUS_PATH}")
    agent = RagCitationAgent.from_env(retriever=build_retriever(docs))
    model = agent.llm.model
    print(f"provider: {agent.llm.base_url} | model: {model} | threshold: {agent.confidence_threshold}\n")
    return agent


def print_result(result):
    print(f"\na: {result.answer}")
    if result.citations:
        print("citations:")
        for c in result.citations:
            print(f"  [{c.doc_id}] \"{c.quote}\"")
    else:
        print("citations: none — answer is NOT grounded on any source")
    print(f"confidence: {result.confidence:.2f} | low_confidence: {result.low_confidence} | insufficient_evidence: {result.insufficient_evidence}\n")


def friendly_answer(agent, question):
    try:
        print_result(agent.answer(question))
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        hint = (
            "check LLM_API_KEY in .env (or your environment)" if status == 401
            else "you may have hit a rate limit, wait and retry" if status == 429
            else "check the LLM error above"
        )
        print(f"\nLLM error {status}: {hint}\n")


def main():
    if "--fetch" in sys.argv or not CORPUS_PATH.exists():
        print("fetching corpus from Wikipedia (cached to data/corpus.jsonl)...")
        docs = fetch_corpus()
        print(f"saved {len(docs)} docs\n")

    agent = build_agent()

    ask_index = sys.argv.index("--ask") + 1 if "--ask" in sys.argv else None
    if ask_index:
        if ask_index >= len(sys.argv):
            print("usage: uv run python demo.py --ask \"your question\"")
            sys.exit(1)
        friendly_answer(agent, sys.argv[ask_index])
        return

    print("ask anything covered by the corpus ('q' to quit)")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"q", "quit", "exit"}:
            break
        friendly_answer(agent, question)


if __name__ == "__main__":
    main()
