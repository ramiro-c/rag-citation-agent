# rag-citation-agent v1

Objective: RAG agent that answers with cited sources, flags low confidence, and falls back to broader retrieval. Portfolio piece — mechanism built by hand, no agent framework.
Branch: main | Delivery: single PR (small repo) | Runner: pytest

## Specs

User's README checklist (verbatim):

- S1: "Recuperar contexto relevante desde el índice/vector store"
- S2: "Responder siempre con fuentes citadas"
- S3: "Detectar y marcar respuestas de baja confianza"
- S4: "Fallback a búsqueda cuando el contexto no alcanza"

Assumptions (marked as such):
- LLM provider: OpenRouter (OpenAI-compatible chat completions), key from `OPENROUTER_API_KEY`.
- Stack: Python, no agent framework. Retrieval v1 = in-memory TF-IDF vector store (numpy + cosine), behind a `Retriever` interface so a real embedding/vector store can be swapped in later.
- All LLM calls go through an injectable client; unit tests use fakes, no network.

## Tasks

- T1 (S1): TF-IDF vector store + `Retriever` interface — inline
- T2 (S2): Structured cited answer (Pydantic: answer, citations, confidence) + OpenRouter client — inline
- T3 (S3): Confidence gate: low-confidence answers are flagged, never silently passed off as grounded — inline
- T4 (S4): Fallback: when confidence < threshold or citations unsupported, re-query with expanded retrieval; if still weak, return explicit "insufficient evidence" answer — inline
- T5: README update (usage, architecture, checklist ticks) — inline
- T6 (user request 2026-10-19): Env-based configuration — provider preset/overrides (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`) and agent knobs (`AGENT_CONFIDENCE_THRESHOLD`, `AGENT_MAX_ATTEMPTS`, `AGENT_RETRIEVAL_K`) — inline, done

## Log

- L1 (2026-10-19): User's original request — start the agent portfolio with rag-citation-agent. Repos react-planning-agent and structured-output-agent deleted (already built). Stack decision by user: OpenRouter provider, "a mano o algo que no sea langgraph - adk de google"; chosen: handmade (no framework).
