# rag-citation-agent

Agente RAG que recupera contexto, responde citando fuentes
y marca respuestas de baja confianza. Si la evidencia es insuficiente,
hace fallback a búsqueda adicional.

**Qué demuestra:** prevención de alucinaciones con citas y umbrales.

## Checklist de features clave

- [ ] Recuperar contexto relevante desde el índice/vector store
- [ ] Responder siempre con fuentes citadas
- [ ] Detectar y marcar respuestas de baja confianza
- [ ] Fallback a búsqueda cuando el contexto no alcanza
