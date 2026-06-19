# OrchardFlow

OrchardFlow is a local multi-agent orchestration prototype.

This branch introduces the WU-101 agent architecture surface:

- LangGraph-compatible supervisor, specialist, and reviewer nodes.
- Conditional retry, rejection, escalation, and completion routes.
- Placeholder-safe OpenAI and Anthropic provider adapters.
- A deterministic fake provider for local tests.
- A schema-aware tool registry with local rate limits.

The WU-102 stacked branch adds:

- Redis-style short-term task memory.
- ChromaDB/PostgreSQL-style long-term semantic memory interfaces with local stores.
- Memory scoring, consolidation, and decay.
- Redis and Celery integration boundaries backed by a local in-process queue.
- Supervisor planning-time long-term memory lookup.

Run the focused check with:

```bash
python3 -m pytest tests/test_agents.py tests/test_memory_queue.py
```
