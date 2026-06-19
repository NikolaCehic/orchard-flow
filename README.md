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

The WU-103 stacked branch adds:

- Escalation triggers for low confidence, repeated failure, sensitive operations, and low quality.
- A local review queue with Notify, Approve Action, Approve Plan, and Take Over levels.
- Review payloads containing context, proposed action, reasoning, and relevant memories.
- A Streamlit-facing review app module that is testable without running a server.

Run the focused check with:

```bash
python3 -m pytest tests/test_agents.py tests/test_memory_queue.py tests/test_review.py
```
