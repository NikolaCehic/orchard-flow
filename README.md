# OrchardFlow

OrchardFlow is a local multi-agent orchestration prototype.

This branch introduces the WU-101 agent architecture surface:

- LangGraph-compatible supervisor, specialist, and reviewer nodes.
- Conditional retry, rejection, escalation, and completion routes.
- Placeholder-safe OpenAI and Anthropic provider adapters.
- A deterministic fake provider for local tests.
- A schema-aware tool registry with local rate limits.

Run the focused check with:

```bash
python3 -m pytest tests/test_agents.py
```
