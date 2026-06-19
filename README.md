# OrchardFlow

OrchardFlow is the local project shell for "Project 2: Multi-Agent Orchestration with Persistent Memory and Human-in-the-Loop."

The product was generated from the Project 2 source brief:

`project-2-multi-agent-orchestration.md`

Current status: WU-101 agent architecture, WU-102 memory/queue work, WU-103 human-in-the-loop review, WU-104 observability/replay, and WU-105 demo/containerization/docs are implemented and locally verified.

Pitch: "Production infrastructure for autonomous AI workflows, not an AI demo."

## WU-101 Surfaces

- `orchardflow.agents`: LangGraph-compatible supervisor, specialist, and reviewer nodes with retry, rejection, escalation, and completion routes.
- `orchardflow.providers`: placeholder-safe OpenAI and Anthropic adapters plus deterministic fake providers for local tests.
- `orchardflow.tools`: tool registry storing name, schema, and rate limits with local sliding-window enforcement.

## WU-102 Surfaces

- `orchardflow.memory`: task-scoped Redis-style short-term memory, ChromaDB/PostgreSQL-style long-term semantic memory, deterministic scoring, consolidation, and decay.
- `orchardflow.queueing`: local FIFO queue with Redis broker and Celery task queue integration boundaries.
- `orchardflow.agents`: optional supervisor planning memory lookup via `long_term_memory`; default behavior remains local and keyless.

## WU-103 Surfaces

- `orchardflow.review`: escalation policy, local review queue, review request payloads, and exact approval levels: Notify, Approve Action, Approve Plan, Take Over.
- `orchardflow.agents`: low-confidence, repeated-failure, sensitive-operation, and low-quality escalation triggers pause execution and package context into the review queue.
- `review_app.py`: Streamlit Review UI entry point showing context, proposed action, reasoning, relevant memories, and approval levels without requiring a live server for tests.

## WU-104 Surfaces

- `orchardflow.observability`: local OpenTelemetry-compatible trace records with trace/span ids, timestamps, per-record latency, cost, error status, trace explorer export, and replay divergence helpers.
- `orchardflow.agents`: graph execution emits trace records for planning decisions, memory retrievals, local tool calls, escalation events, node latency, node cost, and node errors without requiring a live telemetry backend.
- `tests/test_observability.py`: focused coverage for ACC-401 and ACC-402, including replay with modified input that reports divergence.

## WU-105 Surfaces

- `orchardflow.demo`: deterministic end-to-end demo showing task input, supervisor decomposition, specialist outputs, reviewer route-back, memory-informed planning, final human approval, and trace evidence.
- `docs/architecture.md`: architecture diagram, service topology, and final pitch.
- `Dockerfile` and `docker-compose.yml`: local app, Redis, PostgreSQL, and Chroma-oriented services with placeholder API key defaults and no baked live secrets.
- `tests/test_demo.py`: focused coverage for ACC-501, ACC-502, ACC-503, ACC-901, and ACC-902 evidence without requiring live OpenAI, Anthropic, Redis, PostgreSQL, ChromaDB, Celery, or Docker daemon.

## Demo

- `python3 -m orchardflow.demo`
- `orchardflow-demo` after installing the package.

## Focused Checks

- `python3 -m pytest tests/test_agents.py`
- `python3 -m unittest tests/test_agents.py`
- `python3 -m pytest tests/test_memory_queue.py`
- `python3 -m unittest tests/test_memory_queue.py`
- `python3 -m pytest tests/test_review.py`
- `python3 -m unittest tests/test_review.py`
- `python3 -m pytest tests/test_observability.py`
- `python3 -m unittest tests/test_observability.py`
- `python3 -m pytest tests/test_demo.py`
- `python3 -m unittest tests/test_demo.py`
- `python3 -m pytest`

## Local Containers

- `docker compose up --build app`

The compose file defines Redis, PostgreSQL, and Chroma-oriented services for local integration shape. API keys default to `placeholder` and are not required for the demo or tests.
