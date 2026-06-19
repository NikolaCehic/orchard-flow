"""End-to-end OrchardFlow demo for WU-105.

The demo intentionally uses fake providers and local stores so it can run in
tests, containers, and laptops without live API keys or infrastructure.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from orchardflow.agents import create_agent_graph
from orchardflow.memory import ChromaDBSemanticMemoryStore
from orchardflow.observability import json_safe, summarize_execution_state
from orchardflow.providers import FakeProvider
from orchardflow.review import EscalationDecision, LocalReviewQueue, build_review_context


FINAL_PITCH = "Production infrastructure for autonomous AI workflows, not an AI demo."
PROJECT_SOURCE_PATH = "project-2-multi-agent-orchestration.md"
DEFAULT_DEMO_TASK = (
    "Research, analyze, write, and build a production autonomous AI workflow "
    "plan with memory-informed human approval."
)


def build_demo_memory(*, now: float = 1_000.0) -> ChromaDBSemanticMemoryStore:
    """Seed deterministic long-term memories used by the supervisor plan."""

    memory = ChromaDBSemanticMemoryStore(collection_name="orchardflow_demo")
    memory.record_user_preference(
        user_id="demo-user",
        content=(
            "For production autonomous AI workflow plans, lead with architecture, "
            "reviewer route-back evidence, and explicit human approval."
        ),
        metadata={"source": "demo-seed", "preference": "acceptance-evidence"},
        importance=0.95,
        now=now,
    )
    memory.record_outcome(
        task_id="prior-demo",
        content=(
            "Research, analysis, writing, and code specialists worked when the "
            "reviewer routed approved intermediate outputs back to the supervisor."
        ),
        worked=True,
        tools_used=["echo"],
        metadata={"source": "demo-seed", "pattern": "reviewer-route-back"},
        importance=0.85,
        now=now + 1,
    )
    memory.record_tool_use(
        task_id="prior-demo",
        tool_name="echo",
        content=(
            "The local echo tool kept demo specialist calls deterministic without "
            "external services or credentials."
        ),
        success=True,
        metadata={"source": "demo-seed", "integration": "local-tool"},
        importance=0.7,
        now=now + 2,
    )
    return memory


def run_end_to_end_demo(
    *,
    task_input: str = DEFAULT_DEMO_TASK,
    user_id: str = "demo-user",
    memory_now: float = 1_010.0,
) -> dict[str, Any]:
    """Run the full local demo path and return a stable evidence payload."""

    memory = build_demo_memory(now=memory_now - 10)
    review_queue = LocalReviewQueue(queue_name="demo-human-approval")
    graph = create_agent_graph(
        provider=FakeProvider(response_prefix="demo-specialist", confidence=0.95),
        long_term_memory=memory,
        planning_memory_limit=5,
        review_queue=review_queue,
    )

    final_state = graph.run(
        {
            "task": task_input,
            "user_id": user_id,
            "memory_now": memory_now,
        }
    )
    approval = approve_final_output(
        final_state,
        review_queue=review_queue,
        now=memory_now + 10,
    )

    plan = list(final_state.get("plan", []))
    specialist_outputs = list(final_state.get("specialist_outputs", []))
    review_decisions = list(final_state.get("review_decisions", []))
    relevant_memories = list(final_state.get("relevant_memories", []))
    trace_records = list(final_state.get("trace_records", []))
    event_types = sorted(
        {
            str(record.get("event_type"))
            for record in trace_records
            if isinstance(record, Mapping)
        }
    )

    return json_safe(
        {
            "pitch": FINAL_PITCH,
            "source_of_truth": PROJECT_SOURCE_PATH,
            "requires_live_services": False,
            "task_input": task_input,
            "supervisor_decomposition": plan,
            "specialist_execution": {
                "design_target": "parallel specialist delegation",
                "local_demo_mode": "deterministic graph traversal",
                "roles": [
                    output.get("role")
                    for output in specialist_outputs
                    if isinstance(output, Mapping)
                ],
                "outputs": [
                    {
                        "role": output.get("role"),
                        "step_id": output.get("step_id"),
                        "provider": output.get("provider"),
                        "confidence": output.get("confidence"),
                        "tool_calls": [
                            tool_call.get("name")
                            for tool_call in output.get("tool_calls", [])
                            if isinstance(tool_call, Mapping)
                        ],
                    }
                    for output in specialist_outputs
                    if isinstance(output, Mapping)
                ],
            },
            "reviewer_route_back": {
                "observed": any(
                    decision.get("route") == "approve"
                    for decision in review_decisions
                    if isinstance(decision, Mapping)
                ),
                "decisions": [
                    {
                        "role": decision.get("role"),
                        "route": decision.get("route"),
                        "reason": decision.get("reason"),
                    }
                    for decision in review_decisions
                    if isinstance(decision, Mapping)
                ],
            },
            "memory_informed_decision": {
                "observed": bool(relevant_memories),
                "retrieved_memories": [
                    {
                        "id": memory_result.get("record", {}).get("id"),
                        "type": memory_result.get("record", {}).get("memory_type"),
                        "content": memory_result.get("record", {}).get("content"),
                        "score": memory_result.get("score"),
                    }
                    for memory_result in relevant_memories
                    if isinstance(memory_result, Mapping)
                ],
                "trace_has_memory_retrieval": "memory_retrieval" in event_types,
            },
            "human_approval": approval,
            "trace": {
                "trace_id": final_state.get("trace_id"),
                "event_types": event_types,
                "execution_summary": summarize_execution_state(final_state),
            },
            "final_status": final_state.get("final_status"),
        }
    )


def approve_final_output(
    final_state: Mapping[str, Any],
    *,
    review_queue: LocalReviewQueue,
    now: float,
) -> dict[str, Any]:
    """Queue and resolve the final human approval step for the demo."""

    context = build_review_context(final_state)
    context.update(
        {
            "final_status": final_state.get("final_status"),
            "trace_id": final_state.get("trace_id"),
            "pitch": FINAL_PITCH,
        }
    )
    request = review_queue.enqueue(
        decision=EscalationDecision(
            trigger="sensitive_operation",
            approval_level="Approve Plan",
            proposed_action="Approve the final demo output before delivery.",
            reasoning=(
                "The completed multi-agent output is ready for a human approval "
                "checkpoint before delivery."
            ),
        ),
        context=context,
        relevant_memories=final_state.get("relevant_memories", []),
        now=now,
    )
    resolved = review_queue.resolve(
        request.id,
        decision="approved",
        reviewer_notes="Human approved the final demo output.",
        now=now + 1,
    )
    return resolved.to_dict()


def main() -> None:
    """Print the demo evidence payload as JSON."""

    print(json.dumps(run_end_to_end_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
