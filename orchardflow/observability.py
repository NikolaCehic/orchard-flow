"""Local OpenTelemetry-compatible traces and replay helpers for OrchardFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid
from typing import Any, Mapping


TRACE_EVENT_TYPES = (
    "node",
    "planning_decision",
    "tool_call",
    "memory_retrieval",
    "escalation_event",
    "error",
)


def _now_ns() -> int:
    return time.time_ns()


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def _trace_id() -> str:
    return uuid.uuid4().hex


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy of local execution values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def estimate_cost_usd(
    usage: Mapping[str, Any] | None = None,
    *,
    provider: str | None = None,
) -> float:
    """Return a local cost estimate from provider usage metadata.

    Fake/local providers have zero cost. Real adapters can pass a precomputed
    ``cost_usd`` usage field later without introducing a live telemetry backend.
    """

    usage_data = dict(usage or {})
    explicit_cost = usage_data.get("cost_usd")
    if isinstance(explicit_cost, (int, float)):
        return round(float(explicit_cost), 9)
    if provider in {None, "fake", "local"}:
        return 0.0
    return 0.0


def summarize_execution_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize an execution in a stable form for explorer views and replay."""

    plan = list(state.get("plan") or [])
    specialist_outputs = list(state.get("specialist_outputs") or [])
    review_decisions = list(state.get("review_decisions") or [])
    relevant_memories = list(state.get("relevant_memories") or [])
    trace_records = list(state.get("trace_records") or state.get("trace") or [])
    node_sequence = [
        record.get("node")
        for record in trace_records
        if isinstance(record, Mapping) and record.get("event_type") == "node"
    ]
    error_count = sum(
        1
        for record in trace_records
        if isinstance(record, Mapping) and record.get("error") is not None
    )

    memory_ids: list[Any] = []
    for memory in relevant_memories:
        if not isinstance(memory, Mapping):
            continue
        record = memory.get("record")
        if isinstance(record, Mapping):
            memory_ids.append(record.get("id"))

    return {
        "task": state.get("task"),
        "final_status": state.get("final_status") or state.get("status"),
        "status": state.get("status"),
        "plan_roles": [step.get("role") for step in plan if isinstance(step, Mapping)],
        "plan_size": len(plan),
        "specialist_roles": [
            output.get("role")
            for output in specialist_outputs
            if isinstance(output, Mapping)
        ],
        "specialist_count": len(specialist_outputs),
        "review_routes": [
            decision.get("route")
            for decision in review_decisions
            if isinstance(decision, Mapping)
        ],
        "review_count": len(review_decisions),
        "node_sequence": node_sequence,
        "escalation_trigger": state.get("escalation_trigger"),
        "approval_level": state.get("approval_level"),
        "memory_ids": memory_ids,
        "error_count": error_count,
    }


def compare_execution_summaries(
    original: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two execution summaries and return replay divergence data."""

    fields = (
        "task",
        "final_status",
        "plan_roles",
        "specialist_roles",
        "review_routes",
        "node_sequence",
        "escalation_trigger",
        "approval_level",
        "memory_ids",
        "error_count",
    )
    changes = []
    for field_name in fields:
        original_value = original.get(field_name)
        replay_value = replay.get(field_name)
        if original_value != replay_value:
            changes.append(
                {
                    "field": field_name,
                    "original": json_safe(original_value),
                    "replay": json_safe(replay_value),
                }
            )
    return {
        "changed": bool(changes),
        "change_count": len(changes),
        "changes": changes,
    }


@dataclass
class TraceRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    event_type: str
    node: str | None
    step_index: int | None
    start_time_unix_nano: int
    end_time_unix_nano: int
    latency_ms: float
    cost_usd: float = 0.0
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "INTERNAL"
    status_code: str = "OK"
    status_message: str = ""
    resource: dict[str, Any] = field(
        default_factory=lambda: {"service.name": "orchardflow"}
    )

    def to_dict(self) -> dict[str, Any]:
        attributes = dict(self.attributes)
        attributes.setdefault("orchardflow.event_type", self.event_type)
        if self.node is not None:
            attributes.setdefault("orchardflow.node", self.node)
        if self.step_index is not None:
            attributes.setdefault("orchardflow.step_index", self.step_index)
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "event_type": self.event_type,
            "node": self.node,
            "step_index": self.step_index,
            "start_time_unix_nano": self.start_time_unix_nano,
            "end_time_unix_nano": self.end_time_unix_nano,
            "duration_ms": self.latency_ms,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "status": {
                "code": self.status_code,
                "message": self.status_message,
            },
            "attributes": json_safe(attributes),
            "events": json_safe(self.events),
            "resource": json_safe(self.resource),
        }


@dataclass
class TraceRun:
    trace_id: str
    name: str
    input_state: dict[str, Any]
    service_name: str = "orchardflow"
    records: list[TraceRecord] = field(default_factory=list)
    start_time_unix_nano: int = field(default_factory=_now_ns)
    end_time_unix_nano: int | None = None

    def record(
        self,
        *,
        name: str,
        event_type: str,
        node: str | None = None,
        step_index: int | None = None,
        start_time_unix_nano: int | None = None,
        end_time_unix_nano: int | None = None,
        latency_ms: float | None = None,
        cost_usd: float = 0.0,
        error: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        events: list[Mapping[str, Any]] | None = None,
        parent_span_id: str | None = None,
    ) -> TraceRecord:
        if event_type not in TRACE_EVENT_TYPES:
            raise ValueError(f"unknown trace event type: {event_type}")
        start_ns = _now_ns() if start_time_unix_nano is None else start_time_unix_nano
        end_ns = start_ns if end_time_unix_nano is None else end_time_unix_nano
        if latency_ms is None:
            latency_ms = max((end_ns - start_ns) / 1_000_000, 0.0)
        status_code = "ERROR" if error else "OK"
        status_message = error or ""
        record = TraceRecord(
            trace_id=self.trace_id,
            span_id=_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            event_type=event_type,
            node=node,
            step_index=step_index,
            start_time_unix_nano=start_ns,
            end_time_unix_nano=end_ns,
            latency_ms=round(float(latency_ms), 6),
            cost_usd=round(float(cost_usd), 9),
            error=error,
            attributes=dict(attributes or {}),
            events=[dict(event) for event in events or []],
            status_code=status_code,
            status_message=status_message,
            resource={"service.name": self.service_name},
        )
        self.records.append(record)
        return record

    def finish(self, final_state: Mapping[str, Any]) -> None:
        self.end_time_unix_nano = _now_ns()

    def records_as_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def explorer_export(self, final_state: Mapping[str, Any]) -> dict[str, Any]:
        records = self.records_as_dicts()
        node_records = [record for record in records if record["event_type"] == "node"]
        event_records = [record for record in records if record["event_type"] != "node"]
        total_latency_ms = round(
            sum(float(record.get("latency_ms", 0.0)) for record in node_records),
            6,
        )
        total_cost_usd = round(
            sum(float(record.get("cost_usd", 0.0)) for record in records),
            9,
        )
        errors = [record for record in records if record.get("error") is not None]
        event_types = sorted({str(record.get("event_type")) for record in records})
        execution_summary = summarize_execution_state(final_state)
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "service_name": self.service_name,
            "input_state": json_safe(self.input_state),
            "execution_summary": execution_summary,
            "summary": {
                "record_count": len(records),
                "node_count": len(node_records),
                "event_count": len(event_records),
                "event_types": event_types,
                "total_latency_ms": total_latency_ms,
                "total_cost_usd": total_cost_usd,
                "error_count": len(errors),
            },
            "timeline": records,
            "nodes": node_records,
            "events": event_records,
            "replay": {
                "supported": True,
                "input_keys": sorted(str(key) for key in self.input_state.keys()),
                "comparison": "execution_summary_divergence",
            },
        }


class LocalTraceRecorder:
    """In-process trace recorder with no telemetry backend dependency."""

    def __init__(self, *, service_name: str = "orchardflow") -> None:
        self.service_name = service_name
        self._runs: dict[str, TraceRun] = {}

    def start_run(
        self,
        *,
        name: str,
        input_state: Mapping[str, Any],
    ) -> TraceRun:
        run = TraceRun(
            trace_id=_trace_id(),
            name=name,
            input_state=json_safe(input_state),
            service_name=self.service_name,
        )
        self._runs[run.trace_id] = run
        return run

    def get_run(self, trace_id: str) -> TraceRun:
        return self._runs[trace_id]

    def list_runs(self) -> list[TraceRun]:
        return list(self._runs.values())


def replay_trace(
    graph: Any,
    trace_export: Mapping[str, Any],
    *,
    modified_inputs: Mapping[str, Any] | None = None,
    max_steps: int = 50,
) -> dict[str, Any]:
    """Replay a trace through a graph and compare modified-input divergence."""

    base_inputs = dict(trace_export.get("input_state") or {})
    replay_inputs = dict(base_inputs)
    replay_inputs.update(dict(modified_inputs or {}))
    replay_state = graph.run(replay_inputs, max_steps=max_steps)
    original_summary = dict(trace_export.get("execution_summary") or {})
    replay_summary = summarize_execution_state(replay_state)
    divergence = compare_execution_summaries(original_summary, replay_summary)
    return {
        "original_trace_id": trace_export.get("trace_id"),
        "replay_trace_id": replay_state.get("trace_id"),
        "modified_inputs": json_safe(modified_inputs or {}),
        "original_summary": json_safe(original_summary),
        "replay_summary": json_safe(replay_summary),
        "divergence": divergence,
        "replay_trace_explorer": json_safe(replay_state.get("trace_explorer", {})),
    }


__all__ = [
    "LocalTraceRecorder",
    "TRACE_EVENT_TYPES",
    "TraceRecord",
    "TraceRun",
    "compare_execution_summaries",
    "estimate_cost_usd",
    "json_safe",
    "replay_trace",
    "summarize_execution_state",
]
