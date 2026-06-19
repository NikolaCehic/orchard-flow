"""Compatibility tracing helpers for WU-101.

The full observability and replay implementation lands in WU-104.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping
from uuid import uuid4


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def estimate_cost_usd(
    usage: Mapping[str, int],
    *,
    provider: str,
) -> float:
    return 0.0


@dataclass
class TraceRun:
    name: str
    input_state: Mapping[str, Any]
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    records: list[dict[str, Any]] = field(default_factory=list)

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
    ) -> dict[str, Any]:
        start_ns = start_time_unix_nano or time.time_ns()
        end_ns = end_time_unix_nano or start_ns
        if latency_ms is None:
            latency_ms = round(max((end_ns - start_ns) / 1_000_000, 0.0), 6)
        record = {
            "trace_id": self.trace_id,
            "span_id": uuid4().hex,
            "name": name,
            "event_type": event_type,
            "node": node,
            "step_index": step_index,
            "start_time_unix_nano": start_ns,
            "end_time_unix_nano": end_ns,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "error": error,
            "status": "error" if error else "ok",
            "attributes": json_safe(dict(attributes or {})),
        }
        self.records.append(record)
        return record

    def records_as_dicts(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records]

    def finish(self, final_state: Mapping[str, Any]) -> None:
        return None

    def explorer_export(self, final_state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "summary": {"record_count": len(self.records)},
            "timeline": self.records_as_dicts(),
            "replay": {"supported": False},
        }


class LocalTraceRecorder:
    def start_run(
        self,
        *,
        name: str,
        input_state: Mapping[str, Any],
    ) -> TraceRun:
        return TraceRun(name=name, input_state=dict(input_state))
