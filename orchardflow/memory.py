"""Compatibility memory interfaces for the WU-101 agent graph.

The concrete memory stores land in WU-102. These interfaces keep the graph
importable and type-compatible for the first stacked PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    content: str = ""
    memory_type: str = "compat"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
        }


@dataclass(frozen=True)
class MemoryQueryResult:
    record: MemoryRecord
    score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "record": self.record.to_dict(),
            "score": self.score,
        }


class LongTermMemoryStore(Protocol):
    def query(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        """Return planning memories relevant to a task."""
