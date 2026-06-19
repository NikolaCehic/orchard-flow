"""Deterministic memory stores for OrchardFlow WU-102.

The classes in this module are local, dependency-free implementations with
adapter names that match the target production stack. They let tests exercise
Redis, PostgreSQL, and ChromaDB integration boundaries without requiring live
services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Mapping, Protocol, Sequence


MemoryMetadata = Mapping[str, Any]


def _now() -> float:
    return time.time()


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, (list, tuple, set)):
            rendered = " ".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            rendered = _metadata_text(value)
        else:
            rendered = str(value)
        parts.append(f"{key} {rendered}")
    return " ".join(parts)


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _decay_factor(age_seconds: float, half_life_seconds: float) -> float:
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if age_seconds <= 0:
        return 1.0
    return 0.5 ** (age_seconds / half_life_seconds)


@dataclass
class MemoryRecord:
    id: str
    memory_type: str
    content: str
    task_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 1.0
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_accessed_at: float | None = None
    access_count: int = 0
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("memory id cannot be empty")
        if not self.memory_type.strip():
            raise ValueError("memory_type cannot be empty")
        if not self.content.strip():
            raise ValueError("content cannot be empty")
        self.importance = _clamp(float(self.importance))
        self.confidence = _clamp(float(self.confidence))

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.memory_type,
                self.task_id or "",
                self.user_id or "",
                self.content,
                _metadata_text(self.metadata),
            )
            if part
        )

    def is_expired(self, *, now: float | None = None) -> bool:
        current_time = _now() if now is None else now
        return self.expires_at is not None and current_time >= self.expires_at

    def mark_accessed(self, *, now: float | None = None) -> None:
        current_time = _now() if now is None else now
        self.last_accessed_at = current_time
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class MemoryQueryResult:
    record: MemoryRecord
    score: float
    relevance_score: float
    importance_score: float
    confidence_score: float
    recency_score: float
    access_score: float
    decay_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": self.score,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "recency_score": self.recency_score,
            "access_score": self.access_score,
            "decay_factor": self.decay_factor,
        }


class LongTermMemoryStore(Protocol):
    def query(
        self,
        query_text: str,
        *,
        limit: int = 5,
        memory_types: Sequence[str] | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        """Return ranked memories for planning or recall."""


class LocalMemoryScorer:
    """Lexical scorer with deterministic relevance, recency, and decay."""

    def __init__(self, *, half_life_seconds: float = 86_400.0) -> None:
        self.half_life_seconds = half_life_seconds

    def score(self, record: MemoryRecord, query_text: str, *, now: float | None = None) -> MemoryQueryResult:
        current_time = _now() if now is None else now
        relevance = _similarity(query_text, record.searchable_text) if query_text.strip() else 1.0
        age_seconds = max(current_time - record.created_at, 0.0)
        decay = _decay_factor(age_seconds, self.half_life_seconds)
        recency = decay
        access = min(record.access_count / 10.0, 1.0)
        score = (
            0.5 * relevance
            + 0.2 * record.importance
            + 0.15 * record.confidence
            + 0.1 * recency
            + 0.05 * access
        ) * decay
        return MemoryQueryResult(
            record=record,
            score=round(score, 6),
            relevance_score=round(relevance, 6),
            importance_score=round(record.importance, 6),
            confidence_score=round(record.confidence, 6),
            recency_score=round(recency, 6),
            access_score=round(access, 6),
            decay_factor=round(decay, 6),
        )


class LocalShortTermMemoryStore:
    """Task-scoped working memory with Redis-compatible semantics."""

    integration_name = "local-short-term"

    def __init__(
        self,
        *,
        namespace: str = "orchardflow:memory:short",
        default_ttl_seconds: float = 3_600.0,
        scorer: LocalMemoryScorer | None = None,
    ) -> None:
        self.namespace = namespace
        self.default_ttl_seconds = default_ttl_seconds
        self.scorer = scorer or LocalMemoryScorer(half_life_seconds=default_ttl_seconds)
        self._counter = 0
        self._records: dict[str, MemoryRecord] = {}

    def record(
        self,
        *,
        task_id: str,
        content: str,
        memory_type: str = "working",
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        now: float | None = None,
        ttl_seconds: float | None = None,
    ) -> MemoryRecord:
        if not task_id.strip():
            raise ValueError("task_id is required for short-term memory")
        current_time = _now() if now is None else now
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._counter += 1
        record = MemoryRecord(
            id=f"{self.namespace}:{task_id}:{self._counter}",
            memory_type=memory_type,
            content=content,
            task_id=task_id,
            metadata=dict(metadata or {}),
            importance=importance,
            confidence=confidence,
            created_at=current_time,
            updated_at=current_time,
            expires_at=current_time + ttl if ttl > 0 else None,
        )
        self._records[record.id] = record
        return record

    def query(
        self,
        task_id: str,
        query_text: str,
        *,
        limit: int = 5,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        current_time = _now() if now is None else now
        self.expire(now=current_time)
        results: list[MemoryQueryResult] = []
        for record in self._records.values():
            if record.task_id != task_id:
                continue
            result = self.scorer.score(record, query_text, now=current_time)
            if query_text.strip() and result.relevance_score <= 0:
                continue
            results.append(result)
        results.sort(key=lambda item: (-item.score, item.record.created_at, item.record.id))
        for result in results[:limit]:
            result.record.mark_accessed(now=current_time)
        return results[:limit]

    def list_task(self, task_id: str, *, now: float | None = None) -> list[MemoryRecord]:
        current_time = _now() if now is None else now
        self.expire(now=current_time)
        return [
            record
            for record in self._records.values()
            if record.task_id == task_id and not record.is_expired(now=current_time)
        ]

    def clear_task(self, task_id: str) -> int:
        record_ids = [record_id for record_id, record in self._records.items() if record.task_id == task_id]
        for record_id in record_ids:
            del self._records[record_id]
        return len(record_ids)

    def expire(self, *, now: float | None = None) -> int:
        current_time = _now() if now is None else now
        expired_ids = [
            record_id
            for record_id, record in self._records.items()
            if record.is_expired(now=current_time)
        ]
        for record_id in expired_ids:
            del self._records[record_id]
        return len(expired_ids)


class RedisShortTermMemoryStore(LocalShortTermMemoryStore):
    """Redis integration boundary backed by the deterministic local store."""

    integration_name = "redis"

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379/0",
        namespace: str = "orchardflow:memory:short",
        default_ttl_seconds: float = 3_600.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(namespace=namespace, default_ttl_seconds=default_ttl_seconds)
        self.redis_url = redis_url
        self.client = client

    def connection_config(self) -> dict[str, Any]:
        return {
            "integration": self.integration_name,
            "redis_url": self.redis_url,
            "namespace": self.namespace,
            "has_live_client": self.client is not None,
            "local_fallback": True,
        }


class LocalLongTermSemanticMemoryStore(LongTermMemoryStore):
    """Local semantic memory for outcomes, tool use, and preferences."""

    integration_name = "local-long-term"

    def __init__(
        self,
        *,
        namespace: str = "orchardflow:memory:long",
        scorer: LocalMemoryScorer | None = None,
    ) -> None:
        self.namespace = namespace
        self.scorer = scorer or LocalMemoryScorer()
        self._counter = 0
        self._records: dict[str, MemoryRecord] = {}

    def record(
        self,
        *,
        content: str,
        memory_type: str,
        task_id: str | None = None,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> MemoryRecord:
        current_time = _now() if now is None else now
        self._counter += 1
        record = MemoryRecord(
            id=f"{self.namespace}:{self._counter}",
            memory_type=memory_type,
            content=content,
            task_id=task_id,
            user_id=user_id,
            metadata=dict(metadata or {}),
            importance=importance,
            confidence=confidence,
            created_at=current_time,
            updated_at=current_time,
        )
        self._records[record.id] = record
        return record

    def record_outcome(
        self,
        *,
        task_id: str,
        content: str,
        worked: bool,
        tools_used: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.7,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> MemoryRecord:
        merged_metadata = dict(metadata or {})
        merged_metadata.update({"worked": worked, "tools_used": list(tools_used or [])})
        return self.record(
            content=content,
            memory_type="outcome",
            task_id=task_id,
            metadata=merged_metadata,
            importance=importance,
            confidence=confidence,
            now=now,
        )

    def record_tool_use(
        self,
        *,
        task_id: str,
        tool_name: str,
        content: str,
        success: bool,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.6,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> MemoryRecord:
        merged_metadata = dict(metadata or {})
        merged_metadata.update({"tool_name": tool_name, "success": success})
        return self.record(
            content=content,
            memory_type="tool_use",
            task_id=task_id,
            metadata=merged_metadata,
            importance=importance,
            confidence=confidence,
            now=now,
        )

    def record_user_preference(
        self,
        *,
        user_id: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        importance: float = 0.8,
        confidence: float = 1.0,
        now: float | None = None,
    ) -> MemoryRecord:
        return self.record(
            content=content,
            memory_type="user_preference",
            user_id=user_id,
            metadata=dict(metadata or {}),
            importance=importance,
            confidence=confidence,
            now=now,
        )

    def query(
        self,
        query_text: str,
        *,
        limit: int = 5,
        memory_types: Sequence[str] | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        current_time = _now() if now is None else now
        allowed_types = set(memory_types or [])
        results: list[MemoryQueryResult] = []
        for record in self._records.values():
            if allowed_types and record.memory_type not in allowed_types:
                continue
            if task_id is not None and record.task_id != task_id:
                continue
            if user_id is not None and record.user_id not in (None, user_id):
                continue
            result = self.scorer.score(record, query_text, now=current_time)
            if query_text.strip() and result.relevance_score <= 0:
                continue
            results.append(result)
        results.sort(key=lambda item: (-item.score, item.record.created_at, item.record.id))
        selected = results[:limit]
        for result in selected:
            result.record.mark_accessed(now=current_time)
        return selected

    def consolidate(
        self,
        *,
        min_similarity: float = 0.35,
        memory_types: Sequence[str] | None = None,
        now: float | None = None,
    ) -> list[MemoryRecord]:
        current_time = _now() if now is None else now
        allowed_types = set(memory_types or [])
        existing_clusters = {
            tuple(record.metadata.get("source_ids", []))
            for record in self._records.values()
            if record.metadata.get("consolidated")
        }
        source_records = [
            record
            for record in self._records.values()
            if not record.metadata.get("consolidated")
            and (not allowed_types or record.memory_type in allowed_types)
        ]
        used: set[str] = set()
        consolidated: list[MemoryRecord] = []
        for record in sorted(source_records, key=lambda item: (item.memory_type, item.created_at, item.id)):
            if record.id in used:
                continue
            cluster = [record]
            for candidate in source_records:
                if candidate.id == record.id or candidate.id in used:
                    continue
                if candidate.memory_type != record.memory_type:
                    continue
                if _similarity(record.searchable_text, candidate.searchable_text) >= min_similarity:
                    cluster.append(candidate)
            if len(cluster) < 2:
                continue
            source_ids = tuple(sorted(item.id for item in cluster))
            if source_ids in existing_clusters:
                used.update(source_ids)
                continue
            for item in cluster:
                used.add(item.id)
            average_importance = sum(item.importance for item in cluster) / len(cluster)
            average_confidence = sum(item.confidence for item in cluster) / len(cluster)
            content = "Consolidated memory ({memory_type}): {items}".format(
                memory_type=record.memory_type,
                items=" | ".join(item.content for item in cluster),
            )
            consolidated.append(
                self.record(
                    content=content,
                    memory_type=record.memory_type,
                    metadata={
                        "consolidated": True,
                        "source_ids": list(source_ids),
                        "source_count": len(cluster),
                    },
                    importance=min(1.0, average_importance + 0.1),
                    confidence=average_confidence,
                    now=current_time,
                )
            )
        return consolidated

    def apply_decay(
        self,
        *,
        now: float | None = None,
        half_life_seconds: float = 86_400.0,
        prune_below: float | None = None,
    ) -> list[MemoryRecord]:
        current_time = _now() if now is None else now
        decayed: list[MemoryRecord] = []
        pruned_ids: list[str] = []
        for record in self._records.values():
            age_seconds = max(current_time - record.created_at, 0.0)
            factor = _decay_factor(age_seconds, half_life_seconds)
            record.importance = round(_clamp(record.importance * factor), 6)
            record.updated_at = current_time
            decayed.append(record)
            if prune_below is not None and record.importance < prune_below:
                pruned_ids.append(record.id)
        for record_id in pruned_ids:
            del self._records[record_id]
        return decayed

    def all_records(self) -> list[MemoryRecord]:
        return list(self._records.values())


class PostgreSQLMemoryRepository(LocalLongTermSemanticMemoryStore):
    """PostgreSQL integration boundary backed by the deterministic local store."""

    integration_name = "postgresql"

    def __init__(
        self,
        *,
        dsn: str = "postgresql://localhost/orchardflow",
        table_name: str = "agent_memories",
        namespace: str = "orchardflow:memory:postgres",
    ) -> None:
        super().__init__(namespace=namespace)
        self.dsn = dsn
        self.table_name = table_name

    def connection_config(self) -> dict[str, Any]:
        return {
            "integration": self.integration_name,
            "dsn": self.dsn,
            "table_name": self.table_name,
            "local_fallback": True,
        }


class ChromaDBSemanticMemoryStore(LocalLongTermSemanticMemoryStore):
    """ChromaDB integration boundary backed by deterministic lexical search."""

    integration_name = "chromadb"

    def __init__(
        self,
        *,
        collection_name: str = "orchardflow_memories",
        persist_directory: str | None = None,
        namespace: str = "orchardflow:memory:chroma",
    ) -> None:
        super().__init__(namespace=namespace)
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def collection_config(self) -> dict[str, Any]:
        return {
            "integration": self.integration_name,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding": "deterministic-token-overlap-local",
            "local_fallback": True,
        }


@dataclass
class MemorySystem:
    short_term: LocalShortTermMemoryStore = field(default_factory=RedisShortTermMemoryStore)
    long_term: LocalLongTermSemanticMemoryStore = field(default_factory=ChromaDBSemanticMemoryStore)

    def query_for_planning(
        self,
        task: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        return self.long_term.query(task, user_id=user_id, limit=limit, now=now)


def build_local_memory_system() -> MemorySystem:
    return MemorySystem(
        short_term=RedisShortTermMemoryStore(),
        long_term=ChromaDBSemanticMemoryStore(),
    )


__all__ = [
    "ChromaDBSemanticMemoryStore",
    "LocalLongTermSemanticMemoryStore",
    "LocalMemoryScorer",
    "LocalShortTermMemoryStore",
    "LongTermMemoryStore",
    "MemoryQueryResult",
    "MemoryRecord",
    "MemorySystem",
    "PostgreSQLMemoryRepository",
    "RedisShortTermMemoryStore",
    "build_local_memory_system",
]
