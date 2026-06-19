"""Local queue behavior and Redis/Celery integration boundaries for WU-102."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class RedisQueueConfig:
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "orchardflow:queue"

    def to_dict(self) -> dict[str, str]:
        return {
            "integration": "redis",
            "redis_url": self.redis_url,
            "namespace": self.namespace,
        }


@dataclass
class QueueTask:
    id: str
    name: str
    payload: dict[str, Any]
    queue_name: str = "default"
    status: str = "queued"
    attempts: int = 0
    max_retries: int = 0
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    available_at: float = field(default_factory=_now)
    result: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "payload": dict(self.payload),
            "queue_name": self.queue_name,
            "status": self.status,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "available_at": self.available_at,
            "result": self.result,
            "error": self.error,
        }


class QueueError(RuntimeError):
    """Base class for queue failures."""


class QueueTaskNotFound(QueueError):
    """Raised when a task id is unknown to the queue."""


class LocalTaskQueue:
    """Deterministic in-process FIFO queue."""

    integration_name = "local"

    def __init__(self, *, default_queue: str = "default") -> None:
        self.default_queue = default_queue
        self._counter = 0
        self._tasks: dict[str, QueueTask] = {}
        self._order: list[str] = []

    def enqueue(
        self,
        name: str,
        payload: Mapping[str, Any] | None = None,
        *,
        queue_name: str | None = None,
        delay_seconds: float = 0.0,
        max_retries: int = 0,
        now: float | None = None,
    ) -> QueueTask:
        if not name.strip():
            raise ValueError("task name cannot be empty")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        current_time = _now() if now is None else now
        selected_queue = queue_name or self.default_queue
        self._counter += 1
        task = QueueTask(
            id=f"{selected_queue}:{self._counter}",
            name=name,
            payload=dict(payload or {}),
            queue_name=selected_queue,
            status="queued",
            max_retries=max_retries,
            created_at=current_time,
            updated_at=current_time,
            available_at=current_time + max(delay_seconds, 0.0),
        )
        self._tasks[task.id] = task
        self._order.append(task.id)
        return task

    def dequeue(
        self,
        *,
        queue_name: str | None = None,
        now: float | None = None,
    ) -> QueueTask | None:
        current_time = _now() if now is None else now
        selected_queue = queue_name or self.default_queue
        for task_id in self._order:
            task = self._tasks[task_id]
            if task.queue_name != selected_queue:
                continue
            if task.status != "queued":
                continue
            if task.available_at > current_time:
                continue
            task.status = "in_progress"
            task.attempts += 1
            task.updated_at = current_time
            return task
        return None

    def complete(
        self,
        task_id: str,
        *,
        result: Any | None = None,
        now: float | None = None,
    ) -> QueueTask:
        task = self.get(task_id)
        task.status = "complete"
        task.result = result
        task.error = None
        task.updated_at = _now() if now is None else now
        return task

    def fail(
        self,
        task_id: str,
        *,
        error: str,
        retry_delay_seconds: float = 0.0,
        now: float | None = None,
    ) -> QueueTask:
        task = self.get(task_id)
        current_time = _now() if now is None else now
        task.error = error
        task.updated_at = current_time
        if task.attempts <= task.max_retries:
            task.status = "queued"
            task.available_at = current_time + max(retry_delay_seconds, 0.0)
        else:
            task.status = "failed"
        return task

    def get(self, task_id: str) -> QueueTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise QueueTaskNotFound(f"Queue task {task_id!r} was not found") from exc

    def pending_count(self, *, queue_name: str | None = None) -> int:
        selected_queue = queue_name or self.default_queue
        return sum(
            1
            for task in self._tasks.values()
            if task.queue_name == selected_queue and task.status == "queued"
        )

    def all_tasks(self) -> list[QueueTask]:
        return [self._tasks[task_id] for task_id in self._order]


class RedisQueueBroker:
    """Redis broker integration point with no live connection requirement."""

    integration_name = "redis"

    def __init__(self, config: RedisQueueConfig | None = None, *, client: Any | None = None) -> None:
        self.config = config or RedisQueueConfig()
        self.client = client

    def connection_config(self) -> dict[str, Any]:
        config = self.config.to_dict()
        config.update({"has_live_client": self.client is not None, "local_fallback": True})
        return config


class CeleryTaskQueue(LocalTaskQueue):
    """Celery integration boundary backed by the local FIFO queue."""

    integration_name = "celery"

    def __init__(
        self,
        *,
        broker: RedisQueueBroker | None = None,
        app_name: str = "orchardflow",
        default_queue: str = "default",
    ) -> None:
        super().__init__(default_queue=default_queue)
        self.broker = broker or RedisQueueBroker()
        self.app_name = app_name

    def celery_config(self) -> dict[str, Any]:
        return {
            "integration": self.integration_name,
            "app_name": self.app_name,
            "broker_url": self.broker.config.redis_url,
            "task_default_queue": self.default_queue,
            "result_backend": self.broker.config.redis_url,
            "local_fallback": True,
        }

    def task_signature(self, task: QueueTask) -> dict[str, Any]:
        return {
            "task": task.name,
            "id": task.id,
            "queue": task.queue_name,
            "args": [],
            "kwargs": dict(task.payload),
            "celery_app": self.app_name,
        }


__all__ = [
    "CeleryTaskQueue",
    "LocalTaskQueue",
    "QueueError",
    "QueueTask",
    "QueueTaskNotFound",
    "RedisQueueBroker",
    "RedisQueueConfig",
]
