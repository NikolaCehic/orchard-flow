from __future__ import annotations

import unittest

from orchardflow.agents import create_agent_graph
from orchardflow.memory import (
    ChromaDBSemanticMemoryStore,
    PostgreSQLMemoryRepository,
    RedisShortTermMemoryStore,
)
from orchardflow.providers import FakeProvider
from orchardflow.queueing import CeleryTaskQueue, RedisQueueBroker, RedisQueueConfig


class MemoryStoreTests(unittest.TestCase):
    def test_memory_record_query_scoring_consolidation_and_decay(self) -> None:
        short_term = RedisShortTermMemoryStore(
            redis_url="redis://localhost:6379/0",
            default_ttl_seconds=60,
        )
        short_term.record(
            task_id="task-1",
            content="Reviewer approved the concise writing plan.",
            importance=0.8,
            now=100.0,
        )
        short_term.record(
            task_id="task-2",
            content="Unrelated research note for a different task.",
            importance=0.9,
            now=100.0,
        )

        short_results = short_term.query("task-1", "concise writing plan", now=105.0)

        self.assertEqual(short_term.connection_config()["integration"], "redis")
        self.assertEqual(len(short_results), 1)
        self.assertEqual(short_results[0].record.task_id, "task-1")
        self.assertGreater(short_results[0].score, 0)
        self.assertGreater(short_results[0].relevance_score, 0)

        long_term = ChromaDBSemanticMemoryStore(collection_name="orchardflow_test")
        long_term.record_outcome(
            task_id="task-1",
            content="Concise writing specialist plans worked for executive summaries.",
            worked=True,
            tools_used=["echo"],
            importance=0.9,
            now=100.0,
        )
        long_term.record_tool_use(
            task_id="task-1",
            tool_name="echo",
            content="Echo tool safely returned deterministic local text.",
            success=True,
            now=110.0,
        )
        long_term.record_user_preference(
            user_id="user-1",
            content="User prefers concise summaries with direct acceptance evidence.",
            now=120.0,
        )
        old_record = long_term.record_outcome(
            task_id="task-old",
            content="Stale verbose planning style had weak results.",
            worked=False,
            importance=0.8,
            now=0.0,
        )

        results = long_term.query(
            "concise summaries acceptance evidence",
            user_id="user-1",
            now=130.0,
        )

        self.assertEqual(long_term.collection_config()["integration"], "chromadb")
        self.assertEqual(results[0].record.memory_type, "user_preference")
        self.assertGreater(results[0].score, results[-1].score)
        self.assertGreater(results[0].decay_factor, 0)

        long_term.record_outcome(
            task_id="task-2",
            content="Reviewer retry plan worked for concise summaries.",
            worked=True,
            importance=0.7,
            now=140.0,
        )
        long_term.record_outcome(
            task_id="task-3",
            content="Reviewer retry plan worked for concise executive summaries.",
            worked=True,
            importance=0.7,
            now=145.0,
        )

        consolidated = long_term.consolidate(min_similarity=0.35, now=150.0)

        self.assertTrue(consolidated)
        self.assertTrue(consolidated[0].metadata["consolidated"])
        self.assertGreaterEqual(consolidated[0].metadata["source_count"], 2)

        before_decay = old_record.importance
        long_term.apply_decay(now=7_200.0, half_life_seconds=3_600.0)

        self.assertLess(old_record.importance, before_decay)

    def test_postgresql_repository_exposes_local_long_term_interface(self) -> None:
        repository = PostgreSQLMemoryRepository(
            dsn="postgresql://localhost/orchardflow_test",
            table_name="agent_memories_test",
        )

        repository.record_outcome(
            task_id="task-1",
            content="PostgreSQL repository stores durable outcome metadata.",
            worked=True,
            now=100.0,
        )
        results = repository.query("durable outcome metadata", now=101.0)

        self.assertEqual(repository.connection_config()["integration"], "postgresql")
        self.assertEqual(results[0].record.memory_type, "outcome")


class QueueTests(unittest.TestCase):
    def test_local_queue_enqueue_dequeue_with_redis_celery_interfaces(self) -> None:
        broker = RedisQueueBroker(
            RedisQueueConfig(redis_url="redis://localhost:6379/1", namespace="test")
        )
        queue = CeleryTaskQueue(broker=broker, app_name="orchardflow-test")

        first = queue.enqueue(
            "memory.consolidate",
            {"task_id": "task-1"},
            max_retries=1,
            now=100.0,
        )
        second = queue.enqueue(
            "memory.decay",
            {"half_life_seconds": 3600},
            now=101.0,
        )

        self.assertEqual(broker.connection_config()["integration"], "redis")
        self.assertEqual(queue.celery_config()["integration"], "celery")
        self.assertEqual(queue.pending_count(), 2)

        dequeued_first = queue.dequeue(now=102.0)
        dequeued_second = queue.dequeue(now=102.0)

        self.assertEqual(dequeued_first.id, first.id)
        self.assertEqual(dequeued_second.id, second.id)
        self.assertEqual(dequeued_first.status, "in_progress")
        self.assertEqual(queue.task_signature(first)["task"], "memory.consolidate")

        queue.fail(first.id, error="transient", retry_delay_seconds=5, now=103.0)
        self.assertEqual(first.status, "queued")
        self.assertIsNone(queue.dequeue(now=104.0))
        self.assertEqual(queue.dequeue(now=108.0).id, first.id)
        queue.complete(first.id, result={"ok": True}, now=109.0)
        self.assertEqual(first.status, "complete")
        self.assertEqual(first.result, {"ok": True})


class SupervisorMemoryTests(unittest.TestCase):
    def test_supervisor_queries_long_term_memory_at_planning_time(self) -> None:
        memory = ChromaDBSemanticMemoryStore(collection_name="planning")
        memory.record_user_preference(
            user_id="user-1",
            content="For concise summaries, prefer direct acceptance evidence.",
            importance=0.9,
            now=100.0,
        )
        graph = create_agent_graph(
            provider=FakeProvider(confidence=0.95),
            long_term_memory=memory,
            planning_memory_limit=3,
        )

        result = graph.run(
            {
                "task": "Write a concise summary with acceptance evidence",
                "user_id": "user-1",
                "memory_now": 130.0,
            }
        )

        self.assertEqual(result["final_status"], "complete")
        self.assertEqual(len(result["relevant_memories"]), 1)
        retrieved = result["relevant_memories"][0]
        self.assertEqual(retrieved["record"]["memory_type"], "user_preference")
        self.assertGreater(retrieved["score"], 0)


if __name__ == "__main__":
    unittest.main()
