from __future__ import annotations

import unittest

from orchardflow.agents import create_agent_graph
from orchardflow.memory import ChromaDBSemanticMemoryStore
from orchardflow.observability import replay_trace
from orchardflow.providers import FakeProvider


class ObservabilityTests(unittest.TestCase):
    def test_trace_records_cover_planning_memory_tools_latency_cost_and_errors(self) -> None:
        memory = ChromaDBSemanticMemoryStore(collection_name="observability")
        memory.record_user_preference(
            user_id="user-1",
            content="Prefer concise acceptance evidence in summaries.",
            importance=0.9,
            now=100.0,
        )
        graph = create_agent_graph(
            provider=FakeProvider(confidence=0.96),
            long_term_memory=memory,
        )

        result = graph.run(
            {
                "task": "Research, analyze, and write a concise summary",
                "user_id": "user-1",
                "memory_now": 130.0,
            }
        )

        self.assertEqual(result["final_status"], "complete")
        records = result["trace_records"]
        event_types = {record["event_type"] for record in records}

        self.assertIn("node", event_types)
        self.assertIn("planning_decision", event_types)
        self.assertIn("memory_retrieval", event_types)
        self.assertIn("tool_call", event_types)

        for record in records:
            self.assertIn("trace_id", record)
            self.assertIn("span_id", record)
            self.assertIn("start_time_unix_nano", record)
            self.assertIn("end_time_unix_nano", record)
            self.assertIn("latency_ms", record)
            self.assertIn("cost_usd", record)
            self.assertIn("error", record)
            self.assertIn("status", record)
            self.assertGreaterEqual(record["latency_ms"], 0.0)
            self.assertGreaterEqual(record["cost_usd"], 0.0)

        retrievals = [
            record
            for record in records
            if record["event_type"] == "memory_retrieval"
        ]
        self.assertEqual(
            retrievals[0]["attributes"]["memory.retrieved_count"],
            1,
        )
        explorer = result["trace_explorer"]
        self.assertEqual(explorer["trace_id"], result["trace_id"])
        self.assertTrue(explorer["replay"]["supported"])
        self.assertGreater(explorer["summary"]["node_count"], 0)

    def test_escalation_events_are_recorded_for_human_review_pauses(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.95))

        result = graph.run({"task": "Deploy the payment workflow update"})

        self.assertEqual(result["final_status"], "escalated")
        escalation_records = [
            record
            for record in result["trace_records"]
            if record["event_type"] == "escalation_event"
        ]
        self.assertEqual(len(escalation_records), 1)
        self.assertEqual(
            escalation_records[0]["attributes"]["escalation.trigger"],
            "sensitive_operation",
        )
        self.assertEqual(escalation_records[0]["error"], None)

    def test_trace_replay_runs_modified_inputs_and_reports_divergence(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.96))
        original = graph.run({"task": "Write a summary"})

        replay = replay_trace(
            graph,
            original["trace_explorer"],
            modified_inputs={"task": "Deploy the payment workflow update"},
        )

        self.assertNotEqual(
            replay["original_trace_id"],
            replay["replay_trace_id"],
        )
        self.assertTrue(replay["divergence"]["changed"])
        changed_fields = {
            change["field"] for change in replay["divergence"]["changes"]
        }
        self.assertIn("task", changed_fields)
        self.assertIn("final_status", changed_fields)
        self.assertEqual(replay["original_summary"]["final_status"], "complete")
        self.assertEqual(replay["replay_summary"]["final_status"], "escalated")
        self.assertGreater(
            len(replay["replay_trace_explorer"]["timeline"]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
