from __future__ import annotations

import unittest

from orchardflow.agents import create_agent_graph
from orchardflow.providers import FakeProvider
from orchardflow.review import APPROVAL_LEVELS, LocalReviewQueue
from review_app import REVIEW_UI_FIELDS, format_review_request_for_display


class ReviewQueueTests(unittest.TestCase):
    def test_sensitive_operation_pauses_and_packages_review_context(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.95))

        result = graph.run({"task": "Deploy the payment workflow update"})

        self.assertEqual(result["final_status"], "escalated")
        self.assertTrue(result["execution_paused"])
        self.assertEqual(result["review_queue_status"], "queued")
        self.assertEqual(graph.review_queue.pending_count(), 1)

        request = result["review_request"]
        self.assertEqual(request["trigger"], "sensitive_operation")
        self.assertEqual(request["approval_level"], "Approve Action")
        self.assertIn("payment", request["reasoning"])
        self.assertEqual(request["context"]["task"], "Deploy the payment workflow update")
        self.assertEqual(request["context"]["status"], "escalated")
        self.assertTrue(request["proposed_action"])

    def test_low_confidence_review_package_includes_relevant_memories(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.95))
        state = {
            "task": "Analyze launch risk",
            "messages": [],
            "specialist_outputs": [],
            "review_decisions": [],
            "retry_counts": {},
            "current_output": {
                "role": "analysis",
                "content": "Risk analysis is inconclusive.",
                "confidence": 0.2,
                "requires_human": False,
            },
            "relevant_memories": [
                {
                    "record": {
                        "id": "memory-1",
                        "content": "Past launch reviews required explicit approval.",
                    },
                    "score": 0.91,
                }
            ],
        }

        result = graph.reviewer_node(state)

        self.assertEqual(result["status"], "escalated")
        self.assertTrue(result["execution_paused"])
        request = result["review_request"]
        self.assertEqual(request["trigger"], "low_confidence")
        self.assertEqual(request["approval_level"], "Approve Action")
        self.assertEqual(request["relevant_memories"][0]["record"]["id"], "memory-1")
        self.assertEqual(request["context"]["current_output"]["content"], "Risk analysis is inconclusive.")

    def test_repeated_failure_uses_take_over_approval_level(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.95))

        result = graph.run(
            {
                "task": "Write the final summary",
                "failure_count": 2,
            }
        )

        self.assertEqual(result["final_status"], "escalated")
        request = result["review_request"]
        self.assertEqual(request["trigger"], "repeated_failure")
        self.assertEqual(request["approval_level"], "Take Over")
        self.assertIn("consecutive failures", request["reasoning"])

    def test_low_quality_score_pauses_into_review_queue(self) -> None:
        queue = LocalReviewQueue()
        graph = create_agent_graph(
            provider=FakeProvider(confidence=0.9),
            review_queue=queue,
        )

        result = graph.run(
            {
                "task": "Analyze implementation tradeoffs",
                "force_quality_score": 0.2,
            }
        )

        self.assertEqual(result["final_status"], "escalated")
        self.assertEqual(queue.pending_count(), 1)
        request = queue.pending()[0].to_dict()
        self.assertEqual(request["trigger"], "low_quality_score")
        self.assertEqual(request["approval_level"], "Approve Plan")
        self.assertIn("quality score", request["reasoning"])


class ReviewAppTests(unittest.TestCase):
    def test_review_ui_display_contract_has_required_fields_and_levels(self) -> None:
        request = {
            "id": "review-1",
            "status": "queued",
            "trigger": "low_quality_score",
            "approval_level": "Approve Plan",
            "context": {"task": "Check a paused run"},
            "proposed_action": "Approve a revised plan.",
            "reasoning": "Quality score was below threshold.",
            "relevant_memories": [{"record": {"content": "Use direct evidence."}}],
        }

        display = format_review_request_for_display(request)

        self.assertEqual(display["context"]["task"], "Check a paused run")
        self.assertEqual(display["proposed_action"], "Approve a revised plan.")
        self.assertEqual(display["reasoning"], "Quality score was below threshold.")
        self.assertEqual(display["relevant_memories"][0]["record"]["content"], "Use direct evidence.")
        self.assertEqual(display["approval_levels"], list(APPROVAL_LEVELS))
        self.assertEqual(
            display["approval_levels"],
            ["Notify", "Approve Action", "Approve Plan", "Take Over"],
        )
        for field in REVIEW_UI_FIELDS:
            self.assertIn(field, display)


if __name__ == "__main__":
    unittest.main()
