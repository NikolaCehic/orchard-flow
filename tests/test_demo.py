from __future__ import annotations

from pathlib import Path
import unittest

from orchardflow.demo import FINAL_PITCH, run_end_to_end_demo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EndToEndDemoTests(unittest.TestCase):
    def test_demo_shows_full_supervised_workflow_path(self) -> None:
        demo = run_end_to_end_demo()

        self.assertEqual(demo["pitch"], FINAL_PITCH)
        self.assertFalse(demo["requires_live_services"])
        self.assertEqual(demo["final_status"], "complete")
        self.assertIn("Research", demo["task_input"])

        plan_roles = [step["role"] for step in demo["supervisor_decomposition"]]
        self.assertEqual(plan_roles, ["research", "analysis", "writing", "code"])

        specialist_execution = demo["specialist_execution"]
        self.assertEqual(specialist_execution["design_target"], "parallel specialist delegation")
        self.assertEqual(specialist_execution["roles"], plan_roles)
        self.assertTrue(all(output["tool_calls"] == ["echo"] for output in specialist_execution["outputs"]))

        reviewer_route_back = demo["reviewer_route_back"]
        self.assertTrue(reviewer_route_back["observed"])
        routes = [decision["route"] for decision in reviewer_route_back["decisions"]]
        self.assertIn("approve", routes)
        self.assertEqual(routes[-1], "complete")

        memory_decision = demo["memory_informed_decision"]
        self.assertTrue(memory_decision["observed"])
        self.assertTrue(memory_decision["trace_has_memory_retrieval"])
        self.assertGreaterEqual(len(memory_decision["retrieved_memories"]), 1)

        human_approval = demo["human_approval"]
        self.assertEqual(human_approval["status"], "resolved")
        self.assertEqual(human_approval["decision"], "approved")
        self.assertEqual(human_approval["approval_level"], "Approve Plan")
        self.assertEqual(human_approval["context"]["pitch"], FINAL_PITCH)

        event_types = set(demo["trace"]["event_types"])
        self.assertIn("planning_decision", event_types)
        self.assertIn("memory_retrieval", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("node", event_types)


class ContainerAndDocumentationTests(unittest.TestCase):
    def test_docker_artifacts_define_local_services_without_live_secrets(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        compose = (PROJECT_ROOT / "docker-compose.yml").read_text()
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text()
        combined = "\n".join([dockerfile, compose, dockerignore])

        self.assertIn("python:3.11-slim", dockerfile)
        self.assertIn('"python", "-m", "orchardflow.demo"', dockerfile)
        self.assertIn("redis:", compose)
        self.assertIn("postgres:", compose)
        self.assertIn("chroma:", compose)
        self.assertIn("OPENAI_API_KEY:-placeholder", compose)
        self.assertIn("ANTHROPIC_API_KEY:-placeholder", compose)
        self.assertIn(".env", dockerignore)

        forbidden_secret_markers = (
            "sk-",
            "xoxb-",
            "-----BEGIN",
            "live_",
            "anthropic_live",
        )
        for marker in forbidden_secret_markers:
            self.assertNotIn(marker, combined)

    def test_architecture_docs_include_final_pitch(self) -> None:
        architecture = (PROJECT_ROOT / "docs" / "architecture.md").read_text()

        self.assertIn(FINAL_PITCH, architecture)
        self.assertIn("Supervisor", architecture)
        self.assertIn("Human Approval", architecture)
        self.assertIn("Redis", architecture)
        self.assertIn("PostgreSQL", architecture)
        self.assertIn("ChromaDB", architecture)


if __name__ == "__main__":
    unittest.main()
