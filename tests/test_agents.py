from __future__ import annotations

import unittest

from orchardflow.agents import OrchardAgentGraph, create_agent_graph
from orchardflow.providers import (
    AnthropicProvider,
    FakeProvider,
    OpenAIProvider,
    ProviderConfigurationError,
)
from orchardflow.tools import RateLimit, ToolRateLimitExceeded, ToolRegistry


class AgentGraphTests(unittest.TestCase):
    def test_graph_exposes_required_roles_and_conditional_paths(self) -> None:
        graph = create_agent_graph(provider=FakeProvider())

        self.assertIn("supervisor", graph.nodes)
        self.assertIn("reviewer", graph.nodes)
        for role in ("research", "analysis", "writing", "code"):
            self.assertIn(f"{role}_specialist", graph.nodes)

        supervisor_routes = graph.conditional_edges["supervisor"]
        reviewer_routes = graph.conditional_edges["reviewer"]
        self.assertEqual(supervisor_routes["delegate_research"], "research_specialist")
        self.assertEqual(supervisor_routes["escalate"], "escalated")
        self.assertEqual(supervisor_routes["complete"], "complete")
        self.assertEqual(reviewer_routes["retry_research"], "research_specialist")
        self.assertEqual(reviewer_routes["reject"], "rejected")
        self.assertEqual(reviewer_routes["escalate"], "escalated")
        self.assertEqual(reviewer_routes["complete"], "complete")

    def test_graph_completes_with_fake_provider(self) -> None:
        graph = create_agent_graph(provider=FakeProvider(confidence=0.96))

        result = graph.run({"task": "Research, analyze, and write a summary"})

        self.assertEqual(result["final_status"], "complete")
        self.assertGreaterEqual(len(result["specialist_outputs"]), 3)
        self.assertEqual(result["review_decisions"][-1]["route"], "complete")

    def test_graph_retries_then_rejects_low_quality_output(self) -> None:
        graph = OrchardAgentGraph(provider=FakeProvider(confidence=0.5), max_retries=1)

        result = graph.run({"task": "Write a summary"})

        self.assertEqual(result["final_status"], "rejected")
        routes = [decision["route"] for decision in result["review_decisions"]]
        self.assertIn("retry_writing", routes)
        self.assertIn("reject", routes)

    def test_graph_escalates_very_low_confidence_output(self) -> None:
        graph = OrchardAgentGraph(provider=FakeProvider(confidence=0.2))

        result = graph.run({"task": "Analyze a risky plan"})

        self.assertEqual(result["final_status"], "escalated")
        self.assertEqual(result["review_decisions"][-1]["route"], "escalate")


class ProviderTests(unittest.TestCase):
    def test_provider_adapters_are_placeholder_safe(self) -> None:
        env = {
            "OPENAI_API_KEY": "placeholder",
            "OPENAI_MODEL": "local-openai-model",
            "ANTHROPIC_API_KEY": "your-anthropic-api-key",
            "ANTHROPIC_MODEL": "local-anthropic-model",
        }

        openai = OpenAIProvider.from_env(env)
        anthropic = AnthropicProvider.from_env(env)

        self.assertFalse(openai.is_configured)
        self.assertFalse(anthropic.is_configured)
        self.assertTrue(openai.config_summary()["placeholder_safe"])
        self.assertTrue(anthropic.config_summary()["placeholder_safe"])
        with self.assertRaises(ProviderConfigurationError):
            openai.generate("hello")
        with self.assertRaises(ProviderConfigurationError):
            anthropic.generate("hello")

    def test_fake_provider_supports_local_tests_without_keys(self) -> None:
        provider = FakeProvider(response_prefix="ok", confidence=0.88)

        response = provider.generate("local prompt")

        self.assertEqual(response.provider, "fake")
        self.assertEqual(response.confidence, 0.88)
        self.assertIn("local prompt", response.content)
        self.assertEqual(len(provider.calls), 1)


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_stores_schema_and_enforces_rate_limit(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="lookup",
            description="Local lookup test tool.",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            rate_limit=RateLimit(max_calls=2, window_seconds=60),
            handler=lambda args: {"result": args["query"].upper()},
        )

        tool = registry.get("lookup")
        self.assertEqual(tool.name, "lookup")
        self.assertEqual(tool.schema["required"], ["query"])
        self.assertEqual(tool.rate_limit.max_calls, 2)

        self.assertEqual(registry.call("lookup", {"query": "a"}, now=100.0), {"result": "A"})
        self.assertEqual(registry.call("lookup", {"query": "b"}, now=120.0), {"result": "B"})
        self.assertFalse(registry.can_call("lookup", now=130.0))
        with self.assertRaises(ToolRateLimitExceeded):
            registry.call("lookup", {"query": "c"}, now=130.0)

        self.assertTrue(registry.can_call("lookup", now=161.0))
        self.assertEqual(registry.remaining_calls("lookup", now=161.0), 1)


if __name__ == "__main__":
    unittest.main()
