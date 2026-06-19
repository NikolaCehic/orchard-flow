"""LangGraph-compatible agent architecture for OrchardFlow WU-101."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, MutableMapping

from orchardflow.memory import LongTermMemoryStore, MemoryQueryResult
from orchardflow.observability import (
    LocalTraceRecorder,
    TraceRun,
    estimate_cost_usd,
    json_safe,
)
from orchardflow.providers import (
    FakeProvider,
    ProviderClient,
    ProviderConfigurationError,
    ProviderRuntimeError,
)
from orchardflow.review import (
    EscalationDecision,
    EscalationPolicy,
    LocalReviewQueue,
    build_review_context,
)
from orchardflow.tools import ToolRegistry, build_default_tool_registry


AgentState = MutableMapping[str, Any]
NodeCallable = Callable[[AgentState], AgentState]

SPECIALIST_ROLES = ("research", "analysis", "writing", "code")
TERMINAL_NODES = ("complete", "rejected", "escalated")


class LangGraphUnavailable(RuntimeError):
    """Raised when optional LangGraph compilation is requested without LangGraph."""


class GraphExecutionError(RuntimeError):
    """Raised when the local graph runner cannot resolve a next node."""


@dataclass(frozen=True)
class SpecialistDefinition:
    role: str
    node_name: str
    instruction: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphDefinition:
    nodes: Mapping[str, NodeCallable]
    static_edges: Mapping[str, str]
    conditional_edges: Mapping[str, Mapping[str, str]]
    terminal_nodes: tuple[str, ...] = TERMINAL_NODES


@dataclass
class OrchardAgentGraph:
    """Supervisor, specialist, and reviewer graph with conditional routing."""

    provider: ProviderClient = field(default_factory=FakeProvider)
    tool_registry: ToolRegistry = field(default_factory=build_default_tool_registry)
    long_term_memory: LongTermMemoryStore | None = None
    planning_memory_limit: int = 5
    max_retries: int = 2
    retry_quality_threshold: float = 0.72
    escalation_quality_threshold: float = 0.35
    review_queue: LocalReviewQueue = field(default_factory=LocalReviewQueue)
    escalation_policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    trace_recorder: LocalTraceRecorder = field(default_factory=LocalTraceRecorder)

    def __post_init__(self) -> None:
        self.specialists = {
            "research": SpecialistDefinition(
                role="research",
                node_name="research_specialist",
                instruction="Gather source-grounded facts and constraints.",
                tools=("echo",),
            ),
            "analysis": SpecialistDefinition(
                role="analysis",
                node_name="analysis_specialist",
                instruction="Compare evidence, risks, and tradeoffs.",
                tools=("echo",),
            ),
            "writing": SpecialistDefinition(
                role="writing",
                node_name="writing_specialist",
                instruction="Prepare concise user-facing output.",
                tools=("echo",),
            ),
            "code": SpecialistDefinition(
                role="code",
                node_name="code_specialist",
                instruction="Design implementation steps and code-facing changes.",
                tools=("echo",),
            ),
        }
        self.nodes: dict[str, NodeCallable] = {
            "supervisor": self.supervisor_node,
            "research_specialist": self._make_specialist_node("research"),
            "analysis_specialist": self._make_specialist_node("analysis"),
            "writing_specialist": self._make_specialist_node("writing"),
            "code_specialist": self._make_specialist_node("code"),
            "reviewer": self.reviewer_node,
            "complete": self.complete_node,
            "rejected": self.rejected_node,
            "escalated": self.escalated_node,
        }
        self.static_edges = {
            "research_specialist": "reviewer",
            "analysis_specialist": "reviewer",
            "writing_specialist": "reviewer",
            "code_specialist": "reviewer",
        }
        self.conditional_edges = {
            "supervisor": {
                "delegate_research": "research_specialist",
                "delegate_analysis": "analysis_specialist",
                "delegate_writing": "writing_specialist",
                "delegate_code": "code_specialist",
                "complete": "complete",
                "escalate": "escalated",
            },
            "reviewer": {
                "approve": "supervisor",
                "complete": "complete",
                "retry_research": "research_specialist",
                "retry_analysis": "analysis_specialist",
                "retry_writing": "writing_specialist",
                "retry_code": "code_specialist",
                "reject": "rejected",
                "escalate": "escalated",
            },
        }
        self.definition = GraphDefinition(
            nodes=self.nodes,
            static_edges=self.static_edges,
            conditional_edges=self.conditional_edges,
        )

    def supervisor_node(self, state: AgentState) -> AgentState:
        next_state = self._copy_state(state)
        task = str(next_state.get("task", "")).strip()
        next_state.setdefault("messages", [])
        next_state.setdefault("specialist_outputs", [])
        next_state.setdefault("review_decisions", [])
        next_state.setdefault("retry_counts", {})

        supervisor_escalation = self.escalation_policy.evaluate_supervisor(
            next_state,
            task=task,
        )
        if supervisor_escalation is not None:
            next_state["status"] = "escalated"
            self._pause_for_review(next_state, supervisor_escalation)
            next_state["next_route"] = "escalate"
            return next_state

        if not next_state.get("plan"):
            planning_memories = self.query_planning_memory(
                task,
                user_id=next_state.get("user_id"),
                now=next_state.get("memory_now"),
            )
            next_state["relevant_memories"] = [
                memory.to_dict() for memory in planning_memories
            ]
            if planning_memories:
                next_state["messages"].append(
                    {
                        "role": "supervisor",
                        "content": (
                            f"Retrieved {len(planning_memories)} relevant memories "
                            "for planning."
                        ),
                        "memory_ids": [
                            memory.record.id for memory in planning_memories
                        ],
                    }
                )
            next_state["plan"] = self.decompose_task(task)
            next_state["cursor"] = 0

        cursor = int(next_state.get("cursor", 0))
        plan = list(next_state.get("plan", []))
        if cursor >= len(plan):
            next_state["status"] = "complete"
            next_state["next_route"] = "complete"
            return next_state

        step = plan[cursor]
        role = step["role"]
        next_state["current_step"] = step
        next_state["last_specialist_role"] = role
        next_state["status"] = "delegating"
        next_state["next_route"] = f"delegate_{role}"
        next_state["messages"].append(
            {
                "role": "supervisor",
                "content": f"Delegating step {step['id']} to {role}.",
            }
        )
        return next_state

    def decompose_task(self, task: str) -> list[Mapping[str, Any]]:
        lower = task.lower()
        roles: list[str] = []
        if any(word in lower for word in ("research", "find", "source", "facts")):
            roles.append("research")
        if any(word in lower for word in ("analyze", "compare", "risk", "tradeoff")):
            roles.append("analysis")
        if any(word in lower for word in ("write", "draft", "summarize", "summary")):
            roles.append("writing")
        if any(word in lower for word in ("code", "implement", "test", "build")):
            roles.append("code")
        if not roles:
            roles = ["research", "analysis", "writing"]

        deduped = list(dict.fromkeys(roles))
        return [
            {
                "id": f"step-{index}",
                "role": role,
                "objective": f"{role} work for: {task or 'unspecified task'}",
            }
            for index, role in enumerate(deduped, start=1)
        ]

    def query_planning_memory(
        self,
        task: str,
        *,
        user_id: str | None = None,
        now: float | None = None,
    ) -> list[MemoryQueryResult]:
        if self.long_term_memory is None or not task.strip():
            return []
        return self.long_term_memory.query(
            task,
            user_id=user_id,
            limit=self.planning_memory_limit,
            now=now,
        )

    def reviewer_node(self, state: AgentState) -> AgentState:
        next_state = self._copy_state(state)
        output = dict(next_state.get("current_output", {}))
        role = str(output.get("role") or next_state.get("last_specialist_role") or "")
        confidence = float(output.get("confidence", 0.0))
        forced_quality = next_state.get("force_quality_score")
        quality_score = float(forced_quality if forced_quality is not None else confidence)
        retry_counts = dict(next_state.get("retry_counts", {}))
        retry_count = int(retry_counts.get(role, 0))

        decision = {
            "role": role,
            "quality_score": quality_score,
            "retry_count": retry_count,
            "reason": "",
        }

        escalation: EscalationDecision | None = None
        if not next_state.get("force_reject"):
            escalation = self.escalation_policy.evaluate_reviewer(
                next_state,
                output=output,
                quality_score=quality_score,
            )
            if escalation is None and quality_score < self.escalation_quality_threshold:
                escalation = EscalationDecision(
                    trigger="low_quality_score",
                    approval_level="Approve Plan",
                    proposed_action="Review the low-quality output and approve a revised plan.",
                    reasoning=(
                        f"Reviewer quality score {quality_score:.2f} is below the "
                        f"{self.escalation_quality_threshold:.2f} threshold."
                    ),
                )

        if next_state.get("force_reject"):
            route = "reject"
            status = "rejected"
            decision["reason"] = "Reviewer rejected output by explicit state flag."
        elif escalation is not None:
            route = "escalate"
            status = "escalated"
            decision["reason"] = escalation.reasoning
            self._pause_for_review(next_state, escalation)
        elif quality_score < self.retry_quality_threshold and retry_count < self.max_retries:
            retry_counts[role] = retry_count + 1
            route = f"retry_{role}"
            status = "retrying"
            decision["reason"] = "Reviewer requested specialist retry below quality threshold."
        elif quality_score < self.retry_quality_threshold:
            route = "reject"
            status = "rejected"
            decision["reason"] = "Reviewer rejected output after retry budget was exhausted."
        else:
            cursor = int(next_state.get("cursor", 0)) + 1
            next_state["cursor"] = cursor
            plan = list(next_state.get("plan", []))
            if cursor >= len(plan):
                route = "complete"
                status = "complete"
                decision["reason"] = "Reviewer approved the final planned step."
            else:
                route = "approve"
                status = "approved"
                decision["reason"] = "Reviewer approved output and returned to supervisor."

        decision["route"] = route
        next_state["retry_counts"] = retry_counts
        next_state.setdefault("review_decisions", []).append(decision)
        next_state["status"] = status
        next_state["next_route"] = route
        next_state["messages"].append({"role": "reviewer", "content": decision["reason"]})
        return next_state

    def complete_node(self, state: AgentState) -> AgentState:
        next_state = self._copy_state(state)
        next_state["status"] = "complete"
        next_state["final_status"] = "complete"
        return next_state

    def rejected_node(self, state: AgentState) -> AgentState:
        next_state = self._copy_state(state)
        next_state["status"] = "rejected"
        next_state["final_status"] = "rejected"
        return next_state

    def escalated_node(self, state: AgentState) -> AgentState:
        next_state = self._copy_state(state)
        next_state["status"] = "escalated"
        next_state["final_status"] = "escalated"
        return next_state

    def route_supervisor(self, state: AgentState) -> str:
        route = str(state.get("next_route", ""))
        if route not in self.conditional_edges["supervisor"]:
            raise GraphExecutionError(f"Unknown supervisor route: {route!r}")
        return route

    def route_reviewer(self, state: AgentState) -> str:
        route = str(state.get("next_route", ""))
        if route not in self.conditional_edges["reviewer"]:
            raise GraphExecutionError(f"Unknown reviewer route: {route!r}")
        return route

    def next_node(self, current_node: str, state: AgentState) -> str:
        if current_node in self.static_edges:
            return self.static_edges[current_node]
        if current_node == "supervisor":
            return self.conditional_edges["supervisor"][self.route_supervisor(state)]
        if current_node == "reviewer":
            return self.conditional_edges["reviewer"][self.route_reviewer(state)]
        raise GraphExecutionError(f"No outgoing route for node {current_node!r}")

    def run(self, initial_state: Mapping[str, Any], *, max_steps: int = 50) -> AgentState:
        state: AgentState = dict(initial_state)
        current = "supervisor"
        trace_run = self.trace_recorder.start_run(
            name="orchardflow.graph.run",
            input_state=state,
        )
        state["trace_id"] = trace_run.trace_id

        for step_index in range(max_steps):
            before_state = self._copy_state(state)
            start_ns = time.time_ns()
            try:
                state = self.nodes[current](state)
            except Exception as exc:
                end_ns = time.time_ns()
                trace_run.record(
                    name=f"orchardflow.node.{current}",
                    event_type="node",
                    node=current,
                    step_index=step_index,
                    start_time_unix_nano=start_ns,
                    end_time_unix_nano=end_ns,
                    error=str(exc),
                    attributes={
                        "exception.type": type(exc).__name__,
                        "orchardflow.status": before_state.get("status"),
                    },
                )
                self._publish_trace_state(state, trace_run)
                raise

            end_ns = time.time_ns()
            trace_run.record(
                name=f"orchardflow.node.{current}",
                event_type="node",
                node=current,
                step_index=step_index,
                start_time_unix_nano=start_ns,
                end_time_unix_nano=end_ns,
                cost_usd=self._node_cost_usd(current, before_state, state),
                attributes={
                    "orchardflow.status": state.get("status"),
                    "orchardflow.route": state.get("next_route"),
                    "orchardflow.final_status": state.get("final_status"),
                },
            )
            self._record_trace_events(trace_run, current, before_state, state, step_index)
            self._publish_trace_state(state, trace_run)
            if current in TERMINAL_NODES:
                trace_run.finish(state)
                self._publish_trace_state(state, trace_run, include_explorer=True)
                return state
            try:
                current = self.next_node(current, state)
            except Exception as exc:
                trace_run.record(
                    name="orchardflow.route.error",
                    event_type="error",
                    node=current,
                    step_index=step_index,
                    error=str(exc),
                    attributes={"exception.type": type(exc).__name__},
                )
                self._publish_trace_state(state, trace_run, include_explorer=True)
                raise

        raise GraphExecutionError(f"Graph exceeded {max_steps} steps")

    def build_langgraph(self) -> Any:
        """Compile the graph with LangGraph when the optional package is installed."""

        try:
            from langgraph.graph import END, StateGraph
        except ModuleNotFoundError as exc:
            raise LangGraphUnavailable(
                "LangGraph is optional for local tests; install the langgraph extra "
                "to compile this definition."
            ) from exc

        builder = StateGraph(dict)
        for name, node in self.nodes.items():
            builder.add_node(name, node)
        builder.set_entry_point("supervisor")
        for source, target in self.static_edges.items():
            builder.add_edge(source, target)
        builder.add_conditional_edges(
            "supervisor",
            self.route_supervisor,
            dict(self.conditional_edges["supervisor"]),
        )
        builder.add_conditional_edges(
            "reviewer",
            self.route_reviewer,
            dict(self.conditional_edges["reviewer"]),
        )
        for terminal in TERMINAL_NODES:
            builder.add_edge(terminal, END)
        return builder.compile()

    def _make_specialist_node(self, role: str) -> NodeCallable:
        definition = self.specialists[role]

        def specialist_node(state: AgentState) -> AgentState:
            next_state = self._copy_state(state)
            step = dict(next_state.get("current_step", {}))
            prompt = (
                f"{definition.instruction}\n"
                f"Objective: {step.get('objective', next_state.get('task', ''))}"
            )
            try:
                response = self.provider.generate(
                    prompt,
                    system=f"You are OrchardFlow's {role} specialist.",
                    temperature=0.0,
                )
                content = response.content
                provider_name = response.provider
                confidence = response.confidence
                usage = dict(response.usage)
                cost_usd = estimate_cost_usd(usage, provider=provider_name)
                requires_human = False
            except (ProviderConfigurationError, ProviderRuntimeError) as exc:
                content = str(exc)
                provider_name = getattr(self.provider, "provider_name", "unknown")
                confidence = 0.0
                usage = {}
                cost_usd = 0.0
                requires_human = True

            if next_state.get("force_confidence") is not None:
                confidence = float(next_state["force_confidence"])

            tool_calls = self._call_specialist_tools(definition, content)
            output = {
                "role": role,
                "node": definition.node_name,
                "step_id": step.get("id"),
                "content": content,
                "provider": provider_name,
                "confidence": confidence,
                "usage": usage,
                "cost_usd": cost_usd,
                "tool_calls": tool_calls,
                "requires_human": requires_human,
            }
            next_state.setdefault("specialist_outputs", []).append(output)
            next_state["current_output"] = output
            next_state["last_specialist_role"] = role
            next_state["status"] = "reviewing"
            next_state["messages"].append(
                {
                    "role": definition.node_name,
                    "content": content,
                    "confidence": confidence,
                }
            )
            return next_state

        specialist_node.__name__ = f"{role}_specialist_node"
        return specialist_node

    def _call_specialist_tools(
        self,
        definition: SpecialistDefinition,
        content: str,
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for tool_name in definition.tools:
            arguments = {"text": content}
            start_ns = time.time_ns()
            error: str | None = None
            result: Any = None
            try:
                result = self.tool_registry.call(tool_name, arguments)
            except Exception as exc:  # Tool failures are traced without hiding node output.
                error = str(exc)
            end_ns = time.time_ns()
            tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": json_safe(arguments),
                    "result": json_safe(result),
                    "latency_ms": round(max((end_ns - start_ns) / 1_000_000, 0.0), 6),
                    "cost_usd": 0.0,
                    "error": error,
                    "success": error is None,
                }
            )
        return tool_calls

    def _record_trace_events(
        self,
        trace_run: TraceRun,
        current_node: str,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        step_index: int,
    ) -> None:
        if current_node == "supervisor":
            self._record_supervisor_trace_events(
                trace_run,
                before_state,
                after_state,
                step_index,
            )

        if current_node.endswith("_specialist"):
            for output in self._new_mapping_items(
                before_state,
                after_state,
                "specialist_outputs",
            ):
                for tool_call in output.get("tool_calls", []):
                    trace_run.record(
                        name=f"orchardflow.tool.{tool_call.get('name')}",
                        event_type="tool_call",
                        node=current_node,
                        step_index=step_index,
                        latency_ms=float(tool_call.get("latency_ms", 0.0)),
                        cost_usd=float(tool_call.get("cost_usd", 0.0)),
                        error=tool_call.get("error"),
                        attributes={
                            "tool.name": tool_call.get("name"),
                            "tool.success": tool_call.get("success"),
                            "tool.arguments": tool_call.get("arguments"),
                            "tool.result": tool_call.get("result"),
                        },
                    )

        review_request = after_state.get("review_request")
        if review_request and review_request != before_state.get("review_request"):
            request = dict(review_request)
            trace_run.record(
                name="orchardflow.escalation",
                event_type="escalation_event",
                node=current_node,
                step_index=step_index,
                attributes={
                    "escalation.trigger": request.get("trigger"),
                    "escalation.approval_level": request.get("approval_level"),
                    "escalation.proposed_action": request.get("proposed_action"),
                    "escalation.reasoning": request.get("reasoning"),
                    "review.request_id": request.get("id"),
                },
            )

    def _record_supervisor_trace_events(
        self,
        trace_run: TraceRun,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        step_index: int,
    ) -> None:
        plan = list(after_state.get("plan", []))
        current_step = dict(after_state.get("current_step", {}))
        route = after_state.get("next_route")
        planning_attributes = {
            "planning.route": route,
            "planning.plan_size": len(plan),
            "planning.roles": [
                step.get("role") for step in plan if isinstance(step, Mapping)
            ],
            "planning.current_step_id": current_step.get("id"),
            "planning.current_role": current_step.get("role"),
        }
        trace_run.record(
            name="orchardflow.planning.decision",
            event_type="planning_decision",
            node="supervisor",
            step_index=step_index,
            attributes=planning_attributes,
        )

        if not before_state.get("plan") and after_state.get("plan") is not None:
            memories = list(after_state.get("relevant_memories", []))
            memory_ids = []
            for memory in memories:
                if isinstance(memory, Mapping):
                    record = memory.get("record")
                    if isinstance(record, Mapping):
                        memory_ids.append(record.get("id"))
            trace_run.record(
                name="orchardflow.memory.retrieval",
                event_type="memory_retrieval",
                node="supervisor",
                step_index=step_index,
                attributes={
                    "memory.query": after_state.get("task"),
                    "memory.retrieved_count": len(memories),
                    "memory.ids": memory_ids,
                },
            )

    def _new_mapping_items(
        self,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        key: str,
    ) -> list[Mapping[str, Any]]:
        before_count = len(list(before_state.get(key, [])))
        after_items = list(after_state.get(key, []))
        return [
            item
            for item in after_items[before_count:]
            if isinstance(item, Mapping)
        ]

    def _node_cost_usd(
        self,
        current_node: str,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
    ) -> float:
        if not current_node.endswith("_specialist"):
            return 0.0
        cost = 0.0
        for output in self._new_mapping_items(
            before_state,
            after_state,
            "specialist_outputs",
        ):
            cost += float(output.get("cost_usd", 0.0))
            for tool_call in output.get("tool_calls", []):
                if isinstance(tool_call, Mapping):
                    cost += float(tool_call.get("cost_usd", 0.0))
        return round(cost, 9)

    def _publish_trace_state(
        self,
        state: AgentState,
        trace_run: TraceRun,
        *,
        include_explorer: bool = False,
    ) -> None:
        records = trace_run.records_as_dicts()
        state["trace_id"] = trace_run.trace_id
        state["trace_records"] = records
        state["trace"] = records
        if include_explorer:
            state["trace_explorer"] = trace_run.explorer_export(state)

    def _pause_for_review(
        self,
        state: AgentState,
        escalation: EscalationDecision,
    ) -> None:
        request = self.review_queue.enqueue(
            decision=escalation,
            context=build_review_context(state),
            relevant_memories=state.get("relevant_memories", []),
        )
        request_payload = request.to_dict()
        state["execution_paused"] = True
        state["escalation_reason"] = escalation.reasoning
        state["escalation_trigger"] = escalation.trigger
        state["approval_level"] = escalation.approval_level
        state["review_queue_status"] = request.status
        state["review_request"] = request_payload
        state.setdefault("review_requests", []).append(request_payload)
        state["review_queue"] = self.review_queue.to_dict()

    def _copy_state(self, state: Mapping[str, Any]) -> AgentState:
        copied: AgentState = dict(state)
        for key in (
            "messages",
            "specialist_outputs",
            "review_decisions",
            "trace",
            "review_requests",
        ):
            if key in copied:
                copied[key] = list(copied[key])
        if "retry_counts" in copied:
            copied["retry_counts"] = dict(copied["retry_counts"])
        return copied


def create_agent_graph(
    *,
    provider: ProviderClient | None = None,
    tool_registry: ToolRegistry | None = None,
    long_term_memory: LongTermMemoryStore | None = None,
    planning_memory_limit: int = 5,
    max_retries: int = 2,
    review_queue: LocalReviewQueue | None = None,
    escalation_policy: EscalationPolicy | None = None,
    trace_recorder: LocalTraceRecorder | None = None,
) -> OrchardAgentGraph:
    return OrchardAgentGraph(
        provider=provider or FakeProvider(),
        tool_registry=tool_registry or build_default_tool_registry(),
        long_term_memory=long_term_memory,
        planning_memory_limit=planning_memory_limit,
        max_retries=max_retries,
        review_queue=review_queue or LocalReviewQueue(),
        escalation_policy=escalation_policy or EscalationPolicy(),
        trace_recorder=trace_recorder or LocalTraceRecorder(),
    )


__all__ = [
    "AgentState",
    "GraphDefinition",
    "GraphExecutionError",
    "LangGraphUnavailable",
    "OrchardAgentGraph",
    "SPECIALIST_ROLES",
    "SpecialistDefinition",
    "TERMINAL_NODES",
    "create_agent_graph",
]
