"""Human-in-the-loop review queue for OrchardFlow WU-103."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence


APPROVAL_LEVELS = (
    "Notify",
    "Approve Action",
    "Approve Plan",
    "Take Over",
)

ESCALATION_TRIGGERS = (
    "low_confidence",
    "repeated_failure",
    "sensitive_operation",
    "low_quality_score",
)


def _now() -> float:
    return time.time()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass(frozen=True)
class EscalationDecision:
    trigger: str
    approval_level: str
    proposed_action: str
    reasoning: str

    def __post_init__(self) -> None:
        if self.trigger not in ESCALATION_TRIGGERS:
            raise ValueError(f"unknown escalation trigger: {self.trigger}")
        if self.approval_level not in APPROVAL_LEVELS:
            raise ValueError(f"unknown approval level: {self.approval_level}")

    def to_dict(self) -> dict[str, str]:
        return {
            "trigger": self.trigger,
            "approval_level": self.approval_level,
            "proposed_action": self.proposed_action,
            "reasoning": self.reasoning,
        }


@dataclass
class ReviewRequest:
    id: str
    trigger: str
    approval_level: str
    context: dict[str, Any]
    proposed_action: str
    reasoning: str
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    status: str = "queued"
    created_at: float = field(default_factory=_now)
    resolved_at: float | None = None
    decision: str | None = None
    reviewer_notes: str | None = None

    def __post_init__(self) -> None:
        if self.trigger not in ESCALATION_TRIGGERS:
            raise ValueError(f"unknown escalation trigger: {self.trigger}")
        if self.approval_level not in APPROVAL_LEVELS:
            raise ValueError(f"unknown approval level: {self.approval_level}")
        if self.status not in {"queued", "resolved"}:
            raise ValueError(f"unknown review request status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "approval_level": self.approval_level,
            "context": dict(self.context),
            "proposed_action": self.proposed_action,
            "reasoning": self.reasoning,
            "relevant_memories": list(self.relevant_memories),
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "decision": self.decision,
            "reviewer_notes": self.reviewer_notes,
        }


class LocalReviewQueue:
    """In-process review queue with deterministic request ids for local tests."""

    integration_name = "local-review"

    def __init__(self, *, queue_name: str = "human_review") -> None:
        self.queue_name = queue_name
        self._counter = 0
        self._requests: list[ReviewRequest] = []

    def enqueue(
        self,
        *,
        decision: EscalationDecision,
        context: Mapping[str, Any],
        relevant_memories: Sequence[Mapping[str, Any]] | None = None,
        now: float | None = None,
    ) -> ReviewRequest:
        self._counter += 1
        request = ReviewRequest(
            id=f"{self.queue_name}:{self._counter}",
            trigger=decision.trigger,
            approval_level=decision.approval_level,
            context=dict(context),
            proposed_action=decision.proposed_action,
            reasoning=decision.reasoning,
            relevant_memories=[dict(memory) for memory in relevant_memories or []],
            created_at=_now() if now is None else now,
        )
        self._requests.append(request)
        return request

    def pending(self) -> list[ReviewRequest]:
        return [request for request in self._requests if request.status == "queued"]

    def all_requests(self) -> list[ReviewRequest]:
        return list(self._requests)

    def get(self, request_id: str) -> ReviewRequest:
        for request in self._requests:
            if request.id == request_id:
                return request
        raise KeyError(f"review request {request_id!r} was not found")

    def resolve(
        self,
        request_id: str,
        *,
        decision: str,
        reviewer_notes: str | None = None,
        now: float | None = None,
    ) -> ReviewRequest:
        request = self.get(request_id)
        request.status = "resolved"
        request.decision = decision
        request.reviewer_notes = reviewer_notes
        request.resolved_at = _now() if now is None else now
        return request

    def pending_count(self) -> int:
        return len(self.pending())

    def to_dict(self) -> dict[str, Any]:
        return {
            "integration": self.integration_name,
            "queue_name": self.queue_name,
            "pending_count": self.pending_count(),
            "requests": [request.to_dict() for request in self.all_requests()],
        }


@dataclass(frozen=True)
class EscalationPolicy:
    low_confidence_threshold: float = 0.35
    low_quality_threshold: float = 0.35
    repeated_failure_threshold: int = 2
    sensitive_terms: tuple[str, ...] = (
        "delete",
        "deploy",
        "publish",
        "credential",
        "payment",
    )

    def evaluate_supervisor(
        self,
        state: Mapping[str, Any],
        *,
        task: str,
    ) -> EscalationDecision | None:
        failure_count = int(
            state.get("failure_count")
            or state.get("consecutive_failures")
            or state.get("repeated_failures")
            or 0
        )
        if failure_count >= self.repeated_failure_threshold:
            return EscalationDecision(
                trigger="repeated_failure",
                approval_level="Take Over",
                proposed_action="Pause the run and transfer control to a human reviewer.",
                reasoning=(
                    f"Execution has recorded {failure_count} consecutive failures, "
                    "meeting the repeated-failure escalation trigger."
                ),
            )

        if state.get("requires_approval") or state.get("force_escalate"):
            return EscalationDecision(
                trigger="sensitive_operation",
                approval_level="Approve Plan",
                proposed_action="Review and approve the plan before further delegation.",
                reasoning="The run explicitly requires human approval before continuing.",
            )

        task_lower = task.lower()
        matched_terms = [term for term in self.sensitive_terms if term in task_lower]
        if matched_terms:
            return EscalationDecision(
                trigger="sensitive_operation",
                approval_level="Approve Action",
                proposed_action="Approve the sensitive action before execution continues.",
                reasoning=(
                    "The task references sensitive operation terms: "
                    f"{', '.join(matched_terms)}."
                ),
            )
        return None

    def evaluate_reviewer(
        self,
        state: Mapping[str, Any],
        *,
        output: Mapping[str, Any],
        quality_score: float,
    ) -> EscalationDecision | None:
        if output.get("requires_human"):
            return EscalationDecision(
                trigger="low_confidence",
                approval_level="Approve Action",
                proposed_action="Review provider failure context and choose the next action.",
                reasoning="The specialist output requires human handling before continuing.",
            )

        confidence = float(output.get("confidence", 0.0))
        if confidence < self.low_confidence_threshold:
            return EscalationDecision(
                trigger="low_confidence",
                approval_level="Approve Action",
                proposed_action="Review the low-confidence output before routing continues.",
                reasoning=(
                    f"Specialist confidence {confidence:.2f} is below the "
                    f"{self.low_confidence_threshold:.2f} threshold."
                ),
            )

        if quality_score < self.low_quality_threshold:
            return EscalationDecision(
                trigger="low_quality_score",
                approval_level="Approve Plan",
                proposed_action="Review the low-quality output and approve a revised plan.",
                reasoning=(
                    f"Reviewer quality score {quality_score:.2f} is below the "
                    f"{self.low_quality_threshold:.2f} threshold."
                ),
            )

        failure_count = int(output.get("failure_count") or state.get("failure_count") or 0)
        if failure_count >= self.repeated_failure_threshold:
            return EscalationDecision(
                trigger="repeated_failure",
                approval_level="Take Over",
                proposed_action="Pause the run and transfer control to a human reviewer.",
                reasoning=(
                    f"Specialist output reports {failure_count} failures, meeting the "
                    "repeated-failure escalation trigger."
                ),
            )

        return None


def build_review_context(state: Mapping[str, Any]) -> dict[str, Any]:
    """Package the execution state fields a human needs for review."""

    return {
        "task": state.get("task", ""),
        "status": state.get("status"),
        "current_step": dict(state.get("current_step", {}) or {}),
        "current_output": dict(state.get("current_output", {}) or {}),
        "plan": _as_list(state.get("plan")),
        "retry_counts": dict(state.get("retry_counts", {}) or {}),
        "messages": _as_list(state.get("messages")),
        "specialist_outputs": _as_list(state.get("specialist_outputs")),
        "review_decisions": _as_list(state.get("review_decisions")),
    }


__all__ = [
    "APPROVAL_LEVELS",
    "ESCALATION_TRIGGERS",
    "EscalationDecision",
    "EscalationPolicy",
    "LocalReviewQueue",
    "ReviewRequest",
    "build_review_context",
]
