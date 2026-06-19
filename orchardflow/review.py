"""Compatibility review interfaces for WU-101.

The full human-in-the-loop review queue lands in WU-103. This module preserves
the escalation contract needed by the agent graph in the first stacked PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class EscalationDecision:
    trigger: str
    approval_level: str
    proposed_action: str
    reasoning: str


@dataclass
class ReviewRequest:
    decision: EscalationDecision
    context: Mapping[str, Any]
    relevant_memories: list[Any] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"review-{uuid4().hex}")
    status: str = "queued"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "trigger": self.decision.trigger,
            "approval_level": self.decision.approval_level,
            "proposed_action": self.decision.proposed_action,
            "reasoning": self.decision.reasoning,
            "context": dict(self.context),
            "relevant_memories": list(self.relevant_memories),
        }


class LocalReviewQueue:
    def __init__(self) -> None:
        self._requests: list[ReviewRequest] = []

    def enqueue(
        self,
        *,
        decision: EscalationDecision,
        context: Mapping[str, Any],
        relevant_memories: list[Any],
    ) -> ReviewRequest:
        request = ReviewRequest(
            decision=decision,
            context=context,
            relevant_memories=list(relevant_memories),
        )
        self._requests.append(request)
        return request

    def pending_count(self) -> int:
        return sum(1 for request in self._requests if request.status == "queued")

    def to_dict(self) -> list[dict[str, Any]]:
        return [request.to_dict() for request in self._requests]


class EscalationPolicy:
    def evaluate_supervisor(
        self,
        state: Mapping[str, Any],
        *,
        task: str,
    ) -> EscalationDecision | None:
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
                trigger="provider_unconfigured",
                approval_level="Approve Action",
                proposed_action="Review the provider error and decide whether to continue.",
                reasoning="The specialist could not run without configured provider credentials.",
            )
        return None


def build_review_context(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": state.get("task"),
        "status": state.get("status"),
        "current_step": state.get("current_step"),
        "current_output": state.get("current_output"),
    }
