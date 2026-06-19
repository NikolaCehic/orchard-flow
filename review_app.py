"""Streamlit Review UI entry point for OrchardFlow WU-103.

Tests import the pure helpers in this module; running a live Streamlit server is
not required for local verification.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from orchardflow.review import APPROVAL_LEVELS, ReviewRequest


APPROVAL_LEVEL_LABELS = list(APPROVAL_LEVELS)
REVIEW_UI_FIELDS = (
    "context",
    "proposed_action",
    "reasoning",
    "relevant_memories",
    "approval_levels",
)


def format_review_request_for_display(
    request: Mapping[str, Any] | ReviewRequest,
) -> dict[str, Any]:
    payload = request.to_dict() if isinstance(request, ReviewRequest) else dict(request)
    return {
        "id": payload.get("id", ""),
        "status": payload.get("status", "queued"),
        "trigger": payload.get("trigger", ""),
        "selected_approval_level": payload.get(
            "approval_level",
            payload.get("selected_approval_level", ""),
        ),
        "approval_levels": APPROVAL_LEVEL_LABELS,
        "context": dict(payload.get("context", {}) or {}),
        "proposed_action": payload.get("proposed_action", ""),
        "reasoning": payload.get("reasoning", ""),
        "relevant_memories": list(payload.get("relevant_memories", []) or []),
    }


def _sample_request() -> dict[str, Any]:
    return {
        "id": "sample",
        "status": "queued",
        "trigger": "low_confidence",
        "approval_level": "Approve Action",
        "context": {
            "task": "Review a paused OrchardFlow run.",
            "status": "escalated",
            "current_step": {},
            "current_output": {},
        },
        "proposed_action": "Review the low-confidence output before routing continues.",
        "reasoning": "Sample request for local UI rendering without live services.",
        "relevant_memories": [],
    }


def render_review_request(st: Any, request: Mapping[str, Any] | ReviewRequest) -> None:
    display = format_review_request_for_display(request)
    st.header("Human Review Queue")
    st.caption(f"Request {display['id']} | {display['status']} | {display['trigger']}")

    st.subheader("Context")
    st.json(display["context"])

    st.subheader("Proposed Action")
    st.write(display["proposed_action"])

    st.subheader("Reasoning")
    st.write(display["reasoning"])

    st.subheader("Relevant Memories")
    if display["relevant_memories"]:
        for memory in display["relevant_memories"]:
            st.json(memory)
    else:
        st.write("No relevant memories were packaged with this review request.")

    st.subheader("Approval Levels")
    selected_index = 0
    if display["selected_approval_level"] in display["approval_levels"]:
        selected_index = display["approval_levels"].index(display["selected_approval_level"])
    st.radio(
        "Select approval level",
        display["approval_levels"],
        index=selected_index,
    )


def run(review_requests: Sequence[Mapping[str, Any] | ReviewRequest] | None = None) -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is optional. Install OrchardFlow with the review extra to run "
            "the review UI."
        ) from exc

    st.set_page_config(page_title="OrchardFlow Review", layout="wide")
    requests = list(review_requests or [_sample_request()])
    formatted = [format_review_request_for_display(request) for request in requests]
    pending = [request for request in formatted if request["status"] == "queued"]
    selected = pending[0] if pending else formatted[0]
    render_review_request(st, selected)


if __name__ == "__main__":
    run()
