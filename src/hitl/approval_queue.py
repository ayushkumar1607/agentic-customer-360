"""
Real HITL approval queue.

The pipeline submits a request and awaits a decision. A Streamlit UI (or an
auto-approver in test mode) resolves it. This is a genuine interruption in
the flow, not a log line claiming "human approved".
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.utils.logger import get_logger

log = get_logger(__name__)

Decision = Literal["approve", "reject", "modify"]


@dataclass
class HITLRequest:
    request_id: str
    customer_id: str
    action: str
    action_subtype: str | None
    cost_estimate_usd: float
    message_draft: str | None
    reasoning: str
    confidence: float
    evidence: list[str]
    synthesis_snapshot: dict[str, Any]
    critique_snapshot: dict[str, Any]
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decided_at: str | None = None
    decision: Decision | None = None
    reviewer_notes: str = ""
    modified_action: dict[str, Any] | None = None


class HITLQueue:
    """
    In-memory async approval queue.

    submit() returns a request_id.
    await wait_for_decision(request_id) blocks until decide() is called.
    """

    def __init__(self, auto_approve: bool = False, auto_decision: Decision = "approve"):
        self._requests: dict[str, HITLRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self.auto_approve = auto_approve
        self.auto_decision = auto_decision

    def submit(self, req: HITLRequest) -> str:
        self._requests[req.request_id] = req
        self._events[req.request_id] = asyncio.Event()
        log.warning(
            "hitl.submitted",
            request_id=req.request_id,
            customer_id=req.customer_id,
            action=req.action,
            cost=req.cost_estimate_usd,
            confidence=req.confidence,
        )
        if self.auto_approve:
            self.decide(req.request_id, self.auto_decision, "auto-approved (test mode)")
        return req.request_id

    def decide(self, request_id: str, decision: Decision,
               reviewer_notes: str = "",
               modified_action: dict[str, Any] | None = None) -> None:
        req = self._requests.get(request_id)
        if req is None:
            log.error("hitl.unknown_request", request_id=request_id)
            return
        req.decision = decision
        req.decided_at = datetime.now(timezone.utc).isoformat()
        req.reviewer_notes = reviewer_notes
        req.modified_action = modified_action
        self._events[request_id].set()
        log.warning("hitl.decided", request_id=request_id, decision=decision)

    async def wait_for_decision(self, request_id: str, timeout_s: float = 300.0
                                ) -> HITLRequest:
        await asyncio.wait_for(self._events[request_id].wait(), timeout=timeout_s)
        return self._requests[request_id]

    def snapshot(self) -> list[HITLRequest]:
        return list(self._requests.values())

    def new_request_id(self) -> str:
        return f"hitl-{uuid.uuid4().hex[:10]}"