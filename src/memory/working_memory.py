"""Working memory: per-case, transient. Discarded or compacted on resolution."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.streaming.event_generator import RawEvent
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WorkingMemory:
    customer_id: str
    case_id: str = field(default_factory=lambda: f"case-{uuid.uuid4().hex[:8]}")
    active_trigger: dict[str, Any] | None = None
    recent_events: list[RawEvent] = field(default_factory=list)
    structured_findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_proposals: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None

    def note_event(self, ev: RawEvent, max_len: int = 50) -> None:
        self.recent_events.append(ev)
        if len(self.recent_events) > max_len:
            self.recent_events = self.recent_events[-max_len:]

    def publish(self, agent: str, findings: dict[str, Any]) -> None:
        self.structured_findings[agent] = findings
        log.debug("working_memory.publish", agent=agent, case_id=self.case_id)

    def add_proposal(self, agent: str, proposal: dict[str, Any]) -> None:
        self.agent_proposals.append({"agent": agent, "proposal": proposal})

    def resolve(self) -> None:
        self.resolved_at = datetime.now(timezone.utc)

    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "case_id": self.case_id,
            "active_trigger": self.active_trigger,
            "recent_event_ids": [e.event_id for e in self.recent_events],
            "structured_findings": self.structured_findings,
            "agent_proposals": self.agent_proposals,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }