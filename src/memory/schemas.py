"""Typed memory entries."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


MemoryKind = Literal["event", "risk_flag", "intervention", "life_event",
                     "customer_response", "outcome", "note"]


class MemoryEntry(BaseModel):
    memory_id: str
    customer_id: str
    kind: MemoryKind
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    t_event: datetime
    w_base: float = 1.0
    r_reinforce: float = 0.0
    source_event_id: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["t_event"] = self.t_event.isoformat()
        return d


class StateBoardSlot(BaseModel):
    """One agent's published findings on one customer."""
    agent: str
    customer_id: str
    findings: dict[str, Any]
    confidence: float | None = None
    t_published: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None