"""
Trace ID management and span logging.

Every decision the system makes must be reconstructable. We use a
per-event `trace_id` that follows the event through every agent, tool
call, retrieval, and handoff. Spans are emitted as structured logs
that can be replayed into a timeline.
"""
from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.utils.logger import get_logger

log = get_logger(__name__)

# Context variable so the current trace_id flows implicitly through async calls.
_current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default=""
)


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:16]}"


def get_trace_id() -> str:
    tid = _current_trace_id.get()
    if not tid:
        tid = new_trace_id()
        _current_trace_id.set(tid)
    return tid


def set_trace_id(trace_id: str) -> None:
    _current_trace_id.set(trace_id)


@dataclass
class Span:
    """A single unit of work inside a trace."""
    name: str
    trace_id: str
    span_id: str
    start_ts: float
    end_ts: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_span_id: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_ts is None:
            return None
        return (self.end_ts - self.start_ts) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """
    Context manager that emits a structured span log on exit.

    Usage:
        with span("usage_agent.run", customer_id="C1") as s:
            ...
            s.attributes["confidence"] = 0.91
    """
    s = Span(
        name=name,
        trace_id=get_trace_id(),
        span_id=f"span-{uuid.uuid4().hex[:12]}",
        start_ts=time.time(),
        attributes=dict(attributes),
    )
    log.info("span.start", **s.to_dict())
    try:
        yield s
    except Exception as e:
        s.attributes["error"] = str(e)
        s.end_ts = time.time()
        log.error("span.error", **s.to_dict())
        raise
    finally:
        if s.end_ts is None:
            s.end_ts = time.time()
        log.info("span.end", **s.to_dict())