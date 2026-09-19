"""
Episodic memory: per-customer long-term history.

Seeded from `history_seed.jsonl` and updated as new events are observed.
Retrieval is decay-weighted — old entries are preserved but lose influence.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from src.memory.decay import activation
from src.memory.schemas import MemoryEntry
from src.streaming.event_generator import RawEvent
from src.utils.logger import get_logger
from src.utils.tracing import span

log = get_logger(__name__)


def _summarize_event(ev: RawEvent) -> str:
    """Compact human-readable summary — used as the memory's `summary`."""
    p = ev.payload or {}
    et = ev.event_type
    if et == "purchase":
        return f"purchase at {p.get('merchant_name','?')} (${p.get('amount','?')})"
    if et == "deposit":
        return f"deposit (${p.get('amount','?')})"
    if et in ("withdrawal", "outbound_transfer", "inbound_transfer"):
        return f"{et} (${p.get('amount','?')})"
    if et == "login":
        return f"login ({p.get('feature_or_page','?')}, {p.get('device_type','?')})"
    if et == "ticket_created":
        return f"support ticket: {p.get('category','?')}"
    if et == "search_query":
        return f"search: {p.get('search_text','?')}"
    if et == "kyc_update":
        return f"KYC update: {p.get('event_subtype','?')}"
    if et in ("address_change", "marital_status_change", "dependents_change"):
        return f"{et}: {p.get('old_value','?')} -> {p.get('new_value','?')}"
    return et


def _importance_for(ev: RawEvent) -> float:
    """
    Heuristic base weight. High-signal events (KYC, support, life, large tx)
    get higher base weight so they survive decay longer.
    """
    et = ev.event_type
    if et in ("address_change", "marital_status_change", "dependents_change",
              "kyc_update", "loan_application", "loan_disbursed"):
        return 0.95
    if et in ("ticket_created", "call_transcript"):
        return 0.85
    if et in ("outbound_transfer", "inbound_transfer", "withdrawal", "deposit"):
        amt = float((ev.payload or {}).get("amount", 0) or 0)
        return min(0.5 + amt / 10000.0, 0.9)
    if et == "purchase":
        amt = float((ev.payload or {}).get("amount", 0) or 0)
        return min(0.2 + amt / 20000.0, 0.6)
    return 0.3


class EpisodicMemory:
    def __init__(self) -> None:
        self._by_customer: dict[str, list[MemoryEntry]] = defaultdict(list)

    # -- writes -------------------------------------------------------------

    def add_event(self, ev: RawEvent, trace_id: str | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            customer_id=ev.customer_id,
            kind=self._kind_for(ev),
            summary=_summarize_event(ev),
            payload=ev.payload or {},
            t_event=ev.event_time,
            w_base=_importance_for(ev),
            source_event_id=ev.event_id,
            trace_id=trace_id,
        )
        self._by_customer[ev.customer_id].append(entry)
        return entry

    def add_note(self, customer_id: str, summary: str,
                 w_base: float = 0.7, kind: str = "note",
                 payload: dict[str, Any] | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            customer_id=customer_id,
            kind=kind,  # type: ignore[arg-type]
            summary=summary,
            payload=payload or {},
            t_event=datetime.now(timezone.utc),
            w_base=w_base,
        )
        self._by_customer[customer_id].append(entry)
        return entry

    def reinforce(self, customer_id: str, memory_id: str,
                  delta: float = 0.2) -> None:
        for m in self._by_customer.get(customer_id, []):
            if m.memory_id == memory_id:
                m.r_reinforce += delta
                return

    # -- reads --------------------------------------------------------------

    def all_for(self, customer_id: str) -> list[MemoryEntry]:
        return list(self._by_customer.get(customer_id, []))

    def retrieve(
        self,
        customer_id: str,
        *,
        kinds: Iterable[str] | None = None,
        top_k: int = 10,
        min_activation: float = 0.05,
        t_now: datetime | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return top-K memories by decay-weighted activation."""
        with span("episodic.retrieve", customer_id=customer_id,
                  top_k=top_k) as s:
            kinds_set = set(kinds) if kinds else None
            scored: list[tuple[MemoryEntry, float]] = []
            for m in self._by_customer.get(customer_id, []):
                if kinds_set and m.kind not in kinds_set:
                    continue
                score = activation(
                    w_base=m.w_base,
                    t_event=m.t_event,
                    r_reinforce=m.r_reinforce,
                    t_now=t_now,
                )
                if score >= min_activation:
                    scored.append((m, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            out = scored[:top_k]
            s.attributes["candidates"] = len(scored)
            s.attributes["returned"] = len(out)
            return out

    def customers(self) -> list[str]:
        return list(self._by_customer.keys())

    def size(self, customer_id: str) -> int:
        return len(self._by_customer.get(customer_id, []))

    # -- helpers ------------------------------------------------------------

    def _kind_for(self, ev: RawEvent) -> str:
        if ev.event_type in ("address_change", "marital_status_change",
                             "dependents_change", "kyc_update",
                             "loan_application", "loan_disbursed"):
            return "life_event"
        if ev.event_type in ("ticket_created", "call_transcript"):
            return "risk_flag"
        return "event"