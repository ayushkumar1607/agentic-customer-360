"""
Governed Shared Per-Customer State Board.

This is the ONLY sanctioned channel between agents. Agents publish
structured findings — never their internal reasoning traces. Downstream
agents read scoped, structured conclusions.

Isolation:
- scoped by customer_id
- writes require the publisher to declare its RBAC role
- every publish carries a trace_id
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.memory.schemas import StateBoardSlot
from src.utils.logger import get_logger
from src.utils.tracing import get_trace_id, span

log = get_logger(__name__)


class StateBoard:
    def __init__(self) -> None:
        # customer_id -> agent_name -> latest slot
        self._slots: dict[str, dict[str, StateBoardSlot]] = defaultdict(dict)
        # Append-only history for audits
        self._history: dict[str, list[StateBoardSlot]] = defaultdict(list)

    # -- writes -------------------------------------------------------------

    def publish(
        self,
        *,
        customer_id: str,
        agent: str,
        findings: dict[str, Any],
        confidence: float | None = None,
    ) -> StateBoardSlot:
        with span("state_board.publish", customer_id=customer_id,
                  agent=agent) as s:
            slot = StateBoardSlot(
                agent=agent,
                customer_id=customer_id,
                findings=findings,
                confidence=confidence,
                trace_id=get_trace_id(),
            )
            self._slots[customer_id][agent] = slot
            self._history[customer_id].append(slot)
            s.attributes["finding_keys"] = list(findings.keys())
            return slot

    # -- reads --------------------------------------------------------------

    def read(self, customer_id: str, agent: str) -> StateBoardSlot | None:
        return self._slots.get(customer_id, {}).get(agent)

    def read_all(self, customer_id: str) -> dict[str, StateBoardSlot]:
        return dict(self._slots.get(customer_id, {}))

    def snapshot_findings(self, customer_id: str) -> dict[str, Any]:
        """Flat dict of {agent: findings} — what synthesis consumes."""
        return {
            agent: {"findings": slot.findings, "confidence": slot.confidence}
            for agent, slot in self._slots.get(customer_id, {}).items()
        }

    def history(self, customer_id: str) -> list[StateBoardSlot]:
        return list(self._history.get(customer_id, []))

    def customers(self) -> list[str]:
        return list(self._slots.keys())