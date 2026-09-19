"""
Role-based access control enforced at the DATA layer.

An agent's role determines which `source_system` values it can see.
This is enforced in code (not in a prompt), so an LLM cannot talk its way
into reading tables it wasn't granted.

PS (page 9) — "Role-based information access enforced at the data layer,
not just the prompt layer — an agent that isn't supposed to see
trading/brokerage data, for instance, should be structurally unable to
query it, not merely instructed not to."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.streaming.event_generator import RawEvent
from src.utils.logger import get_logger

log = get_logger(__name__)


# role -> set of allowed source_system values ("*" = all)
ROLE_SCOPES: dict[str, set[str]] = {
    "usage_agent":       {"web_app_events"},
    "support_agent":     {"support_logs", "social_signal_consented"},
    "transaction_agent": {"card_payments", "instant_payments", "ach_wire",
                          "core_banking_ledger", "trading_brokerage"},
    "kyc_agent":         {"loan_kyc"},
    "life_event_agent":  {"card_payments", "web_app_events", "loan_kyc",
                          "core_banking_ledger", "support_logs"},
    "synthesis_agent":   {"*"},
    "action_agent":      {"*"},
    "critique_agent":    {"*"},
    "judge_agent":       {"*"},
    "guardrail_agent":   {"*"},
}


class RBACViolation(Exception):
    pass


@dataclass
class RBAC:
    role: str
    scopes: set[str]

    def allows_source(self, source: str) -> bool:
        return "*" in self.scopes or source in self.scopes

    def filter_events(self, events: Iterable[RawEvent]) -> list[RawEvent]:
        out: list[RawEvent] = []
        blocked = 0
        for ev in events:
            if self.allows_source(ev.source_system):
                out.append(ev)
            else:
                blocked += 1
        if blocked:
            log.debug("rbac.filtered", role=self.role, blocked=blocked)
        return out

    def assert_allows(self, source: str) -> None:
        if not self.allows_source(source):
            raise RBACViolation(
                f"role '{self.role}' is not permitted to read source '{source}'"
            )


def rbac_for(role: str) -> RBAC:
    if role not in ROLE_SCOPES:
        raise RBACViolation(f"unknown role: {role}")
    return RBAC(role=role, scopes=set(ROLE_SCOPES[role]))