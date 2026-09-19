"""
Life-Event Inference Agent — agent-dependent trigger, reason tier.

Fires only when the swarm has produced correlated findings. Reads the
State Board (not raw events alone) to reason across signals.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import LifeEventFindings
from src.streaming.event_generator import RawEvent


class LifeEventAgent(BaseAgent):
    name = "life_event_agent"
    role = "life_event_agent"
    description = "Correlates swarm findings into a life-event belief with evidence."
    output_schema = LifeEventFindings
    max_visible_events = 60
    max_tokens = 900

    system_prompt = """You are the Life-Event Inference Agent in a bank's Customer 360 system.

You are triggered AFTER the swarm (Usage, Support, Transaction, KYC agents) has
published findings. You see BOTH the raw events (already filtered by your RBAC scope)
AND the swarm findings via the shared State Board snapshot.

Your job: infer the customer's current life phase as a BELIEF with evidence — not a fact.

Output STRICTLY valid JSON matching this exact schema:
{
  "inferred_state": "<one of the exact enum values listed below>",
  "life_phase": "<free-text label, e.g. 'Home Purchase', 'Medical Hardship', 'Job Change'>",
  "supporting_evidence": [<event_ids that most directly support this inference>],
  "needs_sensitive_handling": <bool — true if the inference involves medical, financial distress, marital change, or relocation. Those must route to HITL before any customer-facing action.>,
  "confidence": <float in [0, 1]>,
  "notes": "<1-2 sentence reasoning>"
}

Allowed values for `inferred_state` (use EXACTLY these strings):
  "no_significant_event"
  "new_child_life_event"
  "marriage_or_relationship_change"
  "job_change_or_promotion"
  "job_loss_or_income_disruption"
  "medical_hardship"
  "financial_distress_general"
  "relocation"
  "retirement_transition"
  "wealth_growth_or_windfall"
  "potential_fraud_or_takeover"
  "elder_vulnerability_or_scam_risk"
  "churn_risk"
  "small_business_cashflow_event"

Rules:
- This is a BELIEF, not a fact. Reflect uncertainty in `confidence`.
- Prefer `no_significant_event` with low confidence if signals are weak or contradictory.
- NEVER invent supporting_evidence — cite actual event_ids from the input.
- A support ticket mentioning "hospital" + "income dropped" is medical_hardship with needs_sensitive_handling=true.
- A KYC address_change + large deposit + home-loan search is relocation or wealth_growth with needs_sensitive_handling=true.
- Set needs_sensitive_handling=true whenever the inference touches health, finances, family, or home.
- Return ONLY the JSON object, no prose, no markdown fences.
"""

    def build_user_prompt(
        self,
        *,
        customer_id: str,
        events: list[RawEvent],
        extra_context: dict[str, Any],
    ) -> str:
        board = extra_context.get("state_board", {})
        compact_events = [
            {
                "event_id": ev.event_id,
                "event_time": ev.event_time.isoformat(),
                "source_system": ev.source_system,
                "event_type": ev.event_type,
                "payload": ev.payload,
            }
            for ev in events[-self.max_visible_events:]
        ]
        return json.dumps(
            {
                "customer_id": self.pii.tokenize(customer_id),
                "swarm_findings": board,
                "recent_events": compact_events,
            },
            default=str,
        )