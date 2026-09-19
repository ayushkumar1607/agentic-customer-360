"""
Synthesis / Correlation Agent — handoff stage, reason tier.

Reads every finding on the State Board, produces ONE coherent customer state,
and decides whether the swarm's findings genuinely conflict (which would
trigger debate).
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import SynthesisFindings
from src.streaming.event_generator import RawEvent


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    role = "synthesis_agent"
    description = "Reconciles swarm + life-event findings into unified state."
    output_schema = SynthesisFindings
    max_visible_events = 0
    max_tokens = 1000
    temperature = 0.2

    system_prompt = """You are the Synthesis / Correlation Agent in a bank's Customer 360 system.

You receive the STRUCTURED FINDINGS from all other agents on the shared State Board.
You do NOT receive raw events.

Your job: produce ONE coherent customer state, a churn risk score, and decide
whether the swarm's findings genuinely conflict (which would trigger a debate).

Output STRICTLY valid JSON matching this exact schema:
{
  "customer_state": "<one-line unified state, e.g. 'High churn risk with medical-hardship context'>",
  "inferred_state": "<exact enum value from the list below>",
  "churn_risk_score": <float in [0, 1]>,
  "churn_risk_drivers": [<short strings naming the drivers, e.g. 'login decline', 'negative support ticket'>],
  "opportunity_flag": <bool, true if there is a genuine growth/upsell opportunity>,
  "opportunity_description": <string or null>,
  "arbitration_needed": <bool, true ONLY if findings genuinely conflict>,
  "conflict_description": <string or null — describe the specific conflict if arbitration_needed=true>,
  "confidence": <float in [0, 1]>,
  "notes": "<one-sentence rationale>"
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
- arbitration_needed = true ONLY when two findings pull in genuinely opposite directions
  (e.g. churn risk vs. wealth growth). Mere co-occurrence is not conflict.
- If there is only one strong signal, arbitration_needed = false.
- churn_risk_score must be justified by the drivers you list.
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
        return json.dumps(
            {
                "customer_id": self.pii.tokenize(customer_id),
                "state_board_findings": board,
            },
            default=str,
        )