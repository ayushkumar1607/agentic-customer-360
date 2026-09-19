"""
Debate Agents — two personas, different model tiers for genuine divergence.

Retention persona argues from the churn-risk perspective (reason tier).
Growth persona argues from the opportunity perspective (alt tier).

Both read the synthesis output from the State Board.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import DebatePosition
from src.streaming.event_generator import RawEvent


_DEBATE_SCHEMA_HINT = """
Output STRICTLY valid JSON matching this exact schema:
{
  "position_label": "<must match your assigned persona: 'retention' or 'growth'>",
  "argument": "<2-4 sentence position statement. Be persuasive but grounded in evidence.>",
  "recommended_action": "<one of: no_action, proactive_retention_outreach, relationship_manager_escalation, personalized_offer, support_intervention, compliance_fraud_hold>",
  "key_evidence": [<event_ids or finding keys that support your position>],
  "confidence": <float in [0, 1]>
}

Rules:
- You are arguing ONE side. You may acknowledge the other side's facts, but you must
  make YOUR case clearly and recommend YOUR action.
- Base every claim on the findings provided. Do NOT invent events.
- Return ONLY the JSON object, no prose, no markdown fences.
"""


class RetentionDebateAgent(BaseAgent):
    name = "debate_retention"
    role = "synthesis_agent"
    description = "Retention persona — argues for keeping the customer."
    output_schema = DebatePosition
    max_visible_events = 0
    max_tokens = 700

    system_prompt = (
        "You are the RETENTION PERSPECTIVE in a bank's Customer 360 debate.\n\n"
        "Your bias: preserve the relationship. Argue that any churn signals deserve "
        "attention and that proactive retention actions are warranted.\n\n"
        "You will receive the Synthesis Agent's unified customer state and supporting findings.\n\n"
        + _DEBATE_SCHEMA_HINT +
        "\nSet position_label = 'retention'."
    )

    def build_user_prompt(self, *, customer_id, events, extra_context) -> str:
        return json.dumps(
            {"customer_id": self.pii.tokenize(customer_id),
             "synthesis_and_findings": extra_context.get("state_board", {})},
            default=str,
        )


class GrowthDebateAgent(BaseAgent):
    name = "debate_growth"
    role = "synthesis_agent"
    description = "Growth persona — argues for expansion opportunity."
    output_schema = DebatePosition
    max_visible_events = 0
    max_tokens = 700

    system_prompt = (
        "You are the GROWTH PERSPECTIVE in a bank's Customer 360 debate.\n\n"
        "Your bias: expand the relationship. Argue that any apparent distress signals "
        "may mask a legitimate life transition (relocation, windfall, new dependent) "
        "that merits a proactive offer, not a defensive retention gesture.\n\n"
        "You will receive the Synthesis Agent's unified customer state and supporting findings.\n\n"
        + _DEBATE_SCHEMA_HINT +
        "\nSet position_label = 'growth'."
    )

    def build_user_prompt(self, *, customer_id, events, extra_context) -> str:
        return json.dumps(
            {"customer_id": self.pii.tokenize(customer_id),
             "synthesis_and_findings": extra_context.get("state_board", {})},
            default=str,
        )