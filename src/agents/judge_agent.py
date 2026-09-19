"""
Judge Agent — resolves the debate. Reason tier, temperature 0.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import JudgeVerdict
from src.streaming.event_generator import RawEvent


class JudgeAgent(BaseAgent):
    name = "judge_agent"
    role = "judge_agent"
    description = "Impartial arbiter — picks the winning strategy from a debate."
    output_schema = JudgeVerdict
    max_visible_events = 0
    max_tokens = 700
    temperature = 0.0

    system_prompt = """You are the JUDGE in a bank's Customer 360 debate.

You receive:
  1. The Synthesis Agent's unified customer state
  2. The Retention persona's argument
  3. The Growth persona's argument

Your job: pick the winning strategy OR declare it balanced.

Output STRICTLY valid JSON matching this exact schema:
{
  "winning_strategy": "retention" | "growth" | "balanced",
  "rationale": "<2-3 sentences explaining your decision. Be specific about which evidence is decisive.>",
  "recommended_action": "<one of: no_action, proactive_retention_outreach, relationship_manager_escalation, personalized_offer, support_intervention, compliance_fraud_hold>",
  "confidence": <float in [0, 1]>,
  "dissent_noted": <string or null — if the losing side had a legitimate point, acknowledge it here.>
}

Rules:
- "balanced" is valid when BOTH positions have merit and a combined action is best.
- Do NOT simply pick the side with higher self-reported confidence — evaluate the arguments.
- If the evidence genuinely supports only one side, pick it and explain why.
- Your recommended_action must be consistent with your winning_strategy:
    retention → proactive_retention_outreach or relationship_manager_escalation
    growth    → personalized_offer
    balanced  → relationship_manager_escalation (human decides)
- Return ONLY the JSON object, no prose, no markdown fences.
"""

    def build_user_prompt(
        self,
        *,
        customer_id: str,
        events: list[RawEvent],
        extra_context: dict[str, Any],
    ) -> str:
        ctx = extra_context or {}
        return json.dumps(
            {
                "customer_id": self.pii.tokenize(customer_id),
                "synthesis": ctx.get("synthesis"),
                "retention_argument": ctx.get("retention_position"),
                "growth_argument": ctx.get("growth_position"),
            },
            default=str,
        )