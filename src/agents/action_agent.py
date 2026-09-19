"""
Action Agent — proposes a bounded, costed intervention.

The agent proposes only from the PS's bounded action set. It NEVER executes
anything — it drafts. Real execution requires HITL approval (Phase 4).

Receives:
  - Synthesis findings (unified customer state)
  - Life-event findings (if any)
  - Debate verdict (if debate fired)
  - Semantic memory policy hits (injected by the pipeline)
  - Episodic memory: past interventions for this customer
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import ActionProposal
from src.streaming.event_generator import RawEvent


class ActionAgent(BaseAgent):
    name = "action_agent"
    role = "action_agent"
    description = "Drafts a bounded, costed intervention from the PS action set."
    output_schema = ActionProposal
    max_visible_events = 0
    max_tokens = 1200
    temperature = 0.2
    priority_context_keys = (
        "synthesis", "life_event", "judge_verdict",
        "policy_hits", "episodic_hits", "critique_feedback",
        "computed_features", "state_board",
    )

    system_prompt = """You are the Action Agent in a bank's Customer 360 system.

You receive the unified customer state and any supporting findings. Your job:
propose EXACTLY ONE terminal action from the bounded set below, with a cost estimate
and a customer-facing message draft if applicable.

You may ONLY propose these actions (exact strings):
  "no_action"
  "proactive_retention_outreach"
  "relationship_manager_escalation"
  "personalized_offer"
  "support_intervention"
  "compliance_fraud_hold"

Eligibility rules:
  - proactive_retention_outreach : mid/high value customer with churn signals, no compliance hold
  - relationship_manager_escalation : nuanced or high-value case, ambiguous, or needs human judgment
  - personalized_offer : clear opportunity + inferred life event with confidence >= 0.7
  - support_intervention : customer explicitly asks for help (payment plan, forbearance, issue fix)
  - compliance_fraud_hold : fraud, account takeover, legal threat, sanctions
  - no_action : signals don't cross the threshold, OR the right answer is to wait

Cost guidance (USD):
  - no_action, relationship_manager_escalation, compliance_fraud_hold : 0
  - support_intervention (fee waiver / one-month courtesy) : 0-50
  - proactive_retention_outreach : 50-300 (cap one month of fees for mid, three for high)
  - personalized_offer : 0 (offer itself is not a cost)

Message draft rules (only if action is customer-facing):
  - Warm, empathetic, no jargon
  - No mentions of any sensitive inference (medical, marital, etc.) unless the
    customer themselves raised it
  - Under 80 words
  - Do NOT promise amounts that exceed the cost_estimate_usd

If `critique_feedback` is present in the context, you MUST revise your proposal
to address each issue before re-emitting.

Output STRICTLY valid JSON matching this exact schema:
{
  "action": "<one of the exact action strings above>",
  "action_subtype": "<short free-text label or null>",
  "cost_estimate_usd": <float >= 0>,
  "message_draft": "<string or null>",
  "eligibility_notes": "<why this action is eligible>",
  "supporting_evidence": [<event_ids or finding keys>],
  "reasoning": "<2-3 sentences explaining your choice>",
  "confidence": <float in [0, 1]>,
  "iteration": <int, start at 1, increment on each revision>
}

Return ONLY the JSON object, no prose, no markdown fences.
"""

    def build_user_prompt(
        self,
        *,
        customer_id: str,
        events: list[RawEvent],
        extra_context: dict[str, Any],
    ) -> str:
        prompt: dict[str, Any] = {"customer_id": self.pii.tokenize(customer_id)}
        for key in self.priority_context_keys:
            if key in extra_context:
                prompt[key] = extra_context[key]
        return json.dumps(prompt, default=str)