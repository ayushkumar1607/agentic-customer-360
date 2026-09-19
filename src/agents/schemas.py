"""
Typed output schemas for every agent.

Every schema MUST have a `confidence: float` field — BaseAgent extracts it
for the State Board. Other fields are agent-specific.

The `InferredState` and `Action` enums come from src.output.schema, which
mirrors the README's allowed values exactly. Do NOT invent new enum values.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.output.schema import Action, InferredState


# ---- swarm agents ----------------------------------------------------------

class UsageFindings(BaseModel):
    """Usage / Engagement Agent output."""
    login_trend_30d: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Relative change in login frequency (7d vs 30d). Negative = declining.",
    )
    login_frequency_7d: float = Field(..., ge=0.0)
    session_dropoff_flag: bool = Field(
        ..., description="True if there is a marked session-length or engagement drop."
    )
    feature_disengagement: list[str] = Field(
        default_factory=list,
        description="Features/pages the customer stopped using.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class SupportFindings(BaseModel):
    """Support / Sentiment Agent output — the AUTHORITATIVE sentiment source."""
    ticket_urgency: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Authoritative sentiment. -1 = very negative, +1 = very positive.",
    )
    churn_intent_flag: bool = Field(
        ..., description="True if the customer signals intent to leave."
    )
    category: str = Field(
        ..., description="Free-text ticket category.",
    )
    life_event_hint: Optional[str] = Field(
        default=None,
        description="If the ticket text hints at a life event, name it here.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class TransactionFindings(BaseModel):
    """Transaction / Billing Agent output."""
    spend_velocity_variance: float = Field(
        ..., description="Variance of spend velocity vs the customer's own baseline."
    )
    failed_billing_count: int = Field(..., ge=0)
    large_transaction_flag: bool = Field(
        ..., description="True if a single transaction is significantly larger than baseline."
    )
    anomalous_pattern: Optional[str] = Field(
        default=None,
        description="Free-text label if an anomaly is detected. Null otherwise.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class KYCFindings(BaseModel):
    """KYC / Compliance Agent output."""
    kyc_valid: bool
    sanctions_match: bool
    address_changed: bool
    marital_status_change: bool
    dependents_change: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


# ---- life-event & synthesis -----------------------------------------------

class LifeEventFindings(BaseModel):
    """Life-Event Inference Agent output — cross-signal belief with evidence."""
    inferred_state: InferredState = Field(
        ..., description="Strict enum from README."
    )
    life_phase: str = Field(
        ..., description="Free-text label (e.g. 'Home Purchase', 'Medical Hardship')."
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="event_ids that support this inference.",
    )
    needs_sensitive_handling: bool = Field(
        ...,
        description="True if the inference is sensitive and any action must go to HITL.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


class SynthesisFindings(BaseModel):
    """Synthesis / Correlation Agent output — the unified customer state."""
    customer_state: str = Field(
        ..., description="One-line unified state."
    )
    inferred_state: InferredState
    churn_risk_score: float = Field(..., ge=0.0, le=1.0)
    churn_risk_drivers: list[str] = Field(default_factory=list)
    opportunity_flag: bool = False
    opportunity_description: Optional[str] = None
    arbitration_needed: bool = Field(
        ...,
        description="True if the swarm findings genuinely conflict.",
    )
    conflict_description: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""


# ---- debate ----------------------------------------------------------------

class DebatePosition(BaseModel):
    """One side of a Retention vs Growth debate."""
    position_label: Literal["retention", "growth"]
    argument: str = Field(..., description="2-4 sentence position statement.")
    recommended_action: Action
    key_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class JudgeVerdict(BaseModel):
    """Judge's resolution of a debate."""
    winning_strategy: Literal["retention", "growth", "balanced"]
    rationale: str
    recommended_action: Action
    confidence: float = Field(..., ge=0.0, le=1.0)
    dissent_noted: Optional[str] = Field(default=None) 
    
    
# ---- action + critique -----------------------------------------------------

class ActionProposal(BaseModel):
    """Action Agent output — a bounded, costed proposal (never executed)."""
    action: Action
    action_subtype: Optional[str] = None
    cost_estimate_usd: float = Field(..., ge=0.0)
    message_draft: Optional[str] = Field(
        default=None,
        description="Customer-facing message if the action is customer-facing. "
                    "Null for internal-only actions like holds or RM escalations.",
    )
    eligibility_notes: str = Field(..., description="Why this action is eligible.")
    supporting_evidence: list[str] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    iteration: int = Field(default=1, ge=1, le=3)


class CritiqueFeedback(BaseModel):
    """Critique / Compliance Agent output — feedback for the Action Agent."""
    verdict: Literal["pass", "revise", "reject"]
    issues: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)
    revision_suggestions: list[str] = Field(default_factory=list)
    cost_within_limits: bool = True
    tone_appropriate: bool = True
    compliance_clean: bool = True
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""