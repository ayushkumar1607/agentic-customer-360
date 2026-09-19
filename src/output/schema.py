"""Strict enums and schemas for inferred-events output.

Source: README_dataset_schema.md — "report strictly these fixed enum values".
Do NOT invent new values here without updating this file.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InferredState(str, Enum):
    NO_SIGNIFICANT_EVENT = "no_significant_event"
    NEW_CHILD_LIFE_EVENT = "new_child_life_event"
    MARRIAGE_OR_RELATIONSHIP_CHANGE = "marriage_or_relationship_change"
    JOB_CHANGE_OR_PROMOTION = "job_change_or_promotion"
    JOB_LOSS_OR_INCOME_DISRUPTION = "job_loss_or_income_disruption"
    MEDICAL_HARDSHIP = "medical_hardship"
    FINANCIAL_DISTRESS_GENERAL = "financial_distress_general"
    RELOCATION = "relocation"
    RETIREMENT_TRANSITION = "retirement_transition"
    WEALTH_GROWTH_OR_WINDFALL = "wealth_growth_or_windfall"
    POTENTIAL_FRAUD_OR_TAKEOVER = "potential_fraud_or_takeover"
    ELDER_VULNERABILITY_OR_SCAM_RISK = "elder_vulnerability_or_scam_risk"
    CHURN_RISK = "churn_risk"
    SMALL_BUSINESS_CASHFLOW_EVENT = "small_business_cashflow_event"


class Action(str, Enum):
    NO_ACTION = "no_action"
    PROACTIVE_RETENTION_OUTREACH = "proactive_retention_outreach"
    RELATIONSHIP_MANAGER_ESCALATION = "relationship_manager_escalation"
    PERSONALIZED_OFFER = "personalized_offer"
    SUPPORT_INTERVENTION = "support_intervention"
    COMPLIANCE_FRAUD_HOLD = "compliance_fraud_hold"


class HITLStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    ESCALATED = "escalated"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_MODIFIED = "human_modified"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Checkpoint(BaseModel):
    """One row of the final inferred-events output."""
    as_of_time: str
    inferred_state: InferredState
    confidence_band: ConfidenceBand
    action: Action
    action_subtype: Optional[str] = None
    hitl_status: HITLStatus
    notes: Optional[str] = None


def confidence_to_band(score: float) -> ConfidenceBand:
    if score >= 0.75:
        return ConfidenceBand.HIGH
    if score >= 0.45:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW