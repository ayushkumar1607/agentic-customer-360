"""No Action gate should fire only when the evidence is truly weak."""
from __future__ import annotations

from src.agents.schemas import SynthesisFindings
from src.orchestration.no_action_gate import check_no_action
from src.output.schema import InferredState


def _synthesis(**overrides) -> SynthesisFindings:
    base = dict(
        customer_state="stable",
        inferred_state=InferredState.NO_SIGNIFICANT_EVENT,
        churn_risk_score=0.10,
        churn_risk_drivers=[],
        opportunity_flag=False,
        opportunity_description=None,
        arbitration_needed=False,
        conflict_description=None,
        confidence=0.8,
        notes="",
    )
    base.update(overrides)
    return SynthesisFindings(**base)


def test_gate_fires_on_low_churn_no_opportunity():
    r = check_no_action(synthesis=_synthesis())
    assert r.is_no_action is True


def test_gate_does_not_fire_on_high_churn():
    r = check_no_action(synthesis=_synthesis(churn_risk_score=0.72))
    assert r.is_no_action is False


def test_gate_fires_on_low_synthesis_confidence():
    r = check_no_action(synthesis=_synthesis(confidence=0.3))
    assert r.is_no_action is True


def test_gate_does_not_fire_on_opportunity():
    r = check_no_action(synthesis=_synthesis(
        churn_risk_score=0.20, opportunity_flag=True,
    ))
    assert r.is_no_action is False