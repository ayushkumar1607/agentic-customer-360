"""
No Action Gate — the explicit "should we do nothing?" decision.

Deterministic. Runs before the Action Agent. If the gate fires, the pipeline
short-circuits to a terminal No Action checkpoint WITHOUT calling the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.schemas import ActionProposal, JudgeVerdict, SynthesisFindings
from src.output.schema import Action
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class NoActionResult:
    is_no_action: bool
    reason: str


def check_no_action(
    *,
    synthesis: SynthesisFindings | None,
    judge: JudgeVerdict | None = None,
    action_proposal: ActionProposal | None = None,
) -> NoActionResult:
    # 1. Judge explicitly says no action
    if judge is not None and judge.recommended_action == Action.NO_ACTION:
        return NoActionResult(True, "judge recommended no_action")

    # 2. Action agent itself proposed no_action
    if action_proposal is not None and action_proposal.action == Action.NO_ACTION:
        return NoActionResult(True, "action agent proposed no_action")

    # 3. Synthesis says nothing meaningful
    if synthesis is not None:
        if synthesis.confidence < 0.4:
            return NoActionResult(
                True,
                f"synthesis confidence below threshold ({synthesis.confidence:.2f})",
            )
        if (
            synthesis.churn_risk_score < 0.30
            and not synthesis.opportunity_flag
            and not synthesis.arbitration_needed
        ):
            return NoActionResult(
                True,
                f"churn risk {synthesis.churn_risk_score:.2f} below 0.30 "
                "and no opportunity/conflict",
            )

    return NoActionResult(False, "gate did not fire")