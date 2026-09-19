"""
Full end-to-end pipeline.

Stage sequence:
  1. Stream events -> CEP
  2. Compute rolling features
  3. Guardrail scan (per event, hard-stop diversion)
  4. Swarm: Usage, Support, Transaction, KYC (parallel)
  5. Agent-dependent trigger -> Life-Event
  6. Synthesis
  7. If synthesis.arbitration_needed: Debate (retention vs growth) -> Judge
  8. No Action Gate (deterministic)
  9. Action Agent + Critique (max 2 iterations)
     ** Special handling: sensitive life events force RM escalation deterministically. **
 10. Guardrail override on final message draft
 11. HITL if required
 12. Write to inferred-events.jsonl
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.action_agent import ActionAgent
from src.agents.critique_agent import CritiqueAgent
from src.agents.debate_agent import GrowthDebateAgent, RetentionDebateAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.kyc_agent import KYCAgent
from src.agents.life_event_agent import LifeEventAgent
from src.agents.schemas import (
    ActionProposal,
    CritiqueFeedback,
    LifeEventFindings,
    SynthesisFindings,
)
from src.agents.support_agent import SupportAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.transaction_agent import TransactionAgent
from src.agents.usage_agent import UsageAgent
from src.guardrails.pii import PIITokenizer
from src.guardrails.scanner import GuardrailScanner
from src.hitl.approval_queue import HITLQueue, HITLRequest
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory
from src.memory.state_board import StateBoard
from src.orchestration.no_action_gate import check_no_action
from src.output.inferred_events_writer import InferredEventsWriter
from src.output.schema import Action, HITLStatus, InferredState
from src.streaming.cep_engine import CEPEngine
from src.streaming.event_generator import stream_scenario
from src.streaming.features import extract_features
from src.utils.logger import get_logger
from src.utils.tracing import get_trace_id, new_trace_id, set_trace_id

log = get_logger("pipeline")

LIFE_EVENT_MIN_SIGNALS = 2
MAX_CRITIQUE_ITERATIONS = 2
HITL_COST_THRESHOLD_USD = 200.0
HITL_CONFIDENCE_MIN = 0.5

# Deterministic safe-action rule:
# If the life event is flagged as sensitive, the pipeline forces this action
# regardless of what the Action Agent would have proposed. This is a PS-level
# requirement (page 4: sensitive inferences must not trigger autonomous
# customer-facing actions).
SENSITIVE_FORCED_ACTION = Action.RELATIONSHIP_MANAGER_ESCALATION


# --- helpers ----------------------------------------------------------------

def _agent_flagged_life_event_signal(name: str, f: dict[str, Any]) -> bool:
    if name == "support_agent":
        return bool(
            f.get("life_event_hint")
            or f.get("ticket_urgency") in ("HIGH", "CRITICAL")
            or (f.get("sentiment_score") or 0) <= -0.5
        )
    if name == "usage_agent":
        return bool(
            f.get("session_dropoff_flag") is True
            or (f.get("login_trend_30d") or 0.0) < -0.15
        )
    if name == "transaction_agent":
        return bool(
            f.get("large_transaction_flag") is True
            or f.get("anomalous_pattern") is not None
            or (f.get("failed_billing_count") or 0) > 0
            or (f.get("spend_velocity_variance") or 0.0) > 2.0
        )
    if name == "kyc_agent":
        return bool(
            f.get("address_changed")
            or f.get("marital_status_change")
            or f.get("dependents_change")
        )
    return False


def _life_event_should_fire(
    board: StateBoard, customer_id: str
) -> tuple[bool, list[str]]:
    slots = board.read_all(customer_id)
    flagged = [
        n for n, s in slots.items() if _agent_flagged_life_event_signal(n, s.findings)
    ]
    return len(flagged) >= LIFE_EVENT_MIN_SIGNALS, flagged


def _requires_hitl(
    *,
    proposal: ActionProposal,
    synthesis: SynthesisFindings | None,
    life_event: LifeEventFindings | None,
    guardrail_hit: bool,
) -> tuple[bool, str]:
    if guardrail_hit:
        return True, "guardrail hard-stop"
    if proposal.cost_estimate_usd > HITL_COST_THRESHOLD_USD:
        return True, (
            f"cost ${proposal.cost_estimate_usd:.0f} exceeds "
            f"${HITL_COST_THRESHOLD_USD:.0f}"
        )
    if proposal.confidence < HITL_CONFIDENCE_MIN:
        return True, (
            f"confidence {proposal.confidence:.2f} below {HITL_CONFIDENCE_MIN}"
        )
    if life_event is not None and life_event.needs_sensitive_handling:
        return True, "sensitive life-event inference"
    if synthesis is not None and synthesis.arbitration_needed:
        return True, "unresolved agent disagreement"
    if proposal.action in (
        Action.COMPLIANCE_FRAUD_HOLD,
        Action.RELATIONSHIP_MANAGER_ESCALATION,
        Action.PERSONALIZED_OFFER,
    ):
        return True, f"high-risk action type: {proposal.action.value}"
    return False, ""


def _build_sensitive_override(
    life_event: LifeEventFindings,
) -> ActionProposal:
    """Deterministically construct the safe action for a sensitive inference."""
    return ActionProposal(
        action=SENSITIVE_FORCED_ACTION,
        action_subtype="sensitive_life_event_review",
        cost_estimate_usd=0.0,
        message_draft=None,
        eligibility_notes=(
            f"Life event '{life_event.inferred_state.value}' is flagged as "
            f"sensitive. Per POL-HITL-002 all customer-facing actions are "
            f"suspended and the case is routed to a relationship manager."
        ),
        supporting_evidence=list(life_event.supporting_evidence),
        reasoning=(
            f"Deterministic override: inferred state '{life_event.inferred_state.value}' "
            f"carries sensitive context (confidence {life_event.confidence:.2f}). "
            f"Human review is mandatory before any customer outreach."
        ),
        confidence=life_event.confidence,
        iteration=1,
    )


def _build_fallback_proposal(
    reason: str,
    life_event: LifeEventFindings | None,
    synthesis: SynthesisFindings | None,
) -> ActionProposal:
    """
    Fallback when the action loop fails or is rejected.

    If we have a life event, we MUST NOT do nothing — escalate to a human.
    Otherwise, safe to no_action.
    """
    if life_event is not None:
        return ActionProposal(
            action=Action.RELATIONSHIP_MANAGER_ESCALATION,
            action_subtype="action_loop_fallback",
            cost_estimate_usd=0.0,
            message_draft=None,
            eligibility_notes=(
                f"Fallback from failed action loop ({reason}). "
                f"Life event '{life_event.inferred_state.value}' detected, "
                f"human judgment required."
            ),
            supporting_evidence=list(life_event.supporting_evidence),
            reasoning=(
                f"Action pipeline did not produce an approved proposal ({reason}). "
                f"Because a life event was detected, the safe default is human "
                f"escalation rather than inaction."
            ),
            confidence=life_event.confidence,
            iteration=1,
        )
    # No life event -> no_action is the correct fallback
    conf = synthesis.confidence if synthesis else 0.5
    return ActionProposal(
        action=Action.NO_ACTION,
        action_subtype=None,
        cost_estimate_usd=0.0,
        message_draft=None,
        eligibility_notes=f"Fallback: {reason}",
        supporting_evidence=[],
        reasoning=(
            f"Action loop failed ({reason}) and no life event was detected. "
            f"No Action is the safe fallback."
        ),
        confidence=conf,
        iteration=1,
    )


# --- stages -----------------------------------------------------------------

async def _run_swarm(
    *,
    customer_id: str,
    events,
    computed_features: dict[str, Any],
    board: StateBoard,
    pii: PIITokenizer,
) -> None:
    ctx = {"computed_features": computed_features}
    agents = [
        UsageAgent(state_board=board, pii=pii),
        SupportAgent(state_board=board, pii=pii),
        TransactionAgent(state_board=board, pii=pii),
        KYCAgent(state_board=board, pii=pii),
    ]
    log.info("stage.swarm.start")

    async def _run(a):
        try:
            await a.run(customer_id=customer_id, events=events, extra_context=ctx)
        except Exception as e:
            log.error("stage.swarm.failed", agent=a.name, error=str(e))

    await asyncio.gather(*(_run(a) for a in agents))
    log.info("stage.swarm.done")


async def _run_debate_and_judge(
    *,
    customer_id: str,
    board: StateBoard,
    pii: PIITokenizer,
    synthesis: SynthesisFindings,
) -> dict[str, Any] | None:
    log.info("stage.debate.start", conflict=synthesis.conflict_description)
    ctx = {"state_board": board.snapshot_findings(customer_id)}

    async def _safe(a, name):
        try:
            return await a.run(
                customer_id=customer_id, events=[], extra_context=ctx
            )
        except Exception as e:
            log.error("stage.debate.failed", agent=name, error=str(e))
            return None

    r_pos, g_pos = await asyncio.gather(
        _safe(RetentionDebateAgent(state_board=board, pii=pii), "debate_retention"),
        _safe(GrowthDebateAgent(state_board=board, pii=pii), "debate_growth"),
    )
    if r_pos is None or g_pos is None:
        return None

    judge = JudgeAgent(state_board=board, pii=pii)
    try:
        verdict = await judge.run(
            customer_id=customer_id,
            events=[],
            extra_context={
                "synthesis": synthesis.model_dump(),
                "retention_position": r_pos.model_dump(),
                "growth_position": g_pos.model_dump(),
            },
        )
        log.info("stage.debate.done", verdict=verdict.winning_strategy)
        return verdict.model_dump()
    except Exception as e:
        log.error("stage.debate.judge_failed", error=str(e))
        return None


async def _run_action_loop(
    *,
    customer_id: str,
    board: StateBoard,
    pii: PIITokenizer,
    synthesis: SynthesisFindings,
    life_event: LifeEventFindings | None,
    judge_verdict: dict[str, Any] | None,
    policy_hits: list[dict[str, Any]],
    episodic_hits: list[dict[str, Any]],
    computed_features: dict[str, Any],
) -> tuple[ActionProposal | None, CritiqueFeedback | None]:
    """
    Run the Action Agent + Critique Agent refinement loop.

    Special rule: if a life event is present AND flagged sensitive,
    short-circuit to the deterministic RM-escalation proposal. We still run
    the Critique Agent once so the audit trail captures the compliance view.
    """
    action_agent = ActionAgent(state_board=board, pii=pii)
    critique_agent = CritiqueAgent(state_board=board, pii=pii)

    # --- DETERMINISTIC SENSITIVE-LIFE-EVENT OVERRIDE ---
    if life_event is not None and life_event.needs_sensitive_handling:
        log.info(
            "stage.action.sensitive_override",
            inferred_state=life_event.inferred_state.value,
            confidence=life_event.confidence,
        )
        proposal = _build_sensitive_override(life_event)
        try:
            feedback = await critique_agent.run(
                customer_id=customer_id,
                events=[],
                extra_context={
                    "proposal": proposal.model_dump(),
                    "synthesis": synthesis.model_dump(),
                    "life_event": life_event.model_dump(),
                    "policy_hits": policy_hits,
                    "judge_verdict": judge_verdict,
                },
            )
        except Exception as e:
            log.error("stage.critique.failed", error=str(e))
            feedback = None
        return proposal, feedback

    # --- NORMAL LOOP ---
    ctx_base = {
        "synthesis": synthesis.model_dump(),
        "life_event": life_event.model_dump() if life_event else None,
        "judge_verdict": judge_verdict,
        "policy_hits": policy_hits,
        "episodic_hits": episodic_hits,
        "computed_features": computed_features,
    }

    proposal: ActionProposal | None = None
    feedback: CritiqueFeedback | None = None

    for iteration in range(1, MAX_CRITIQUE_ITERATIONS + 1):
        log.info("stage.action.iteration", iteration=iteration)
        ctx = dict(ctx_base)
        if feedback is not None:
            ctx["critique_feedback"] = feedback.model_dump()

        try:
            proposal = await action_agent.run(
                customer_id=customer_id, events=[], extra_context=ctx
            )
        except Exception as e:
            log.error("stage.action.failed", error=str(e))
            return None, None

        try:
            feedback = await critique_agent.run(
                customer_id=customer_id,
                events=[],
                extra_context={
                    "proposal": proposal.model_dump(),
                    "synthesis": synthesis.model_dump(),
                    "life_event": life_event.model_dump() if life_event else None,
                    "policy_hits": policy_hits,
                    "judge_verdict": judge_verdict,
                },
            )
        except Exception as e:
            log.error("stage.critique.failed", error=str(e))
            return proposal, None

        log.info(
            "stage.critique.verdict",
            iteration=iteration,
            verdict=feedback.verdict,
            issues=feedback.issues,
            suggestions=feedback.revision_suggestions,
        )

        if feedback.verdict == "pass":
            log.info("stage.critique.passed", iteration=iteration)
            return proposal, feedback

        # On "revise" or "reject", loop for one more attempt with the
        # feedback as context. The last iteration's proposal will be
        # returned even if not approved — the pipeline decides the fallback.

    log.warning(
        "stage.critique.max_iterations_reached",
        final_verdict=feedback.verdict if feedback else "none",
    )
    return proposal, feedback


# --- main pipeline ----------------------------------------------------------

async def run_pipeline(
    *,
    scenario_dir: Path,
    hitl_queue: HITLQueue,
    writer: InferredEventsWriter,
    auto_hitl: bool = True,
) -> dict[str, Any]:
    board = StateBoard()
    pii = PIITokenizer()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    scanner = GuardrailScanner()

    set_trace_id(new_trace_id())
    pipeline_trace = get_trace_id()

    # --- 1. Stream + CEP + guardrails ---
    cep = CEPEngine()
    customer_id: str | None = None
    hard_stops: list[dict[str, Any]] = []
    last_event_time: datetime | None = None

    async for ev in stream_scenario(
        scenario_dir, seed_speed=1e6, live_speed=1e6, jitter_prob=0.0
    ):
        set_trace_id(new_trace_id())
        cep.ingest(ev)
        episodic.add_event(ev, trace_id=get_trace_id())
        customer_id = customer_id or ev.customer_id
        last_event_time = ev.event_time
        hs = scanner.scan_event(ev)
        if hs is not None:
            hard_stops.append(hs.to_dict())

    assert customer_id is not None
    log.info(
        "stage.stream.done", customer_id=customer_id, hard_stops=len(hard_stops)
    )

    # --- 2. Compute features ---
    computed_features = extract_features(cep.all_windows(customer_id))

    # --- 3. Guardrail diversion ---
    if hard_stops:
        log.warning("stage.guardrail.hard_stop", count=len(hard_stops))
        is_fraud = any(
            "fraud" in h.get("route", "")
            or "takeover" in h.get("reason", "").lower()
            for h in hard_stops
        )
        cp = writer.write(
            as_of_time=last_event_time,
            customer_id=customer_id,
            inferred_state=(
                InferredState.POTENTIAL_FRAUD_OR_TAKEOVER
                if is_fraud
                else InferredState.NO_SIGNIFICANT_EVENT
            ),
            confidence=0.95,
            action=Action.COMPLIANCE_FRAUD_HOLD,
            action_subtype=hard_stops[0].get("route"),
            hitl_status=HITLStatus.ESCALATED,
            notes=f"Hard-stop guardrail fired: {hard_stops[0].get('reason')}",
            trace_id=pipeline_trace,
        )
        return {
            "customer_id": customer_id,
            "checkpoint": cp.model_dump(),
            "hard_stop": hard_stops[0],
        }

    # --- 4. Swarm ---
    events_30d = cep.window(customer_id, "30d")
    await _run_swarm(
        customer_id=customer_id,
        events=events_30d,
        computed_features=computed_features,
        board=board,
        pii=pii,
    )

    # --- 5. Life-event (agent-dependent trigger) ---
    life_event: LifeEventFindings | None = None
    should_fire, flagged = _life_event_should_fire(board, customer_id)
    log.info("stage.life_event.check", flagged=flagged, should_fire=should_fire)
    if should_fire:
        le = LifeEventAgent(state_board=board, pii=pii)
        try:
            life_event = await le.run(
                customer_id=customer_id,
                events=events_30d,
                extra_context={
                    "state_board": board.snapshot_findings(customer_id),
                    "computed_features": computed_features,
                },
            )
        except Exception as e:
            log.error("stage.life_event.failed", error=str(e))

    # --- 6. Synthesis ---
    synthesis: SynthesisFindings | None = None
    synth = SynthesisAgent(state_board=board, pii=pii)
    try:
        synthesis = await synth.run(
            customer_id=customer_id,
            events=[],
            extra_context={
                "state_board": board.snapshot_findings(customer_id)
            },
        )
    except Exception as e:
        log.error("stage.synthesis.failed", error=str(e))

    # --- 7. Debate + Judge (only if conflict) ---
    judge_verdict: dict[str, Any] | None = None
    if synthesis is not None and synthesis.arbitration_needed:
        judge_verdict = await _run_debate_and_judge(
            customer_id=customer_id, board=board, pii=pii, synthesis=synthesis
        )

    # --- 8. No Action gate ---
    gate = check_no_action(synthesis=synthesis, judge=None, action_proposal=None)
    log.info(
        "stage.no_action_gate",
        is_no_action=gate.is_no_action,
        reason=gate.reason,
    )

    if gate.is_no_action:
        cp = writer.write(
            as_of_time=last_event_time,
            customer_id=customer_id,
            inferred_state=(
                synthesis.inferred_state
                if synthesis
                else InferredState.NO_SIGNIFICANT_EVENT
            ),
            confidence=synthesis.confidence if synthesis else 0.5,
            action=Action.NO_ACTION,
            action_subtype=None,
            hitl_status=HITLStatus.AUTO_APPROVED,
            notes=f"No Action gate fired: {gate.reason}",
            trace_id=pipeline_trace,
        )
        return {
            "customer_id": customer_id,
            "checkpoint": cp.model_dump(),
            "gate_reason": gate.reason,
        }

    # --- 9. Semantic policy lookup + episodic recall ---
    policy_query = " ".join(
        filter(
            None,
            [
                synthesis.customer_state if synthesis else "",
                life_event.life_phase if life_event else "",
                "retention offer compliance HITL",
            ],
        )
    )
    policy_hits = semantic.query(policy_query, k=4)
    episodic_hits = [
        {"summary": m.summary, "kind": m.kind, "score": round(s, 4)}
        for m, s in episodic.retrieve(
            customer_id,
            kinds=["risk_flag", "life_event", "intervention"],
            top_k=3,
            t_now=last_event_time,
        )
    ]

    # --- 10. Action + Critique loop ---
    assert synthesis is not None  # gate would have fired if it were None
    proposal, feedback = await _run_action_loop(
        customer_id=customer_id,
        board=board,
        pii=pii,
        synthesis=synthesis,
        life_event=life_event,
        judge_verdict=judge_verdict,
        policy_hits=policy_hits,
        episodic_hits=episodic_hits,
        computed_features=computed_features,
    )

    # --- 10b. Fallback if loop failed or was rejected ---
    loop_failed = proposal is None or (feedback is not None and feedback.verdict != "pass")

    if loop_failed:
        reason = (
            "no proposal returned"
            if proposal is None
            else f"critique verdict={feedback.verdict}"
        )
        log.warning("stage.action.fallback", reason=reason)
        proposal = _build_fallback_proposal(
            reason=reason, life_event=life_event, synthesis=synthesis
        )

    # --- 11. Guardrail override on final message draft ---
    guardrail_hit = False
    if proposal.message_draft:
        hs = scanner.scan_text(proposal.message_draft)
        if hs is not None:
            guardrail_hit = True
            log.warning(
                "stage.guardrail.message_hit",
                rule=hs.rule_id,
                matched=hs.matched_keywords,
            )
            proposal = ActionProposal(
                action=Action.COMPLIANCE_FRAUD_HOLD,
                action_subtype="guardrail_override",
                cost_estimate_usd=0.0,
                message_draft=None,
                eligibility_notes="Auto-overridden by guardrail.",
                supporting_evidence=proposal.supporting_evidence,
                reasoning=f"Guardrail rule '{hs.rule_id}' matched: {hs.reason}",
                confidence=0.95,
                iteration=proposal.iteration,
            )

    # --- 12. HITL decision ---
    needs_hitl, hitl_reason = _requires_hitl(
        proposal=proposal,
        synthesis=synthesis,
        life_event=life_event,
        guardrail_hit=guardrail_hit,
    )

    hitl_status = HITLStatus.AUTO_APPROVED
    final_action = proposal.action
    final_subtype = proposal.action_subtype

    if needs_hitl:
        if auto_hitl:
            log.info("stage.hitl.auto", reason=hitl_reason)
            hitl_status = HITLStatus.ESCALATED
        else:
            req = HITLRequest(
                request_id=hitl_queue.new_request_id(),
                customer_id=customer_id,
                action=proposal.action.value,
                action_subtype=proposal.action_subtype,
                cost_estimate_usd=proposal.cost_estimate_usd,
                message_draft=proposal.message_draft,
                reasoning=proposal.reasoning,
                confidence=proposal.confidence,
                evidence=proposal.supporting_evidence,
                synthesis_snapshot=synthesis.model_dump() if synthesis else {},
                critique_snapshot=feedback.model_dump() if feedback else {},
            )
            rid = hitl_queue.submit(req)
            decided = await hitl_queue.wait_for_decision(rid)
            if decided.decision == "approve":
                hitl_status = HITLStatus.HUMAN_APPROVED
            elif decided.decision == "reject":
                hitl_status = HITLStatus.HUMAN_REJECTED
                final_action = Action.NO_ACTION
                final_subtype = None
            elif decided.decision == "modify":
                hitl_status = HITLStatus.HUMAN_MODIFIED
                if decided.modified_action:
                    final_action = Action(
                        decided.modified_action.get("action", final_action.value)
                    )
                    final_subtype = decided.modified_action.get(
                        "action_subtype", final_subtype
                    )

    # --- 13. Write checkpoint ---
    cp = writer.write(
        as_of_time=last_event_time,
        customer_id=customer_id,
        inferred_state=(
            synthesis.inferred_state
            if synthesis
            else InferredState.NO_SIGNIFICANT_EVENT
        ),
        confidence=proposal.confidence,
        action=final_action,
        action_subtype=final_subtype,
        hitl_status=hitl_status,
        notes=f"{proposal.reasoning} | hitl={hitl_reason or 'not required'}",
        trace_id=pipeline_trace,
    )

    return {
        "customer_id": customer_id,
        "checkpoint": cp.model_dump(),
        "proposal": proposal.model_dump(),
        "critique": feedback.model_dump() if feedback else None,
        "judge_verdict": judge_verdict,
        "hitl_required": needs_hitl,
        "hitl_reason": hitl_reason,
    }