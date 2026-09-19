"""
End-to-end swarm smoke test.

Pipeline:
  1. Stream events → CEP
  2. Compute rolling features (passed to swarm agents as context)
  3. Swarm (parallel): Usage, Support, Transaction, KYC
  4. Agent-dependent trigger: Life-Event
     Fires when >= 2 agents have "flagged a life-event signal" — see
     agent_flagged_life_event_signal() for the exact rule per agent.
  5. Handoff: Synthesis
  6. If arbitration_needed: Debate (retention vs growth, parallel) → Judge
  7. Print final State Board snapshot

Run:
    python -m scripts.smoke_swarm data/raw/scenario_01
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from src.agents.judge_agent import JudgeAgent
from src.agents.kyc_agent import KYCAgent
from src.agents.life_event_agent import LifeEventAgent
from src.agents.debate_agent import GrowthDebateAgent, RetentionDebateAgent
from src.agents.support_agent import SupportAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.transaction_agent import TransactionAgent
from src.agents.usage_agent import UsageAgent
from src.guardrails.pii import PIITokenizer
from src.memory.state_board import StateBoard
from src.streaming.cep_engine import CEPEngine
from src.streaming.event_generator import stream_scenario
from src.streaming.features import extract_features
from src.utils.logger import get_logger
from src.utils.tracing import new_trace_id, set_trace_id

log = get_logger("smoke_swarm")

LIFE_EVENT_MIN_SIGNALS = 2


async def build_cep(scenario_dir: Path) -> tuple[CEPEngine, str]:
    cep = CEPEngine()
    customer_id: str | None = None
    n = 0
    async for ev in stream_scenario(
        scenario_dir, seed_speed=1e6, live_speed=1e6, jitter_prob=0.0
    ):
        set_trace_id(new_trace_id())
        cep.ingest(ev)
        customer_id = customer_id or ev.customer_id
        n += 1
    log.info("cep.ready", events=n, customer_id=customer_id)
    assert customer_id is not None
    return cep, customer_id


async def run_swarm(
    *,
    customer_id: str,
    events,
    computed_features: dict[str, Any],
    state_board: StateBoard,
    pii: PIITokenizer,
) -> None:
    context = {"computed_features": computed_features}
    agents = [
        UsageAgent(state_board=state_board, pii=pii),
        SupportAgent(state_board=state_board, pii=pii),
        TransactionAgent(state_board=state_board, pii=pii),
        KYCAgent(state_board=state_board, pii=pii),
    ]
    log.info("swarm.start", agents=[a.name for a in agents])

    async def _run(a):
        try:
            await a.run(customer_id=customer_id, events=events, extra_context=context)
        except Exception as e:
            log.error("swarm.agent_failed", agent=a.name, error=str(e))

    await asyncio.gather(*(_run(a) for a in agents))
    log.info("swarm.done")


# --- agent-dependent trigger ------------------------------------------------

def agent_flagged_life_event_signal(agent_name: str, findings: dict[str, Any]) -> bool:
    """
    Return True if this agent's findings constitute a life-event signal.

    This mirrors the PS definition: "the Life-Event Inference Agent only
    activates once both the [agents] have independently flagged something
    in the same time window." Flag = a meaningful, non-trivial finding.
    """
    if agent_name == "support_agent":
        # Any life-event hint, OR a high/CRITICAL urgency ticket, OR negative sentiment
        return bool(
            findings.get("life_event_hint")
            or findings.get("ticket_urgency") in ("HIGH", "CRITICAL")
            or (findings.get("sentiment_score") or 0) <= -0.5
        )
    if agent_name == "usage_agent":
        # Meaningful engagement decline
        return bool(
            findings.get("session_dropoff_flag") is True
            or (findings.get("login_trend_30d") or 0.0) < -0.15
        )
    if agent_name == "transaction_agent":
        # Anomalous financial behavior
        return bool(
            findings.get("large_transaction_flag") is True
            or findings.get("anomalous_pattern") is not None
            or (findings.get("failed_billing_count") or 0) > 0
            or (findings.get("spend_velocity_variance") or 0.0) > 0.5
        )
    if agent_name == "kyc_agent":
        # Any documented life change
        return bool(
            findings.get("address_changed")
            or findings.get("marital_status_change")
            or findings.get("dependents_change")
        )
    return False


def life_event_should_fire(
    state_board: StateBoard,
    customer_id: str,
    min_signals: int = LIFE_EVENT_MIN_SIGNALS,
) -> bool:
    slots = state_board.read_all(customer_id)
    flagged: list[str] = []
    for name, slot in slots.items():
        if agent_flagged_life_event_signal(name, slot.findings):
            flagged.append(name)
    log.info(
        "trigger.life_event_check",
        flagged_agents=flagged,
        signals=len(flagged),
        needed=min_signals,
        total_findings=len(slots),
    )
    return len(flagged) >= min_signals


# --- main pipeline ----------------------------------------------------------

async def main(scenario_dir: Path) -> None:
    state_board = StateBoard()
    pii = PIITokenizer()

    # 1. CEP
    cep, customer_id = await build_cep(scenario_dir)
    events = cep.window(customer_id, "30d")
    log.info("events.for_swarm", count=len(events))

    # 2. Compute authoritative features ONCE
    computed_features = extract_features(cep.all_windows(customer_id))
    log.info("features.computed", keys=list(computed_features.keys()))

    # 3. Swarm
    await run_swarm(
        customer_id=customer_id,
        events=events,
        computed_features=computed_features,
        state_board=state_board,
        pii=pii,
    )

    # 4. Life-Event (agent-dependent trigger)
    if life_event_should_fire(state_board, customer_id):
        log.info("life_event.firing")
        le = LifeEventAgent(state_board=state_board, pii=pii)
        try:
            await le.run(
                customer_id=customer_id,
                events=events,
                extra_context={
                    "state_board": state_board.snapshot_findings(customer_id),
                    "computed_features": computed_features,
                },
            )
        except Exception as e:
            log.error("life_event.failed", error=str(e))
    else:
        log.info("life_event.skipped")

    # 5. Synthesis
    synth = SynthesisAgent(state_board=state_board, pii=pii)
    synthesis_result = None
    try:
        synthesis_result = await synth.run(
            customer_id=customer_id,
            events=[],
            extra_context={"state_board": state_board.snapshot_findings(customer_id)},
        )
    except Exception as e:
        log.error("synthesis.failed", error=str(e))

    # 6. Debate + Judge (only if conflict)
    if synthesis_result is not None and getattr(synthesis_result, "arbitration_needed", False):
        log.info("debate.firing", conflict=synthesis_result.conflict_description)
        retention = RetentionDebateAgent(state_board=state_board, pii=pii)
        growth = GrowthDebateAgent(state_board=state_board, pii=pii)
        ctx = {"state_board": state_board.snapshot_findings(customer_id)}

        async def _safe(a, name):
            try:
                return await a.run(customer_id=customer_id, events=[], extra_context=ctx)
            except Exception as e:
                log.error("debate.agent_failed", agent=name, error=str(e))
                return None

        r_pos, g_pos = await asyncio.gather(
            _safe(retention, "debate_retention"),
            _safe(growth, "debate_growth"),
        )

        if r_pos is not None and g_pos is not None:
            judge = JudgeAgent(state_board=state_board, pii=pii)
            try:
                await judge.run(
                    customer_id=customer_id,
                    events=[],
                    extra_context={
                        "synthesis": synthesis_result.model_dump(),
                        "retention_position": r_pos.model_dump(),
                        "growth_position": g_pos.model_dump(),
                    },
                )
                log.info("judge.done")
            except Exception as e:
                log.error("judge.failed", error=str(e))
    else:
        log.info("debate.skipped", reason="no arbitration needed")

    # 7. Print
    print("\n" + "=" * 80)
    print("FINAL STATE BOARD SNAPSHOT —", customer_id)
    print("=" * 80)
    snapshot = state_board.snapshot_findings(customer_id)
    for agent_name, payload in snapshot.items():
        print(f"\n[{agent_name}]  confidence={payload.get('confidence')}")
        print(json.dumps(payload.get("findings"), indent=2, default=str))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/scenario_01")
    asyncio.run(main(d))