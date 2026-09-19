"""
Smoke test: streams scenario_01, seeds episodic memory, runs a decay-weighted
retrieval, seeds semantic memory, and does a policy lookup.

Run:
    python -m scripts.smoke_memory data/raw/scenario_01
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.guardrails.pii import PIITokenizer
from src.guardrails.rbac import rbac_for
from src.guardrails.scanner import GuardrailScanner
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory
from src.memory.state_board import StateBoard
from src.memory.working_memory import WorkingMemory
from src.streaming.event_generator import stream_scenario
from src.utils.logger import get_logger
from src.utils.tracing import new_trace_id, set_trace_id

log = get_logger("smoke_memory")


async def main(scenario_dir: Path) -> None:
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    board = StateBoard()
    scanner = GuardrailScanner()
    pii = PIITokenizer()

    customer_id = None
    hard_stops = []
    n_events = 0
    t_first = None
    t_last = None

    async for ev in stream_scenario(scenario_dir, seed_speed=1e6,
                                    live_speed=1e6, jitter_prob=0.0):
        set_trace_id(new_trace_id())
        customer_id = customer_id or ev.customer_id
        n_events += 1
        t_first = t_first or ev.event_time
        t_last = ev.event_time

        episodic.add_event(ev, trace_id=new_trace_id())

        hs = scanner.scan_event(ev)
        if hs is not None:
            hard_stops.append(hs.to_dict())

    log.info("stream.done", customer_id=customer_id, events=n_events,
             episodic_size=episodic.size(customer_id),
             hard_stops=len(hard_stops),
             t_first=t_first.isoformat() if t_first else None,
             t_last=t_last.isoformat() if t_last else None)

    # --- decay-weighted retrieval at the LAST event time -------------------
    top = episodic.retrieve(
        customer_id,
        kinds=["risk_flag", "life_event", "intervention"],
        top_k=5,
        t_now=t_last or datetime.now(timezone.utc),
    )
    log.info("episodic.top_k", count=len(top))
    for m, score in top:
        log.info("episodic.hit",
                 memory_id=m.memory_id, kind=m.kind, score=round(score, 4),
                 summary=m.summary)

    # --- semantic policy lookup -------------------------------------------
    hits = semantic.query("customer is in financial distress and asked for a payment plan", k=3)
    log.info("semantic.hits", count=len(hits))
    for h in hits:
        log.info("semantic.hit", id=h["id"], distance=h["distance"],
                 text=h["text"][:120])

    # --- RBAC demo --------------------------------------------------------
    usage = rbac_for("usage_agent")
    kyc = rbac_for("kyc_agent")
    log.info("rbac.usage.allowed", sources=sorted(usage.scopes))
    log.info("rbac.kyc.allowed", sources=sorted(kyc.scopes))

    # --- State Board demo -------------------------------------------------
    board.publish(
        customer_id=customer_id,
        agent="usage_agent",
        findings={"login_trend_30d": -0.29, "session_dropoff_flag": False},
        confidence=0.71,
    )
    board.publish(
        customer_id=customer_id,
        agent="support_agent",
        findings={"ticket_urgency": "HIGH", "churn_intent_flag": False},
        confidence=0.83,
    )
    log.info("state_board.snapshot",
             snapshot=board.snapshot_findings(customer_id))

    # --- Working memory demo ---------------------------------------------
    wm = WorkingMemory(customer_id=customer_id)
    wm.publish("usage_agent", {"login_trend_30d": -0.29})
    wm.publish("support_agent", {"ticket_urgency": "HIGH"})
    log.info("working_memory", **wm.to_dict())


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/scenario_01")
    asyncio.run(main(d))