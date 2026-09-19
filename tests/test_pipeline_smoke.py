"""
End-to-end smoke test for the pipeline against a real scenario.

Marked `integration` — skip on systems without an API key.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.hitl.approval_queue import HITLQueue
from src.output.inferred_events_writer import InferredEventsWriter


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM integration test",
)
@pytest.mark.asyncio
async def test_pipeline_runs_scenario_01(scenario_dir: Path, tmp_path: Path):
    from src.orchestration.pipeline import run_pipeline

    writer = InferredEventsWriter(path=tmp_path / "inferred-events.jsonl")
    hitl = HITLQueue(auto_approve=True)

    result = await run_pipeline(
        scenario_dir=scenario_dir,
        hitl_queue=hitl,
        writer=writer,
        auto_hitl=True,
    )

    # Basic assertions — exact values depend on LLM behaviour
    assert "checkpoint" in result
    cp = result["checkpoint"]
    assert cp["inferred_state"] in {
        "medical_hardship", "financial_distress_general", "no_significant_event",
    }
    assert cp["action"] in {
        "no_action", "relationship_manager_escalation",
        "support_intervention", "proactive_retention_outreach",
    }
    # The writer should have produced at least one line
    assert writer.path.exists()
    assert writer.path.stat().st_size > 0