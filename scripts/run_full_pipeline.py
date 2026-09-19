"""
Run the full end-to-end pipeline on one or more scenarios.

Usage:
    python -m scripts.run_full_pipeline data/raw/scenario_01
    python -m scripts.run_full_pipeline data/raw/scenario_01 data/raw/scenario_02
    python -m scripts.run_full_pipeline --all
    python -m scripts.run_full_pipeline data/raw/scenario_01 --hitl-live

The --hitl-live flag disables auto-approval, so the pipeline will block on
each HITL request. Use `python -m scripts.hitl_cli` in another terminal to
approve/reject.

By default the pipeline runs in auto-approve mode (fast end-to-end demo).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from config.settings import settings
from src.hitl.approval_queue import HITLQueue
from src.orchestration.pipeline import run_pipeline
from src.output.inferred_events_writer import InferredEventsWriter
from src.utils.logger import get_logger

log = get_logger("run_full_pipeline")


async def run_one(scenario_dir: Path, hitl_queue: HITLQueue,
                  writer: InferredEventsWriter, auto_hitl: bool) -> dict:
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario_dir}")
    print(f"{'=' * 80}")
    result = await run_pipeline(
        scenario_dir=scenario_dir,
        hitl_queue=hitl_queue,
        writer=writer,
        auto_hitl=auto_hitl,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", type=Path,
                    help="One or more scenario directories.")
    ap.add_argument("--all", action="store_true",
                    help="Run every scenario_* under data/raw/")
    ap.add_argument("--hitl-live", action="store_true",
                    help="Disable auto-approval; block on real HITL.")
    args = ap.parse_args()

    if args.all:
        scenarios = sorted(Path("data/raw").glob("scenario_*"))
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = [Path("data/raw/scenario_01")]

    auto_hitl = not args.hitl_live
    hitl_queue = HITLQueue(auto_approve=auto_hitl)
    writer = InferredEventsWriter()

    log.info("pipeline.batch_start", scenarios=[str(s) for s in scenarios],
             auto_hitl=auto_hitl)

    results = []
    for s in scenarios:
        try:
            results.append(await run_one(s, hitl_queue, writer, auto_hitl))
        except Exception as e:
            log.error("pipeline.scenario_failed", scenario=str(s), error=str(e))

    # Write the array version too
    arr_path = writer.write_array()
    log.info("pipeline.batch_done",
             scenarios=len(scenarios),
             checkpoints=len(writer.read_all()),
             jsonl=str(writer.path),
             json_array=str(arr_path))


if __name__ == "__main__":
    asyncio.run(main())