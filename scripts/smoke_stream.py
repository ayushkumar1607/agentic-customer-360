"""
Run:
    python -m scripts.smoke_stream data/raw/scenario_01
    python -m scripts.smoke_stream data/raw/scenario_01/live_stream.jsonl
    python -m scripts.smoke_stream data/raw/sample_stream.jsonl
"""
import asyncio
import json
import sys
from pathlib import Path

from src.streaming.cep_engine import CEPEngine
from src.streaming.event_generator import (
    resolve_scenario_dir,
    resolve_stream_path,
    stream_events,
    stream_scenario,
)
from src.streaming.features import extract_features
from src.utils.logger import get_logger
from src.utils.tracing import new_trace_id, set_trace_id

log = get_logger("smoke")


async def dump_first_n(path: Path, n: int = 5) -> None:
    print(f"\n=== First {n} raw lines of {path} ===")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                obj = json.loads(line)
                print(f"[{i}] keys={list(obj.keys())}")
                print(f"     {json.dumps(obj, indent=2)[:600]}")
            except Exception as e:
                print(f"[{i}] RAW: {line[:200]}  (err={e})")


async def run_scenario(scenario_dir: Path, live_speed: float) -> None:
    live = scenario_dir / "live_stream.jsonl"
    await dump_first_n(live, 3)

    cep = CEPEngine()
    counts: dict[str, int] = {}
    n_live = 0
    seeded = 0

    async for ev in stream_scenario(
        scenario_dir,
        seed_history=True,
        seed_speed=1e6,
        live_speed=live_speed,
        jitter_prob=0.05,
    ):
        set_trace_id(new_trace_id())
        cep.ingest(ev)
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        if ev.event_id.startswith("EVT_"):
            # crude heuristic; history ids often differ, adjust if needed
            pass

    # Count history vs live based on where the live file's first event_time sits
    # Simpler: count both files independently.
    hist = scenario_dir / "history_seed.jsonl"
    if hist.exists():
        async for _ in stream_events(hist, speed=1e9):
            seeded += 1

    n_live = sum(counts.values()) - seeded if seeded else sum(counts.values())

    log.info("stream.done", customers=len(cep.customers()),
             total=sum(counts.values()), history_seeded=seeded, live_events=n_live)
    log.info("event_types", counts=counts)

    for cid in cep.customers()[:3]:
        feats = cep.compute(cid, extract_features)
        log.info("features.sample", customer_id=cid, **feats)
        log.info("lag_stats", customer_id=cid, **cep.lag_stats(cid))
        log.info("late_insertions", customer_id=cid, **cep.late_insertions(cid))


async def run_single_file(path: Path, speed: float) -> None:
    await dump_first_n(path, 3)
    cep = CEPEngine()
    counts: dict[str, int] = {}
    async for ev in stream_events(path, speed=speed, jitter_prob=0.05):
        set_trace_id(new_trace_id())
        cep.ingest(ev)
        counts[ev.event_type] = counts.get(ev.event_type, 0) + 1

    log.info("stream.done", customers=len(cep.customers()),
             total=sum(counts.values()))
    log.info("event_types", counts=counts)

    for cid in cep.customers()[:3]:
        feats = cep.compute(cid, extract_features)
        log.info("features.sample", customer_id=cid, **feats)


async def main(arg: str, speed: float) -> None:
    scenario_dir = resolve_scenario_dir(arg)
    if scenario_dir is not None:
        await run_scenario(scenario_dir, live_speed=speed)
    else:
        await run_single_file(resolve_stream_path(arg), speed=speed)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "data/raw/scenario_01"
    sp = float(sys.argv[2]) if len(sys.argv) > 2 else 5000.0
    asyncio.run(main(a, sp))