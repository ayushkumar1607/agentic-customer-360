"""
Async replay of the provided event stream.

Schema reality (from README_dataset_schema.md):
    event_id, event_time, ingestion_time, customer_id, account_id,
    source_system, event_type, schema_version, payload

Design notes:
- event_time     = authoritative time (used for windowing / CEP)
- ingestion_time = arrival time (used to detect late / out-of-order)
- We preserve source_system and account_id because agents need them.
- Supports: single JSONL file OR a scenario directory (loads history_seed
  first, then live_stream).
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---- event dataclass -------------------------------------------------------

@dataclass
class RawEvent:
    """Normalized event envelope matching the dataset schema."""
    event_id: str
    customer_id: str
    account_id: str | None
    source_system: str
    event_type: str
    schema_version: str
    event_time: datetime          # authoritative
    ingestion_time: datetime      # arrival
    payload: dict = field(default_factory=dict)

    @property
    def is_late(self) -> bool:
        return self.ingestion_time > self.event_time

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "customer_id": self.customer_id,
            "account_id": self.account_id,
            "source_system": self.source_system,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "event_time": self.event_time.isoformat(),
            "ingestion_time": self.ingestion_time.isoformat(),
            "payload": self.payload,
        }


# ---- parsing helpers -------------------------------------------------------

def _parse_ts(v: str | int | float | None) -> datetime:
    if v is None:
        return datetime.now(timezone.utc)
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    s = str(v).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def _normalize(raw: dict, idx: int) -> RawEvent | None:
    """
    Normalize one raw JSON line into a RawEvent.

    Strictly follows the dataset schema; falls back to common aliases so
    the pipeline never crashes on a single bad line.
    """
    try:
        customer_id = raw.get("customer_id") or raw.get("customerId")
        if not customer_id:
            return None

        event_time = _parse_ts(raw.get("event_time") or raw.get("timestamp"))
        ingestion_time = _parse_ts(
            raw.get("ingestion_time") or raw.get("event_time") or raw.get("timestamp")
        )

        return RawEvent(
            event_id=str(raw.get("event_id") or f"evt-{idx}"),
            customer_id=str(customer_id),
            account_id=raw.get("account_id"),
            source_system=str(raw.get("source_system") or "unknown"),
            event_type=str(raw.get("event_type") or "unknown"),
            schema_version=str(raw.get("schema_version") or "1.0"),
            event_time=event_time,
            ingestion_time=ingestion_time,
            payload=raw.get("payload") or {},
        )
    except Exception as e:
        log.warning("event.normalize_failed", idx=idx, error=str(e))
        return None


def _load_jsonl_lines(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("event.bad_json", line=line[:120])


# ---- streaming API ---------------------------------------------------------

async def stream_events(
    path: Path,
    *,
    speed: float = 1.0,
    jitter_prob: float = 0.0,
    max_events: int | None = None,
) -> AsyncIterator[RawEvent]:
    """
    Yield RawEvents from a JSONL file in ingestion order.

    `speed` accelerates replay. `jitter_prob` randomly swaps adjacent
    events to simulate out-of-order arrival (PS requires this).
    """
    if not path.exists():
        raise FileNotFoundError(f"Event stream not found: {path}")

    buf: list[RawEvent] = []
    idx = 0
    emitted = 0

    for raw in _load_jsonl_lines(path):
        ev = _normalize(raw, idx)
        idx += 1
        if ev is None:
            continue

        # Out-of-order injection
        if jitter_prob > 0 and buf and random.random() < jitter_prob:
            buf.insert(0, ev)
        else:
            buf.append(ev)

        while buf:
            next_ev = buf.pop(0)
            if buf:
                gap = (buf[0].ingestion_time - next_ev.ingestion_time).total_seconds()
                if gap > 0:
                    await asyncio.sleep(min(gap / speed, 2.0))
            yield next_ev
            emitted += 1
            if max_events and emitted >= max_events:
                return

    for ev in buf:
        yield ev


async def load_all(path: Path) -> list[RawEvent]:
    """Load entire file into memory (tests, seeding)."""
    out: list[RawEvent] = []
    async for ev in stream_events(path, speed=1e9):
        out.append(ev)
    return out


# ---- scenario-aware loading ------------------------------------------------

def resolve_scenario_dir(arg: str | Path) -> Path | None:
    """If `arg` is a scenario dir (contains live_stream.jsonl), return it."""
    p = Path(arg)
    if p.is_dir() and (p / "live_stream.jsonl").exists():
        return p
    return None


def resolve_stream_path(arg: str | Path) -> Path:
    """Return the JSONL file path (accepts dir or file)."""
    p = Path(arg)
    if p.is_dir():
        live = p / "live_stream.jsonl"
        if live.exists():
            return live
        jsons = sorted(p.glob("*.jsonl"))
        if jsons:
            return jsons[0]
    return p


async def stream_scenario(
    scenario_dir: Path,
    *,
    seed_history: bool = True,
    seed_speed: float = 1e6,
    live_speed: float = 1000.0,
    jitter_prob: float = 0.02,
    max_live_events: int | None = None,
) -> AsyncIterator[RawEvent]:
    """
    Stream a full scenario: history_seed first (fast), then live_stream.

    Emits RawEvents in order. The caller is expected to inspect
    `ev.ingestion_time` and treat the switch point as "live starts now".
    """
    if seed_history:
        hist = scenario_dir / "history_seed.jsonl"
        if hist.exists():
            log.info("stream.history_seed.start", path=str(hist))
            n = 0
            async for ev in stream_events(hist, speed=seed_speed):
                n += 1
                yield ev
            log.info("stream.history_seed.done", events=n)

    live = scenario_dir / "live_stream.jsonl"
    if not live.exists():
        raise FileNotFoundError(f"live_stream.jsonl missing in {scenario_dir}")

    log.info("stream.live.start", path=str(live))
    async for ev in stream_events(
        live, speed=live_speed, jitter_prob=jitter_prob, max_events=max_live_events
    ):
        yield ev
    log.info("stream.live.done")