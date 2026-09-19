"""
Complex Event Processing (CEP-lite).

Responsibilities:
- Maintain per-customer rolling windows keyed by event_time.
- Handle late / out-of-order arrivals: an event whose event_time falls
  inside an existing window is inserted at the correct position and
  the window is re-aggregated.
- Track ingestion lag per customer (PS: "correct handling of late events").

NOTE: This module uses `ingestion_time` (not `arrival_time`) — the field
name is dictated by the dataset schema in README_dataset_schema.md.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable

from src.streaming.event_generator import RawEvent
from src.streaming.windows import DEFAULT_WINDOWS, WindowSpec
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WindowState:
    spec: WindowSpec
    # sorted ascending by event_time; tuple is (event_time, event)
    events: list[tuple[datetime, RawEvent]] = field(default_factory=list)
    late_insertions: int = 0

    def add(self, ev: RawEvent) -> None:
        entry = (ev.event_time, ev)
        if self.events and ev.event_time < self.events[-1][0]:
            # Late arrival — insert at the correct position.
            times = [t for t, _ in self.events]
            idx = bisect.bisect_left(times, ev.event_time)
            self.events.insert(idx, entry)
            self.late_insertions += 1
        else:
            self.events.append(entry)

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.spec.seconds)
        i = 0
        for i, (t, _) in enumerate(self.events):
            if t >= cutoff:
                break
        else:
            i = len(self.events)
        if i:
            self.events = self.events[i:]

    def snapshot(self) -> list[RawEvent]:
        return [ev for _, ev in self.events]


class CEPEngine:
    """
    Per-customer window state.

        cep = CEPEngine()
        cep.ingest(event)
        feats = cep.compute(customer_id, extract_features)
    """

    def __init__(self, windows: Iterable[WindowSpec] | None = None):
        self.windows = list(windows or DEFAULT_WINDOWS)
        self._state: dict[str, dict[str, WindowState]] = defaultdict(
            lambda: {w.name: WindowState(w) for w in self.windows}
        )
        # Per-customer "now" = max ingestion_time observed so far.
        self._now: dict[str, datetime] = {}
        # Per-customer ingestion lag samples (seconds).
        self._lag_seconds: dict[str, list[float]] = defaultdict(list)

    def ingest(self, ev: RawEvent) -> None:
        cust_windows = self._state[ev.customer_id]
        for w in self.windows:
            cust_windows[w.name].add(ev)

        prev = self._now.get(ev.customer_id)
        if prev is None or ev.ingestion_time > prev:
            self._now[ev.customer_id] = ev.ingestion_time

        lag = (ev.ingestion_time - ev.event_time).total_seconds()
        if lag > 0:
            self._lag_seconds[ev.customer_id].append(lag)

        now = self._now[ev.customer_id]
        for w in self.windows:
            cust_windows[w.name].prune(now)

    def window(self, customer_id: str, name: str) -> list[RawEvent]:
        return self._state[customer_id][name].snapshot()

    def all_windows(self, customer_id: str) -> dict[str, list[RawEvent]]:
        return {n: self.window(customer_id, n) for n in self._state[customer_id]}

    def customers(self) -> list[str]:
        return list(self._state.keys())

    def lag_stats(self, customer_id: str) -> dict[str, float]:
        lags = self._lag_seconds.get(customer_id, [])
        if not lags:
            return {"count": 0, "max_s": 0.0, "mean_s": 0.0}
        return {
            "count": len(lags),
            "max_s": max(lags),
            "mean_s": sum(lags) / len(lags),
        }

    def late_insertions(self, customer_id: str) -> dict[str, int]:
        return {
            n: self._state[customer_id][n].late_insertions
            for n in self._state[customer_id]
        }

    def compute(
        self,
        customer_id: str,
        feature_fn: Callable[[dict[str, list[RawEvent]]], dict[str, Any]],
    ) -> dict[str, Any]:
        return feature_fn(self.all_windows(customer_id))