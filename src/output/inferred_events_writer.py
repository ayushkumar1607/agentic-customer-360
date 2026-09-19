"""
Writes the inferred-events.jsonl file in the exact README schema.

Each line is a Checkpoint object. Also provides a helper to read them back
as a JSON array (the format shown in the README).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config.settings import settings
from src.output.schema import (
    Action,
    Checkpoint,
    ConfidenceBand,
    HITLStatus,
    InferredState,
    confidence_to_band,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class InferredEventsWriter:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.inferred_events_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # truncate at start of a run
        self.path.write_text("", encoding="utf-8")
        log.info("inferred_writer.ready", path=str(self.path))

    def write(
        self,
        *,
        as_of_time: datetime | None = None,
        customer_id: str,
        inferred_state: InferredState,
        confidence: float,
        action: Action,
        action_subtype: str | None = None,
        hitl_status: HITLStatus = HITLStatus.AUTO_APPROVED,
        notes: str | None = None,
        trace_id: str | None = None,
    ) -> Checkpoint:
        ts = (as_of_time or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
        cp = Checkpoint(
            as_of_time=ts,
            inferred_state=inferred_state,
            confidence_band=confidence_to_band(confidence),
            action=action,
            action_subtype=action_subtype,
            hitl_status=hitl_status,
            notes=notes,
        )
        record = cp.model_dump()
        record["customer_id"] = customer_id        # informational, ignored by scorer
        if trace_id:
            record["trace_id"] = trace_id
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        log.info("inferred_writer.wrote", **{
            k: v for k, v in record.items() if k in
            ("as_of_time", "inferred_state", "confidence_band", "action", "hitl_status")
        })
        return cp

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_array(self, out_path: Path | None = None) -> Path:
        """Write the same checkpoints as a JSON array (README format)."""
        arr_path = Path(out_path or self.path.with_suffix(".json"))
        arr = self.read_all()
        arr_path.write_text(json.dumps(arr, indent=2), encoding="utf-8")
        return arr_path