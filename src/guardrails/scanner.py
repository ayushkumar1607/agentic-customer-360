"""
Deterministic hard-stop scanner.

Loaded from config/guardrails.yaml. Runs BEFORE any agentic reasoning.
If a rule fires, the pipeline is diverted and no normal action is generated.

This is intentionally rule-based — an LLM cannot reason around it.
PS (page 3, 9): hard-coded, deterministic rules that override agent autonomy.

Path resolution:
    Config path is anchored to the PROJECT ROOT (derived from this file's
    location), NOT to the current working directory. This makes the scanner
    work identically whether imported from pytest, a script, or a notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.streaming.event_generator import RawEvent
from src.utils.logger import get_logger
from src.utils.tracing import span

log = get_logger(__name__)

# src/guardrails/scanner.py -> parents[0]=guardrails, [1]=src, [2]=project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _PROJECT_ROOT / "config" / "guardrails.yaml"


@dataclass
class HardStop:
    rule_id: str
    route: str
    terminal_action: str
    hitl_required: bool
    reason: str
    matched_keywords: list[str] = field(default_factory=list)
    matched_field: str | None = None
    source_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "route": self.route,
            "terminal_action": self.terminal_action,
            "hitl_required": self.hitl_required,
            "reason": self.reason,
            "matched_keywords": self.matched_keywords,
            "matched_field": self.matched_field,
            "source_event_id": self.source_event_id,
        }


class GuardrailScanner:
    def __init__(self, config_path: Path | None = None):
        config_path = Path(config_path or CONFIG_PATH)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Guardrails config not found at {config_path}. "
                f"Expected at project root: config/guardrails.yaml"
            )

        with config_path.open("r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f) or {}

        self._rules: dict[str, dict] = self._cfg.get("hard_stops", {}) or {}
        self._soft: dict[str, list[str]] = self._cfg.get("soft_signals", {}) or {}

        log.info(
            "guardrails.loaded",
            path=str(config_path),
            top_level_keys=list(self._cfg.keys()),
            hard_stop_rules=list(self._rules.keys()),
            soft_signal_groups=list(self._soft.keys()),
        )

        # Fail LOUDLY if the file exists but contains no usable rules.
        # Silent empty rules are the worst possible failure mode: guardrails
        # appear to work but never fire.
        if not self._rules:
            raise RuntimeError(
                f"Guardrails config at {config_path} has no 'hard_stops' entries. "
                f"Top-level keys found: {list(self._cfg.keys())}. "
                f"Check the YAML structure — 'hard_stops' must be a top-level "
                f"mapping with named rules beneath it."
            )

    # -- public API ---------------------------------------------------------

    def scan_event(self, ev: RawEvent) -> HardStop | None:
        """Return the first hard-stop that fires on this event, or None."""
        with span(
            "guardrail.scan_event",
            event_id=ev.event_id,
            event_type=ev.event_type,
        ) as s:
            for rule_id, rule in self._rules.items():
                hit = self._match_rule(rule_id, rule, ev)
                if hit is not None:
                    s.attributes["fired_rule"] = rule_id
                    # NOTE: hit.to_dict() already contains 'rule_id', so we
                    # must NOT pass rule_id= as an extra kwarg — that caused
                    # a duplicate-key TypeError in structlog.
                    log.warning("guardrail.hard_stop", **hit.to_dict())
                    return hit
        return None

    def scan_text(
        self, text: str, source_event_id: str | None = None
    ) -> HardStop | None:
        """Scan a bare string (useful for LLM outputs, drafts, messages)."""
        if not text:
            return None
        low = text.lower()
        for rule_id, rule in self._rules.items():
            for kw in rule.get("match_any", []):
                if kw.lower() in low:
                    return HardStop(
                        rule_id=rule_id,
                        route=rule["route"],
                        terminal_action=rule["terminal_action"],
                        hitl_required=bool(rule.get("hitl_required", True)),
                        reason=rule.get("reason", rule_id),
                        matched_keywords=[kw],
                        matched_field="text",
                        source_event_id=source_event_id,
                    )
        return None

    def soft_signals(self, text: str) -> dict[str, list[str]]:
        """Return which soft-signal groups match. Does NOT halt the pipeline."""
        if not text:
            return {}
        low = text.lower()
        out: dict[str, list[str]] = {}
        for group, kws in self._soft.items():
            hits = [k for k in kws if k.lower() in low]
            if hits:
                out[group] = hits
        return out

    def pii_fields(self) -> Iterable[str]:
        return self._cfg.get("pii_fields", []) or []

    # -- internals ----------------------------------------------------------

    def _match_rule(
        self, rule_id: str, rule: dict, ev: RawEvent
    ) -> HardStop | None:
        kws = rule.get("match_any", []) or []
        fields = rule.get("scan_fields", []) or []
        payload = ev.payload or {}

        # 1) Scan declared payload fields
        for field_name in fields:
            raw_val = payload.get(field_name)
            if raw_val is None:
                continue
            text = str(raw_val).lower()
            for kw in kws:
                if kw.lower() in text:
                    return HardStop(
                        rule_id=rule_id,
                        route=rule["route"],
                        terminal_action=rule["terminal_action"],
                        hitl_required=bool(rule.get("hitl_required", True)),
                        reason=rule.get("reason", ""),
                        matched_keywords=[kw],
                        matched_field=field_name,
                        source_event_id=ev.event_id,
                    )

        # 2) Match the event_type itself (cheap, occasionally useful)
        et = (ev.event_type or "").lower()
        for kw in kws:
            if kw.lower() == et:
                return HardStop(
                    rule_id=rule_id,
                    route=rule["route"],
                    terminal_action=rule["terminal_action"],
                    hitl_required=bool(rule.get("hitl_required", True)),
                    reason=rule.get("reason", ""),
                    matched_keywords=[kw],
                    matched_field="event_type",
                    source_event_id=ev.event_id,
                )
        return None