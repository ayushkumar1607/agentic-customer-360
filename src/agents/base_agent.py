"""
Base agent contract.

Every specialist agent inherits from BaseAgent. The framework guarantees:

- RBAC enforced before the agent can see any events (data-layer, not prompt)
- All text I/O is PII-scrubbed before it hits an LLM
- Every run is wrapped in a span with the agent's name, role, model, and tier
- Output is validated against a pydantic schema before being published
- The agent publishes STRUCTURED FINDINGS to the State Board, never its raw
  internal reasoning trace
- Model tier is resolved per-agent from config/agents.yaml

PROMPT STRUCTURE — deliberate ordering:
    The user prompt is a JSON object whose keys appear in this order:
      1. customer_id
      2. computed_features     (authoritative CEP-derived numbers — FIRST)
      3. state_board           (findings from other agents, if relevant)
      4. synthesis / debate_positions (task-specific context)
      5. events                (raw events — LAST, because they are bulky)
      6. other_context         (anything else)

    Small models pay most attention to the beginning of the prompt. Putting
    authoritative aggregates first prevents them from being ignored.
"""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ValidationError

from config.settings import settings
from src.guardrails.pii import PIITokenizer
from src.guardrails.rbac import RBAC, RBACViolation, rbac_for
from src.memory.state_board import StateBoard
from src.streaming.event_generator import RawEvent
from src.utils.llm_client import complete_json
from src.utils.logger import get_logger
from src.utils.tracing import span

log = get_logger(__name__)

AGENT_REGISTRY_PATH = Path("config/agents.yaml")


# ---- registry loader -------------------------------------------------------

_REGISTRY_CACHE: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        if AGENT_REGISTRY_PATH.exists():
            with AGENT_REGISTRY_PATH.open("r", encoding="utf-8") as f:
                _REGISTRY_CACHE = yaml.safe_load(f) or {}
        else:
            log.warning("agents.registry_missing", path=str(AGENT_REGISTRY_PATH))
            _REGISTRY_CACHE = {"agents": {}}
    return _REGISTRY_CACHE


def _model_id_for_tier(tier: str) -> str:
    return {
        "fast":   settings.groq_model_fast,
        "reason": settings.groq_model_reason,
        "alt":    settings.groq_model_alt,
    }.get(tier, settings.groq_model_fast)


# ---- base agent ------------------------------------------------------------

class BaseAgent(ABC):
    # --- subclasses must override these ------------------------------------
    name: ClassVar[str] = "base_agent"
    role: ClassVar[str] = "synthesis_agent"
    output_schema: ClassVar[type[BaseModel]] = BaseModel
    system_prompt: ClassVar[str] = ""

    # --- optional overrides ------------------------------------------------
    description: ClassVar[str] = ""
    temperature: ClassVar[float] = 0.1
    max_tokens: ClassVar[int] = 1200
    max_visible_events: ClassVar[int] = 200

    # extra_context keys that should be hoisted to the TOP of the user prompt
    priority_context_keys: ClassVar[tuple[str, ...]] = (
        "computed_features",
        "state_board",
        "synthesis",
        "retention_position",
        "growth_position",
    )

    # --- framework ---------------------------------------------------------

    def __init__(
        self,
        *,
        state_board: StateBoard,
        pii: PIITokenizer,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.state_board = state_board
        self.pii = pii

        try:
            self.rbac: RBAC = rbac_for(self.role)
        except RBACViolation as e:
            raise RuntimeError(
                f"agent '{self.name}' has invalid role '{self.role}': {e}"
            )

        self.registry_entry: dict[str, Any] = (
            _load_registry().get("agents", {}).get(self.name, {}) or {}
        )
        self.model: str = model or self._resolve_model_from_registry()
        self.model_tier: str = self.registry_entry.get("model", "fast")

        if temperature is not None:
            self.temperature_value = temperature
        elif "temperature" in self.registry_entry:
            self.temperature_value = float(self.registry_entry["temperature"])
        else:
            self.temperature_value = self.temperature

        log.info(
            "agent.init",
            agent=self.name,
            role=self.role,
            model=self.model,
            tier=self.model_tier,
            temperature=self.temperature_value,
            rbac_sources=sorted(self.rbac.scopes),
        )

    # --- public API --------------------------------------------------------

    async def run(
        self,
        *,
        customer_id: str,
        events: list[RawEvent],
        extra_context: dict[str, Any] | None = None,
    ) -> BaseModel:
        with span(
            f"agent.{self.name}.run",
            customer_id=customer_id,
            model=self.model,
            tier=self.model_tier,
        ) as s:
            visible = self.rbac.filter_events(events)
            s.attributes["visible_events"] = len(visible)
            s.attributes["total_events"] = len(events)
            if not visible and events:
                log.warning(
                    "agent.no_visible_events",
                    agent=self.name,
                    role=self.role,
                )

            user_prompt = self.build_user_prompt(
                customer_id=customer_id,
                events=visible,
                extra_context=extra_context or {},
            )
            user_prompt = self.pii.scrub_text(user_prompt)

            raw = await complete_json(
                system=self.system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature_value,
                max_tokens=self.max_tokens,
            )

            try:
                parsed = self.output_schema.model_validate(raw)
            except ValidationError as e:
                log.error(
                    "agent.schema_violation",
                    agent=self.name,
                    error=str(e),
                    raw_preview=json.dumps(raw)[:400],
                )
                raise

            self.state_board.publish(
                customer_id=customer_id,
                agent=self.name,
                findings=parsed.model_dump(),
                confidence=getattr(parsed, "confidence", None),
            )
            s.attributes["published"] = True
            s.attributes["confidence"] = getattr(parsed, "confidence", None)
            return parsed

    # --- prompt scaffolding -------------------------------------------------

    def build_user_prompt(
        self,
        *,
        customer_id: str,
        events: list[RawEvent],
        extra_context: dict[str, Any],
    ) -> str:
        """
        Build the user prompt with priority fields FIRST.

        See module docstring for the deliberate key ordering.
        """
        prompt: dict[str, Any] = {
            "customer_id": self.pii.tokenize(customer_id),
        }

        # Hoist priority context keys to the top level, in declared order
        remaining: dict[str, Any] = dict(extra_context)
        for key in self.priority_context_keys:
            if key in remaining:
                prompt[key] = remaining.pop(key)

        # Any leftover context goes under a clearly-named key
        if remaining:
            prompt["other_context"] = remaining

        # Raw events LAST (they are the bulkiest part)
        if self.max_visible_events > 0:
            prompt["events"] = [
                {
                    "event_id": ev.event_id,
                    "event_time": ev.event_time.isoformat(),
                    "source_system": ev.source_system,
                    "event_type": ev.event_type,
                    "account_id": ev.account_id,
                    "payload": ev.payload,
                }
                for ev in events[-self.max_visible_events:]
            ]

        return json.dumps(prompt, default=str)

    # --- model tier resolution ---------------------------------------------

    def _resolve_model_from_registry(self) -> str:
        tier = self.registry_entry.get("model", "fast")
        return _model_id_for_tier(tier)

    # --- diagnostics -------------------------------------------------------

    def describe(self) -> str:
        return (
            f"{self.name} [role={self.role}, tier={self.model_tier}]"
            f" — {self.description or 'no description provided'}"
        )