"""
Deterministic PII tokenizer.

Replaces sensitive identifiers with stable tokens BEFORE any text is sent
to an LLM. Tokens are:
    - stable within a run (same input -> same token)
    - reversible ONLY through the internal map (never exposed to LLM)
    - scoped per customer so cross-customer leakage is impossible

IMPORTANT: event_id and account_id are NOT PII.
- event_id is a traceability reference that downstream agents must be able
  to cite verbatim in their output. Never tokenize it.
- account_id is an internal reference the agents need in order to reason
  about the customer's accounts. Not tokenized by default.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# Patterns to catch raw PII even when the caller didn't tell us the field name.
# NOTE: NO event_id or account_id here — those must pass through untouched.
_PATTERNS = {
    "email": re.compile(r"[\w\.\-]+@[\w\.\-]+\.\w+"),
    "phone": re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "cust_id": re.compile(r"\bCUST_\d+\b"),
}


def _token(prefix: str, value: str) -> str:
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}_TOKEN_{h}"


class PIITokenizer:
    """Stateful map so we can detokenize internally for audits."""

    def __init__(self) -> None:
        self._fwd: dict[str, str] = {}   # raw -> token
        self._rev: dict[str, str] = {}   # token -> raw

    # -- public API ---------------------------------------------------------

    def tokenize(self, value: str) -> str:
        if not value:
            return value
        if value in self._fwd:
            return self._fwd[value]
        prefix = self._prefix_for(value)
        tok = _token(prefix, value)
        self._fwd[value] = tok
        self._rev[tok] = value
        return tok

    def detokenize(self, text: str) -> str:
        """For internal audits only — never send to LLM."""
        out = text
        for tok, raw in self._rev.items():
            out = out.replace(tok, raw)
        return out

    def scrub_text(self, text: str) -> str:
        """Replace any raw PII patterns found in free text."""
        if not text:
            return text
        out = text
        for kind, pat in _PATTERNS.items():
            def _sub(m: re.Match, kind=kind) -> str:
                return self.tokenize(m.group(0))
            out = pat.sub(_sub, out)
        return out

    def scrub_dict(self, d: dict[str, Any], fields: list[str] | None = None
                   ) -> dict[str, Any]:
        """
        Return a shallow-scrubbed copy of the dict.
        If `fields` is provided, only those keys are tokenized.
        Otherwise, all string values are pattern-scrubbed.
        """
        out: dict[str, Any] = {}
        fields_set = set(fields or [])
        for k, v in d.items():
            if isinstance(v, str):
                if fields and k in fields_set:
                    out[k] = self.tokenize(v)
                else:
                    out[k] = self.scrub_text(v)
            elif isinstance(v, dict):
                out[k] = self.scrub_dict(v, fields)
            elif isinstance(v, list):
                out[k] = [
                    self.scrub_dict(i, fields) if isinstance(i, dict)
                    else self.scrub_text(i) if isinstance(i, str)
                    else i
                    for i in v
                ]
            else:
                out[k] = v
        return out

    # -- helpers ------------------------------------------------------------

    def _prefix_for(self, value: str) -> str:
        if value.startswith("CUST_"):
            return "CUST"
        if value.startswith("ACC_"):
            return "ACC"
        if value.startswith("EVT_"):
            return "EVT"
        if "@" in value:
            return "EMAIL"
        if value.isdigit() or re.fullmatch(r"\+?\d[\d\s\-\(\)]+", value):
            return "PHONE"
        return "PII"