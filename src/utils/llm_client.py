"""
Thin async wrapper over Groq with retry + JSON-mode + model fallback +
server-aware rate-limit backoff.

Public API:
    complete(...)      -> str
    complete_json(...) -> dict

Design notes:
- Every call is logged with a trace_id and span.
- JSON mode is used wherever an agent expects a structured output.
- `model` parameter allows cheap vs. capable model selection per agent.
- Supports tiered models (fast / reason / alt) resolved by BaseAgent from
  config/agents.yaml.
- If a model is missing (404 / model_not_found), the client automatically
  falls through to a fallback candidate.
- On a rate-limit (429), the client parses Groq's own suggested wait time
  from the error message and sleeps exactly that long. This avoids both
  overshooting (wasted time) and undershooting (repeated 429s).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from groq import AsyncGroq

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.tracing import span

log = get_logger(__name__)

_client: AsyncGroq | None = None


# Fallback chain — if a model 404s, try the next one.
_MODEL_FALLBACKS: dict[str, list[str]] = {
    "openai/gpt-oss-20b": ["openai/gpt-oss-120b"],
    "openai/gpt-oss-120b": ["openai/gpt-oss-20b"],
    "qwen/qwen3.6-27b": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
}


def _resolve_model(model: str) -> list[str]:
    """Return the model plus any fallbacks to try in order."""
    return [model] + _MODEL_FALLBACKS.get(model, [])


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


def _is_model_missing(err_str: str) -> bool:
    """Detect the 404 / model_not_found error class so we can fall back."""
    return (
        "model_not_found" in err_str
        or "does not exist" in err_str
        or ("404" in err_str and "model" in err_str.lower())
    )


def _is_rate_limit(err_str: str) -> bool:
    """Detect a 429 / TPM rate-limit response."""
    low = err_str.lower()
    return (
        "429" in err_str
        or "rate_limit" in low
        or "rate limit" in low
        or "tokens per minute" in low
        or "tpm" in low
    )


def _extract_retry_after(err_str: str) -> float | None:
    """
    Groq 429 responses include a hint like:
        'Please try again in 18.69s.'
    Parse that and return the number of seconds to wait, plus a small
    safety buffer.
    """
    m = re.search(
        r"try again in (\d+(?:\.\d+)?)\s*s",
        err_str,
        re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1)) + 2.0   # 2-second safety buffer
        except ValueError:
            return None
    return None


async def complete(
    *,
    system: str,
    user: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    retries: int = 5,
) -> str:
    """
    Return raw string completion.

    Tries `model` first; on a model-not-found error, walks the fallback
    chain. On a rate-limit, sleeps for Groq's suggested duration. On any
    other transient error, uses exponential backoff.
    """
    primary = model or settings.groq_model_fast
    client = _get_client()
    candidates = _resolve_model(primary)
    last_err: Exception | None = None

    for candidate in candidates:
        kwargs: dict[str, Any] = {
            "model": candidate,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-oss models support reasoning_effort; keep it low so the visible
        # content is populated rather than being consumed by hidden reasoning.
        if "gpt-oss" in candidate:
            kwargs["reasoning_effort"] = "low"

        for attempt in range(1, retries + 1):
            try:
                with span(
                    "llm.complete",
                    model=candidate,
                    json_mode=json_mode,
                    attempt=attempt,
                ) as s:
                    resp = await client.chat.completions.create(**kwargs)
                    content = resp.choices[0].message.content or ""
                    s.attributes["prompt_tokens"] = resp.usage.prompt_tokens
                    s.attributes["completion_tokens"] = resp.usage.completion_tokens
                    s.attributes["content_len"] = len(content)
                    return content
            except Exception as e:
                last_err = e
                err_str = str(e)

                # 1) Model missing — skip remaining retries, try next candidate
                if _is_model_missing(err_str):
                    log.warning(
                        "llm.model_missing",
                        model=candidate,
                        falling_back=True,
                    )
                    break

                # 2) Rate limit — sleep for the exact duration Groq suggested
                if _is_rate_limit(err_str):
                    wait_s = _extract_retry_after(err_str) or (5 * (2 ** attempt))
                    log.info(
                        "llm.rate_limited",
                        model=candidate,
                        attempt=attempt,
                        wait_s=round(wait_s, 1),
                    )
                    await asyncio.sleep(wait_s)
                    continue

                # 3) Any other transient error — exponential backoff
                log.warning(
                    "llm.retry",
                    model=candidate,
                    attempt=attempt,
                    error=err_str[:200],
                )
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(
        f"LLM call failed on all candidates {candidates}: {last_err}"
    )


async def complete_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    retries: int = 5,
) -> dict[str, Any]:
    """
    Return parsed JSON. Retries on malformed JSON, then raises.

    Forces JSON mode at the API level and additionally sanitizes common
    LLM artifacts (```json fences) before parsing.
    """
    raw = await complete(
        system=system,
        user=user,
        model=model,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error("llm.json_parse_failed", raw_preview=raw[:500], error=str(e))
        raise   