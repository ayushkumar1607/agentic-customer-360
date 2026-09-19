"""
Decay-weighted retrieval activation.

    A(m, t) = W_base * exp(-lambda * (t - t_event)) * (1 + R_reinforce)

Where:
    W_base       base importance of the memory (0..1)
    lambda       decay rate (larger = forget faster)
    t - t_event  age of the memory in DAYS
    R_reinforce  reinforcement from recurring evidence
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from config.settings import settings


def age_days(t_event: datetime, t_now: datetime | None = None) -> float:
    if t_now is None:
        t_now = datetime.now(timezone.utc)
    if t_event.tzinfo is None:
        t_event = t_event.replace(tzinfo=timezone.utc)
    if t_now.tzinfo is None:
        t_now = t_now.replace(tzinfo=timezone.utc)
    return max((t_now - t_event).total_seconds() / 86400.0, 0.0)


def activation(
    *,
    w_base: float,
    t_event: datetime,
    r_reinforce: float = 0.0,
    decay_lambda: float | None = None,
    t_now: datetime | None = None,
) -> float:
    lam = settings.memory_decay_lambda if decay_lambda is None else decay_lambda
    age = age_days(t_event, t_now)
    return w_base * math.exp(-lam * age) * (1.0 + r_reinforce)