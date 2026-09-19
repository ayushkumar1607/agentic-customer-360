"""
Metric primitives for the scoring harness.

Kept deliberately small and dependency-free so the harness runs anywhere.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ConfusionMatrix:
    """Per-class TP / FP / FN counts for a single categorical field."""
    field_name: str
    tp: Counter
    fp: Counter
    fn: Counter
    total: int

    def precision(self, cls: str) -> float:
        t = self.tp.get(cls, 0)
        f = self.fp.get(cls, 0)
        return t / (t + f) if (t + f) > 0 else 0.0

    def recall(self, cls: str) -> float:
        t = self.tp.get(cls, 0)
        f = self.fn.get(cls, 0)
        return t / (t + f) if (t + f) > 0 else 0.0

    def f1(self, cls: str) -> float:
        p, r = self.precision(cls), self.recall(cls)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def accuracy(self) -> float:
        correct = sum(self.tp.values())
        return correct / self.total if self.total > 0 else 0.0

    def macro_f1(self) -> float:
        classes = set(self.tp) | set(self.fp) | set(self.fn)
        if not classes:
            return 0.0
        return sum(self.f1(c) for c in classes) / len(classes)

    def all_classes(self) -> list[str]:
        return sorted(set(self.tp) | set(self.fp) | set(self.fn))


def confusion(
    *,
    field_name: str,
    expected: Sequence[str | None],
    predicted: Sequence[str | None],
) -> ConfusionMatrix:
    """Build a confusion matrix from parallel lists."""
    assert len(expected) == len(predicted), "length mismatch"
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for e, p in zip(expected, predicted):
        e_s = e if e is not None else "__null__"
        p_s = p if p is not None else "__null__"
        if e_s == p_s:
            tp[e_s] += 1
        else:
            fp[p_s] += 1
            fn[e_s] += 1
    return ConfusionMatrix(
        field_name=field_name,
        tp=tp, fp=fp, fn=fn,
        total=len(expected),
    )


@dataclass
class ConfidenceCalibration:
    """How well do our confidence_bands align with correctness?"""
    high_total: int
    high_correct: int
    medium_total: int
    medium_correct: int
    low_total: int
    low_correct: int

    def accuracy_in_band(self, band: str) -> float:
        total = getattr(self, f"{band}_total")
        correct = getattr(self, f"{band}_correct")
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "high": {
                "n": self.high_total,
                "acc": round(self.accuracy_in_band("high"), 3),
            },
            "medium": {
                "n": self.medium_total,
                "acc": round(self.accuracy_in_band("medium"), 3),
            },
            "low": {
                "n": self.low_total,
                "acc": round(self.accuracy_in_band("low"), 3),
            },
        }