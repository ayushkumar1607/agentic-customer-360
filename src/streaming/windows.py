"""Rolling window definitions used by the CEP engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WindowUnit(str, Enum):
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"
    DAYS = "d"


_UNIT_SECONDS = {
    WindowUnit.SECONDS: 1,
    WindowUnit.MINUTES: 60,
    WindowUnit.HOURS: 3600,
    WindowUnit.DAYS: 86400,
}


@dataclass(frozen=True)
class WindowSpec:
    name: str
    size: int
    unit: WindowUnit

    @property
    def seconds(self) -> int:
        return self.size * _UNIT_SECONDS[self.unit]


# The windows we maintain per customer. Chosen to match the feature set
# committed to in the mid-term architecture.
DEFAULT_WINDOWS: list[WindowSpec] = [
    WindowSpec("1h", 1, WindowUnit.HOURS),
    WindowSpec("24h", 24, WindowUnit.HOURS),
    WindowSpec("7d", 7, WindowUnit.DAYS),
    WindowSpec("30d", 30, WindowUnit.DAYS),
]