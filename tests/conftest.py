"""
Shared pytest fixtures.

This file MUST live at tests/conftest.py. If pytest reports
"fixture 'make_event' not found", it means this file is missing or mislocated.

pytest automatically discovers conftest.py files in the rootdir and every
test subdirectory. The fixtures defined here are available to every test
under tests/ without any import.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# --- defensive sys.path setup ----------------------------------------------
# Ensures `from src.X import Y` works no matter where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- imports that depend on the path being correct --------------------------
from src.streaming.event_generator import RawEvent   # noqa: E402


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def make_event():
    """
    Factory for RawEvent with sensible defaults.

    Usage:
        def test_foo(make_event):
            ev = make_event(event_type="login", source_system="web_app_events")
    """
    counter = {"n": 0}

    def _make(
        *,
        customer_id: str = "CUST_TEST",
        event_type: str = "purchase",
        source_system: str = "card_payments",
        payload: dict | None = None,
        event_time: datetime | None = None,
        ingestion_time: datetime | None = None,
        account_id: str | None = "ACC_CC_001",
    ) -> RawEvent:
        counter["n"] += 1
        t = event_time or datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        return RawEvent(
            event_id=f"EVT_TEST_{counter['n']:04d}",
            customer_id=customer_id,
            account_id=account_id,
            source_system=source_system,
            event_type=event_type,
            schema_version="1.0",
            event_time=t,
            ingestion_time=ingestion_time or t,
            payload=payload or {"amount": 25.0, "merchant_name": "TestMerchant"},
        )

    return _make


@pytest.fixture
def scenario_dir() -> Path:
    """Path to scenario_01 if present, else skip the test."""
    p = _PROJECT_ROOT / "data" / "raw" / "scenario_01"
    if not p.exists():
        pytest.skip("scenario_01 not present")
    return p