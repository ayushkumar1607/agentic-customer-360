"""CEP window management, late-event handling, event-time ordering."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.streaming.cep_engine import CEPEngine


def _t(days: int, hours: int = 0) -> datetime:
    base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(days=days, hours=hours)


def test_cep_basic_ingest(make_event):
    """5 events 1 minute apart should all fit in the 1h window."""
    cep = CEPEngine()
    base = _t(0, 0)
    for i in range(5):
        cep.ingest(
            make_event(
                event_time=base + timedelta(minutes=i),
                event_type="login",
            )
        )
    assert len(cep.window("CUST_TEST", "1h")) == 5
    assert len(cep.window("CUST_TEST", "24h")) == 5


def test_cep_1h_window_prunes_older_events(make_event):
    """Events spread across 5 hours: the 1h window keeps only the last hour."""
    cep = CEPEngine()
    for i in range(5):
        cep.ingest(make_event(event_time=_t(0, i), event_type="login"))

    # "now" is hour 4 -> cutoff is hour 3 -> only hours 3 and 4 survive
    events_1h = cep.window("CUST_TEST", "1h")
    assert len(events_1h) == 2
    assert all(e.event_time >= _t(0, 3) for e in events_1h)

    # The 24h window still holds all 5
    assert len(cep.window("CUST_TEST", "24h")) == 5


def test_cep_late_event_lands_in_correct_position(make_event):
    """A late event whose event_time falls between two existing events should
    be inserted at the correct position, not appended at the end."""
    cep = CEPEngine()
    e1 = make_event(event_time=_t(0, 10), event_type="login")
    e3 = make_event(event_time=_t(0, 12), event_type="login")
    cep.ingest(e1)
    cep.ingest(e3)

    # Late event with event_time between e1 and e3
    e2 = make_event(
        event_time=_t(0, 11),
        ingestion_time=_t(0, 15),
        event_type="login",
    )
    cep.ingest(e2)

    events = cep.window("CUST_TEST", "24h")
    times = [e.event_time for e in events]
    assert times == sorted(times), "window must be sorted by event_time"
    assert len(events) == 3
    late = cep.late_insertions("CUST_TEST")
    assert late["24h"] >= 1


def test_cep_window_pruning(make_event):
    """Events older than the window size should be dropped on prune."""
    cep = CEPEngine()
    cep.ingest(make_event(event_time=_t(0)))
    cep.ingest(make_event(event_time=_t(40)))
    assert len(cep.window("CUST_TEST", "30d")) == 1


def test_cep_lag_tracking(make_event):
    cep = CEPEngine()
    ev = make_event(
        event_time=_t(0),
        ingestion_time=_t(0) + timedelta(hours=2),
    )
    cep.ingest(ev)
    stats = cep.lag_stats("CUST_TEST")
    assert stats["count"] == 1
    assert stats["max_s"] == 7200.0