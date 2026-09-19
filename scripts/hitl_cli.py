"""
Terminal-based HITL reviewer.

Run this in a second terminal while the pipeline is running with --hitl-live.
It polls the queue, shows pending requests, and lets the reviewer approve,
reject, or modify.

For the smoke test we run in auto-approve mode; this CLI is for demoing a
real interruption.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.hitl.approval_queue import HITLQueue, HITLStatus  # noqa: F401


# Placeholder — full CLI wiring is trivial once we use a persistent queue
# (Redis / SQLite). For the current in-memory queue, the pipeline and CLI
# must share a process. See ui/app.py for the real interactive path.


async def main() -> None:
    print("HITL CLI must run in-process with the pipeline.")
    print("Use `ui/app.py` (Streamlit) for the real HITL reviewer UI.")


if __name__ == "__main__":
    asyncio.run(main())