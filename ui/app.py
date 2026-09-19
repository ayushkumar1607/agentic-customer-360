"""
Streamlit HITL reviewer.

Run in a second terminal while the pipeline is running with --hitl-live.
For a same-process demo, the pipeline reads/writes a JSONL file at
output/hitl_queue.jsonl that this UI reads and updates.

Usage:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

QUEUE_FILE = Path("output/hitl_queue.jsonl")

st.set_page_config(page_title="Customer 360 — HITL Review", layout="wide")
st.title("🛡️ HITL Review — Proactive Intervention Desk")

if not QUEUE_FILE.exists():
    st.info("No HITL requests yet. Start the pipeline with `--hitl-live`.")
    st.stop()

requests = []
for line in QUEUE_FILE.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        requests.append(json.loads(line))
    except json.JSONDecodeError:
        pass

pending = [r for r in requests if r.get("decision") is None]
decided = [r for r in requests if r.get("decision") is not None]

st.subheader(f"Pending requests: {len(pending)}")
for req in pending:
    with st.expander(
        f"⚠️ {req['customer_id']} — {req['action']} "
        f"(${req['cost_estimate_usd']:.0f}, conf {req['confidence']:.2f})",
        expanded=True,
    ):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("**Proposed action**")
            st.write(req["action"])
            if req.get("action_subtype"):
                st.caption(f"Subtype: {req['action_subtype']}")
            st.markdown("**Reasoning**")
            st.write(req["reasoning"])
            if req.get("message_draft"):
                st.markdown("**Message draft**")
                st.info(req["message_draft"])
            st.markdown("**Supporting evidence**")
            st.write(req.get("evidence", []))
        with c2:
            st.markdown("**Cost / Confidence**")
            st.metric("Cost (USD)", f"${req['cost_estimate_usd']:.0f}")
            st.metric("Confidence", f"{req['confidence']:.2f}")
            st.markdown("**Synthesis snapshot**")
            st.json(req.get("synthesis_snapshot", {}))

        notes = st.text_input("Reviewer notes", key=f"notes-{req['request_id']}")
        colA, colB, colC = st.columns(3)
        if colA.button("✅ Approve", key=f"a-{req['request_id']}"):
            _update(req["request_id"], "approve", notes)
            st.success("Approved"); st.rerun()
        if colB.button("❌ Reject", key=f"r-{req['request_id']}"):
            _update(req["request_id"], "reject", notes)
            st.error("Rejected"); st.rerun()
        if colC.button("✏️ Modify", key=f"m-{req['request_id']}"):
            _update(req["request_id"], "modify", notes,
                    modified={"action": req["action"]})
            st.warning("Modified"); st.rerun()

if decided:
    st.subheader(f"Resolved: {len(decided)}")
    for r in decided[-5:]:
        st.caption(f"{r['customer_id']} — {r['action']} → {r['decision']}")