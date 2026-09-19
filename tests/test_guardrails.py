"""Deterministic guardrails: hard-stops, PII, RBAC."""
from __future__ import annotations

import pytest

from src.guardrails.pii import PIITokenizer
from src.guardrails.rbac import RBACViolation, rbac_for
from src.guardrails.scanner import GuardrailScanner


def test_scanner_fires_on_legal_threat(make_event):
    scanner = GuardrailScanner()
    ev = make_event(
        event_type="ticket_created",
        source_system="support_logs",
        payload={
            "raw_text": "I will sue you and my attorney will be in touch.",
            "channel": "email",
            "category": "billing_dispute",
            "resolution_status": "open",
        },
    )
    hit = scanner.scan_event(ev)
    assert hit is not None
    assert hit.rule_id == "legal_threat"
    assert hit.terminal_action == "compliance_fraud_hold"


def test_scanner_no_false_positive_on_benign_ticket(make_event):
    scanner = GuardrailScanner()
    ev = make_event(
        event_type="ticket_created",
        source_system="support_logs",
        payload={
            "raw_text": "Can I set up a payment plan?",
            "channel": "chat",
            "category": "payment_arrangements",
            "resolution_status": "open",
        },
    )
    assert scanner.scan_event(ev) is None


def test_pii_tokenizer_does_not_mangle_event_ids():
    pii = PIITokenizer()
    text = "Event EVT_000462 and account ACC_CC_001 belong to CUST_00088"
    scrubbed = pii.scrub_text(text)
    # event IDs and account IDs must pass through untouched
    assert "EVT_000462" in scrubbed
    assert "ACC_CC_001" in scrubbed
    # customer IDs get tokenized
    assert "CUST_00088" not in scrubbed
    assert "CUST_TOKEN_" in scrubbed


def test_rbac_blocks_cross_domain_reads(make_event):
    usage = rbac_for("usage_agent")
    kyc = rbac_for("kyc_agent")

    tx_event = make_event(source_system="card_payments", event_type="purchase")
    web_event = make_event(source_system="web_app_events", event_type="login")

    assert usage.filter_events([tx_event, web_event]) == [web_event]
    assert kyc.filter_events([tx_event, web_event]) == []


def test_rbac_unknown_role_raises():
    with pytest.raises(RBACViolation):
        rbac_for("not_a_real_role")