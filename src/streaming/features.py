"""
Rolling feature computation aligned to the real dataset schema.

Event families (from source_system):
    card_payments        -> purchase, refund, decline
    instant_payments/ach -> inbound_transfer, outbound_transfer
    core_banking_ledger  -> deposit, withdrawal, standing_instruction, fee, interest_credit
    trading_brokerage    -> buy, sell, dividend, deposit_to_brokerage
    loan_kyc             -> loan_application, loan_disbursed, kyc_update,
                            address_change, marital_status_change, dependents_change
    web_app_events       -> login, search_query, feature_used, session_duration
    support_logs         -> ticket_created, ticket_resolved, call_transcript
    social_signal        -> life_event_mention

Sentiment handling — IMPORTANT:
    The function `_derive_sentiment` below is a LEXICON-BASED PLACEHOLDER.
    It exists ONLY so the CEP layer can emit a cheap numeric rolling signal
    without blocking on an API call.

    It is NOT the authoritative sentiment for the decision path.

    The Support / Sentiment Agent (Phase 3) performs the final classification
    via LLM / fine-tuned classifier and publishes the authoritative
    sentiment_score, urgency, and churn_intent to the State Board.

    Downstream agents must read sentiment from the State Board (Support Agent
    findings), NOT from this CEP feature.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.streaming.event_generator import RawEvent


# ---- event family predicates ----------------------------------------------

_LOGIN_TYPES = {"login", "session_duration"}

_TX_SOURCES = {
    "card_payments",
    "instant_payments",
    "ach_wire",
    "core_banking_ledger",
    "trading_brokerage",
}

_TX_TYPES = {
    "purchase", "refund", "decline",
    "inbound_transfer", "outbound_transfer",
    "deposit", "withdrawal", "fee", "interest_credit",
    "standing_instruction",
    "buy", "sell", "dividend", "deposit_to_brokerage",
}

_FAILED_TYPES = {"decline"}

_SUPPORT_TYPES = {"ticket_created", "ticket_resolved", "call_transcript"}

_KYC_TYPES = {
    "kyc_update", "address_change", "marital_status_change",
    "dependents_change", "loan_application", "loan_disbursed",
}

_SEARCH_TYPES = {"search_query", "feature_used"}


def _is_login(ev: RawEvent) -> bool:
    return ev.source_system == "web_app_events" and ev.event_type in _LOGIN_TYPES


def _is_transaction(ev: RawEvent) -> bool:
    return ev.event_type in _TX_TYPES or ev.source_system in _TX_SOURCES


def _is_support(ev: RawEvent) -> bool:
    return ev.source_system == "support_logs" or ev.event_type in _SUPPORT_TYPES


def _is_failed(ev: RawEvent) -> bool:
    if ev.event_type in _FAILED_TYPES:
        return True
    status = str(ev.payload.get("status", "")).lower()
    return status in {"failed", "declined", "rejected", "error"}


def _is_kyc(ev: RawEvent) -> bool:
    return ev.source_system == "loan_kyc" or ev.event_type in _KYC_TYPES


# ---- field extractors ------------------------------------------------------

def _amount(ev: RawEvent) -> float | None:
    p = ev.payload or {}
    for k in ("amount", "value", "amt", "transaction_amount"):
        if p.get(k) is not None:
            try:
                return float(p[k])
            except (TypeError, ValueError):
                return None
    return None


# Minimal lexicon. Kept narrow on purpose — this is a rolling signal only.
_NEG_WORDS = {
    "angry", "furious", "unacceptable", "terrible", "worst", "cancel",
    "lawsuit", "legal", "attorney", "refund", "scam", "fraud",
    "stolen", "unauthorized", "close my account", "ridiculous", "disgusted",
    "frustrated", "useless", "never again", "disappointed",
}
_POS_WORDS = {
    "thanks", "thank you", "great", "excellent", "happy", "love",
    "appreciate", "helpful", "resolved", "perfect",
}


def _derive_sentiment(text: str) -> float:
    """
    PLACEHOLDER — lexicon-based sentiment for CEP-side rolling features only.

    The Support Agent (Phase 3) performs the authoritative sentiment and
    urgency classification via LLM/HF classifier. This function exists so
    the CEP layer can emit a numeric rolling signal without blocking on
    an API call.

    Do NOT use this value as the final sentiment in the decision path.
    Downstream agents must read sentiment from the State Board.
    """
    if not text:
        return 0.0
    t = text.lower()
    neg = sum(1 for w in _NEG_WORDS if w in t)
    pos = sum(1 for w in _POS_WORDS if w in t)
    total = neg + pos
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def _support_text(ev: RawEvent) -> str:
    p = ev.payload or {}
    return (
        p.get("raw_text")
        or p.get("transcript")
        or p.get("message")
        or ""
    )


# ---- main feature extractor ------------------------------------------------

def extract_features(windows: dict[str, list[RawEvent]]) -> dict[str, Any]:
    w1h = windows.get("1h", [])
    w24h = windows.get("24h", [])
    w7d = windows.get("7d", [])
    w30d = windows.get("30d", [])

    # Login frequency
    logins_7d = sum(1 for e in w7d if _is_login(e))
    logins_30d = sum(1 for e in w30d if _is_login(e))
    login_frequency_7d = logins_7d / 7.0
    login_frequency_30d = logins_30d / 30.0
    login_trend_30d = (
        (login_frequency_7d - login_frequency_30d) / login_frequency_30d
        if login_frequency_30d > 0 else 0.0
    )

    # Transaction velocity
    transaction_velocity_1h = sum(1 for e in w1h if _is_transaction(e))

    # Spend variance / mean / std (30d)
    amounts = [a for a in (_amount(e) for e in w30d if _is_transaction(e)) if a is not None]
    if len(amounts) >= 2:
        spend_variance_30d = float(np.var(amounts))
        spend_mean_30d = float(np.mean(amounts))
        spend_std_30d = float(np.std(amounts))
    else:
        spend_variance_30d = 0.0
        spend_mean_30d = float(amounts[0]) if amounts else 0.0
        spend_std_30d = 0.0

    # Sentiment (PLACEHOLDER rolling signal — see module docstring)
    sent_7d = [
        _derive_sentiment(_support_text(e))
        for e in w7d if _is_support(e) and _support_text(e)
    ]
    sent_30d = [
        _derive_sentiment(_support_text(e))
        for e in w30d if _is_support(e) and _support_text(e)
    ]
    sentiment_7d_mean = float(np.mean(sent_7d)) if sent_7d else 0.0
    sentiment_30d_mean = float(np.mean(sent_30d)) if sent_30d else 0.0
    sentiment_delta_7d = sentiment_7d_mean - sentiment_30d_mean

    # Failed transactions
    failed_transaction_count = sum(1 for e in w30d if _is_failed(e))

    # Life-event signal counters
    kyc_events_30d = [e for e in w30d if _is_kyc(e)]
    search_events_7d = [e for e in w7d if e.event_type in _SEARCH_TYPES]

    home_loan_search_flag = any(
        "loan" in str(e.payload.get("search_text", "")).lower()
        or "mortgage" in str(e.payload.get("search_text", "")).lower()
        for e in search_events_7d
    )

    # Distribution
    type_counts_30d = dict(Counter(e.event_type for e in w30d))
    source_counts_30d = dict(Counter(e.source_system for e in w30d))

    return {
        "login_frequency_7d": round(login_frequency_7d, 4),
        "login_frequency_30d": round(login_frequency_30d, 4),
        "login_trend_30d": round(login_trend_30d, 4),
        "transaction_velocity_1h": transaction_velocity_1h,
        "spend_variance_30d": round(spend_variance_30d, 4),
        "spend_mean_30d": round(spend_mean_30d, 4),
        "spend_std_30d": round(spend_std_30d, 4),
        "failed_transaction_count": failed_transaction_count,
        "sentiment_7d_mean_rolling": round(sentiment_7d_mean, 4),
        "sentiment_30d_mean_rolling": round(sentiment_30d_mean, 4),
        "sentiment_delta_7d_rolling": round(sentiment_delta_7d, 4),
        "kyc_event_count_30d": len(kyc_events_30d),
        "kyc_event_types_30d": [e.event_type for e in kyc_events_30d],
        "search_event_count_7d": len(search_events_7d),
        "home_loan_search_flag": home_loan_search_flag,
        "event_type_counts_30d": type_counts_30d,
        "source_counts_30d": source_counts_30d,
        "window_sizes": {
            "1h": len(w1h), "24h": len(w24h),
            "7d": len(w7d), "30d": len(w30d),
        },
    }