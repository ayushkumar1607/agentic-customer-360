# Agentic Customer 360 — Proactive Intervention Desk

> Inter IIT Tech Meet 15.0 — Prepathon 2026 — NLP Track
> An event-driven ambient multi-agent system that continuously infers customer
> life events and decides on bounded, costed interventions — or explicitly
> chooses No Action.

---

## TL;DR

Given a live stream of banking events (transactions, logins, support tickets,
KYC updates), the system:

1. **Streams** events through an async CEP layer with rolling windows
2. **Enforces** deterministic guardrails, PII tokenization, and data-layer RBAC
3. **Runs a swarm** of four specialist agents (Usage, Support, Transaction, KYC)
4. **Infers life events** when ≥2 agents flag correlated signals
5. **Synthesizes** findings into one coherent customer state
6. **Debates + judges** only when the swarm genuinely conflicts
7. **Decides** on one of six bounded terminal actions — or explicit No Action
8. **Refines** proposals through a bounded Critique loop (max 2 iterations)
9. **Routes to HITL** for costly, sensitive, or high-risk actions
10. **Emits** a falsifiable `inferred-events.jsonl` in the required schema

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- A Groq API key (free tier: https://console.groq.com/keys)

### 2. Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt