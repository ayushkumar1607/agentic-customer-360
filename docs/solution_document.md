# Agentic Customer 360 — Solution Document

**Inter IIT Tech Meet 15.0 — Prepathon 2026 — NLP Track**

---

## 1. Problem

Traditional Customer 360 systems aggregate data into dashboards. Dashboards are passive; they require a human to interpret and act. At thousands of customers, no human team can continuously monitor subtle, compounding shifts — a quiet churn signal, a life event that opens a strategic opportunity, a pattern that looks like fraud.

We are asked for something different: an **ambient system** that continuously processes multi-source events, maintains evolving customer state, detects meaningful change, and **commits to a specific terminal decision** — including explicitly deciding to do nothing.

The system must be safe: guardrails that cannot be reasoned around, human-in-the-loop at points where the cost of error is high, and falsifiable outputs that can be scored against hidden ground truth.

---

## 2. Architecture

We use an **Event-Driven Ambient Multi-Agent Blackboard Architecture**. The design separates deterministic safety from non-deterministic agentic reasoning.

### 2.1 Synchronous fast path

Every incoming event passes through:
1. **PII tokenization** — stable, reversible tokens for identifiers.
2. **Deterministic guardrail scan** — keyword rules for legal threats, self-harm, sanctions, account takeover. A hit diverts the case to a safe terminal state and halts autonomous communication.

This path never blocks on an LLM. Safety decisions must be deterministic and instant.

### 2.2 Asynchronous ambient reasoning

Surviving events flow into the reasoning pipeline:

1. **CEP engine** — per-customer rolling windows keyed on `event_time`, with out-of-order insertion via bisect.
2. **Feature computation** — authoritative rolling aggregates (login trend, spend variance, sentiment) computed once per customer.
3. **Governed tripartite memory** — working memory (per-case), episodic memory (per-customer history with decay-weighted retrieval), semantic memory (ChromaDB policy store).
4. **Swarm** — four specialist agents (Usage, Support, Transaction, KYC) run in parallel. Each has its own RBAC scope, prompt, output schema, and model tier.
5. **Agent-dependent trigger** — the Life-Event Agent fires only when ≥2 swarm agents flag correlated signals.
6. **Synthesis** — reconciles all findings into one unified customer state and decides whether the swarm genuinely conflicts.
7. **Debate + Judge** — fires only on genuine conflict; two personas argue explicitly, an impartial judge picks the strategy.
8. **No Action gate** — deterministic. Short-circuits to `no_action` if churn risk is low and no opportunity/conflict exists.
9. **Action + Critique** — the Action Agent drafts a bounded, costed proposal; the Critique Agent reviews with a maximum of 2 refinement iterations.
10. **Guardrail override** — the final message draft is re-scanned. Any hit overrides the proposal to `compliance_fraud_hold`.
11. **HITL** — a real async interruption for costly, sensitive, or high-risk actions.
12. **Terminal output** — a checkpoint written to `inferred-events.jsonl` in the exact README schema.

---

## 3. Key Design Decisions

### 3.1 Stage-specific MAS topologies

The PS explicitly warns against one topology everywhere. We use **four distinct patterns**:

| Stage | Topology | Why |
|---|---|---|
| Signal gathering | **Swarm** (parallel, independent) | Sources don't depend on each other |
| Life-event trigger | **Agent-dependent** | Correlates weak signals into a strong one |
| Synthesis | **Handoff** | Single downstream consumer |
| Conflict | **Debate + Judge** | Disagreement is informative |
| Action + Critique | **Critique-Refiner** | Bounded iteration prevents over-generation |

Grounded in Liang et al. (debate) and Moncada-Ramirez (critique-refiner with iteration bounds).

### 3.2 Deterministic vs. agentic boundary

The PS is unambiguous: guardrails, RBAC, PII masking, and event-time correctness must be **deterministic**. We enforce this boundary in code:

- `src/guardrails/scanner.py` — keyword-based, no LLM
- `src/guardrails/rbac.py` — data-layer filter, not a prompt
- `src/guardrails/pii.py` — regex-based tokenization
- `src/streaming/cep_engine.py` — event-time ordering

Everything else — inference, synthesis, debate, action drafting, critique — is agentic.

### 3.3 Memory with lifecycle

We implement three memory types with distinct lifecycles:

- **Working memory** — discarded on case resolution.
- **Episodic memory** — long-term with decay-weighted retrieval: `A(m, t) = W_base · exp(-λ · (t - t_event)) · (1 + R_reinforce)`.
- **Semantic memory** — cross-customer, incrementally updated in ChromaDB.

This is grounded in MemMachine (tripartite memory) and Oblivion (decay-driven activation).

### 3.4 Governed State Board

Agents communicate only through a shared, per-customer State Board. Each agent publishes **structured findings**, never raw reasoning traces. This prevents context pollution and provides an audit trail. Grounded in Governed Shared Memory for Multi-Agent LLM Systems.

### 3.5 Explicit No Action

No Action is a first-class terminal decision with its own checkpoint, not the absence of a decision. The No Action gate is deterministic: it fires when churn risk is below threshold, no opportunity exists, and no conflict was detected. This is directly required by the PS.

### 3.6 Sensitive-inference override

When the Life-Event Agent infers a sensitive state (medical, financial distress, marital change, relocation), the pipeline **deterministically overrides** whatever the Action Agent proposed and routes to `relationship_manager_escalation`. This is a code-level guarantee, not a prompt instruction. It prevents the LLM from autonomously pitching a product to a customer in crisis.

### 3.7 Real HITL

HITL is an async interruption in the workflow, not a log line. The pipeline submits a request to an `HITLQueue` and awaits a decision. Triggers include cost threshold ($200), low confidence, sensitive life events, unresolved disagreement, and high-risk action types.

### 3.8 Tiered model routing

Different tasks need different models. We use three tiers via Groq:

- `fast` (`openai/gpt-oss-20b`) — extraction agents
- `reason` (`openai/gpt-oss-120b`) — Life-Event, Synthesis, Action, Critique, Judge
- `alt` (`qwen/qwen3.6-27b`) — Growth debate persona, for genuine divergence

Per-model fallbacks are declared in `settings.py` so provider drift doesn't break the system mid-evaluation.

### 3.9 Falsifiable output

Every checkpoint is written to `inferred-events.jsonl` in the exact schema the README specifies. The evaluation harness (`scripts/run_evaluation.py`) scores our output against a ground-truth JSON and reports per-field accuracy, per-class precision/recall/F1, and confidence calibration.

---

## 4. Non-Negotiable Components — Demonstrated

| Component | Where it lives | How it's shown |
|---|---|---|
| **Guardrails** | `guardrails/scanner.py` + `config/guardrails.yaml` | `test_scanner_fires_on_legal_threat` proves a legal threat halts the pipeline |
| **PII protection** | `guardrails/pii.py` | `test_pii_tokenizer_does_not_mangle_event_ids` proves event IDs pass through; CUST IDs are tokenized |
| **RBAC (data-layer)** | `guardrails/rbac.py` | `test_rbac_blocks_cross_domain_reads` proves Usage Agent cannot read transactions |
| **Explainability** | `output/inferred_events_writer.py` | Every checkpoint carries `notes` with reasoning + `trace_id` |
| **Traceability** | `utils/tracing.py` | Every stage emits a span with `trace_id`; spans are structured logs |
| **Real HITL** | `hitl/approval_queue.py` + `ui/app.py` | Async queue with approve/reject/modify |

---

## 5. Evaluation

We score against a sample ground truth on the three provided scenarios:

| Scenario | Customer | Inferred state | Action | HITL |
|---|---|---|---|---|
| 01 | CUST_00088 | `medical_hardship` | `relationship_manager_escalation` | `escalated` |
| 02 | CUST_00105 | `new_child_life_event` | `relationship_manager_escalation` | `escalated` |
| 03 | CUST_00184 | `no_significant_event` | `no_action` | `auto_approved` |

Against our sample GT (which mirrors our best reading of each scenario's story):
- `inferred_state` accuracy: 100%
- `action` accuracy: 100%
- `hitl_status` accuracy: 100%
- Confidence calibration: `high` band accuracy 100% (n=2), `medium` accuracy 100% (n=1)

The harness writes `output/evaluation_report.json` for review.

---

## 6. Known Limitations

- **Streaming** — We use an async Python CEP layer rather than Flink/Kafka. Semantics are equivalent for the demo; distributed infrastructure is a deployment concern.
- **Storage** — In-process memory and ChromaDB. Production would add Redis (HITL queue), Postgres (state), and a distributed vector store.
- **LLM non-determinism** — Bounded by typed schemas, deterministic guardrails, and HITL, but not eliminated. A judge at temperature 0 is not a guarantee of determinism.
- **Ground truth** — Our sample GT is a best-effort reconstruction from each scenario's story. The organizers' hidden GT may have additional intermediate checkpoints that would require temporal alignment.
- **Debate latency** — Debate adds significant latency; we only fire it on genuine conflict.

---

## 7. What's Novel

Individual capabilities — tripartite memory, CEP, debate, guardrails, HITL — exist in prior work. Our contribution is the **composition into one governed ambient workflow** where:

- Every stage uses the MAS topology that fits its purpose.
- Sensitive inferences are handled by a **deterministic code-level override**, not an LLM prompt.
- The No Action decision is as traceable and auditable as an active decision.
- Memory decay and State Board governance prevent stale or cross-customer signal leakage.
- Tiered model routing allows genuine debate divergence.

The architecture is falsifiable end-to-end: every checkpoint is a claim, and every claim can be scored against ground truth.