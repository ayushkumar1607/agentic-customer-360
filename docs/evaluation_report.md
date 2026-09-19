# Evaluation Report

**Run date:** 2026-09-19
**Pipeline version:** Phase 4 (final)
**Scenarios evaluated:** 3 (`scenario_01`, `scenario_02`, `scenario_03`)
**Ground truth:** `data/ground_truth/sample_ground_truth.json`
**Harness:** `src/evaluation/scoring_harness.py`, invoked via `python -m scripts.run_evaluation`

---

## 1. Methodology

For each scenario we run the full pipeline end-to-end:

1. Stream events (history seed + live stream) through CEP
2. Run deterministic guardrails, PII tokenization, RBAC-scoped feature computation
3. Run the specialist swarm (Usage, Support, Transaction, KYC)
4. Agent-dependent trigger fires Life-Event Inference when ≥2 agents flag signals
5. Synthesis produces unified customer state
6. Debate + Judge fires only on genuine conflict
7. No Action gate checks whether evidence crosses action threshold
8. Action Agent drafts bounded proposal; Critique Agent reviews (max 2 iterations)
9. Sensitive-inference override forces `relationship_manager_escalation` when applicable
10. HITL is required if the action is costly, sensitive, or high-risk
11. Checkpoint written to `output/inferred-events.jsonl`

The harness then pairs each checkpoint with its expected values and computes:

- Per-field accuracy (exact match)
- Per-class precision, recall, and F1 (macro-averaged)
- Confidence-band calibration

---

## 2. Per-Scenario Results

### Scenario 01 — CUST_00088

**Story:** Support ticket explicitly states hospitalization and income drop; customer requests a payment plan.

| Field | Expected | Predicted | Match |
|---|---|---|---|
| `inferred_state` | `medical_hardship` | `medical_hardship` | ✅ |
| `action` | `relationship_manager_escalation` | `relationship_manager_escalation` | ✅ |
| `hitl_status` | `escalated` | `escalated` | ✅ |
| `confidence_band` | — | `high` | — |

**Reasoning trace:**
- Life-Event Agent inferred `medical_hardship` at 0.85 confidence from the ticket text and a related search.
- Sensitive-inference rule (`needs_sensitive_handling = true`) triggered a deterministic code-level override, forcing RM escalation rather than any customer-facing action.
- Critique Agent reviewed and cited `POL-HITL-002` (sensitive-inference policy), verdict `pass`.

### Scenario 02 — CUST_00105

**Story:** Dependents change in KYC, combined with healthcare and pharmacy spend consistent with a new child.

| Field | Expected | Predicted | Match |
|---|---|---|---|
| `inferred_state` | `new_child_life_event` | `new_child_life_event` | ✅ |
| `action` | `relationship_manager_escalation` | `relationship_manager_escalation` | ✅ |
| `hitl_status` | `escalated` | `escalated` | ✅ |
| `confidence_band` | — | `high` | — |

**Reasoning trace:**
- Life-Event Agent inferred `new_child_life_event` at 0.78 confidence.
- Sensitive-inference override forced RM escalation.

**History — this scenario initially failed.** See Section 5 (Honest Failures) for the root cause and the fix.

### Scenario 03 — CUST_00184

**Story:** No signals cross the action threshold.

| Field | Expected | Predicted | Match |
|---|---|---|---|
| `inferred_state` | `no_significant_event` | `no_significant_event` | ✅ |
| `action` | `no_action` | `no_action` | ✅ |
| `hitl_status` | `auto_approved` | `auto_approved` | ✅ |
| `confidence_band` | — | `medium` | — |

**Reasoning trace:**
- Synthesis reported churn risk 0.05, no opportunity, no conflict.
- No Action gate fired deterministically (churn below 0.30, no opportunity, no conflict).
- The explicit No Action decision is logged with a `trace_id` and is as auditable as any active decision.

---

## 3. Aggregate Metrics

### Field-level accuracy

| Field | Accuracy | Macro F1 |
|---|---|---|
| `inferred_state` | 100.00% | 1.000 |
| `action` | 100.00% | 1.000 |
| `hitl_status` | 100.00% | 1.000 |
| **Overall (all three fields)** | **100.00%** | — |

### Confidence calibration

| Band | n | Accuracy |
|---|---|---|
| `high` | 2 | 100.00% |
| `medium` | 1 | 100.00% |
| `low` | 0 | — |

---

## 4. Coverage Gaps

These are things the current evaluation **does not** exercise, but which are implemented and unit-tested:

| Gap | What's missing | Test coverage |
|---|---|---|
| **Debate + Judge path** | None of the 3 scenarios produced a genuine conflict, so no live debate was triggered | Unit-level only — no end-to-end debate scenario |
| **Low-confidence band** | No checkpoint had `confidence_band = low` | N/A |
| **Hard-stop guardrails** | No live scenario triggered a legal-threat, self-harm, sanctions, or account-takeover hard-stop | Covered by `test_scanner_fires_on_legal_threat` |
| **HITL reject / modify paths** | All HITL requests were auto-approved | No live reject/modify scenario |
| **Multi-checkpoint temporal alignment** | Each scenario produces exactly one final checkpoint | Harness matches latest-GT-to-latest-predicted only |

A more complete evaluation would include:
- A scripted scenario that produces a genuine Retention-vs-Growth conflict
- A scenario that trips a hard-stop guardrail (legal threat + fraud pattern)
- A scenario with low synthesis confidence to exercise the `low` calibration band
- A HITL reject test to confirm `no_action` is the correct fallback

---

## 5. Honest Failures and Fixes

The PS explicitly rewards documenting failure modes: *"A well-documented failure mode is worth more in evaluation than a suspiciously perfect scorecard."*

### Failure 1 — Scenario 02 initially returned `no_action` despite a valid life event

**Symptom:** The pipeline correctly inferred `new_child_life_event` but the Action Agent's proposal was rejected by the Critique Agent, and the fallback defaulted to `no_action`. This meant we inferred a meaningful life event and then did nothing about it — exactly the anti-pattern the PS warns against.

**Root cause:** Sensitive-inference handling was left to LLM judgment. The Action Agent had to decide "should this go to a human?", and when the Critique Agent disagreed, the fallback path defaulted to inaction.

**Fix:**
1. Introduced a **deterministic code-level override** in `src/orchestration/pipeline.py`. When the Life-Event Agent marks an inference as `needs_sensitive_handling`, the pipeline now constructs an RM-escalation proposal directly — the Action Agent's LLM judgment is bypassed.
2. Added a fallback rule: if the action loop fails (no proposal, or rejected) and a life event is present, default to `relationship_manager_escalation`, **never** `no_action`.

**Lesson:** Sensitive-inference handling is a compliance requirement, not a reasoning task. It belongs in code, not in a prompt.

### Failure 2 — Guardrail scanner silently loaded zero rules

**Symptom:** The scanner appeared to run but never fired on any input, even obvious legal-threat text. The log showed `hard_stop_rules=[]` with no error.

**Root cause:** The config path was relative to the current working directory (`Path("config/guardrails.yaml")`). When the pipeline or tests ran from a different directory, the file resolved to the wrong location and the YAML loaded empty.

**Fix:**
1. Anchored the config path to the module location: `Path(__file__).resolve().parents[2] / "config" / "guardrails.yaml"`.
2. Added a **loud `RuntimeError`** if the file loads with zero `hard_stops` entries. Silent failures in safety-critical layers are worse than loud ones.

**Lesson:** A guardrail layer that appears to work but silently does nothing is worse than no guardrail at all.

### Failure 3 — Small model ignored `computed_features` and returned zeros

**Symptom:** The Usage Agent returned `login_trend_30d = 0.0` despite CEP computing `-0.2857`. The Transaction Agent returned `confidence = 0.0` with `notes = "no computed features available"` — even though `computed_features` was passed in.

**Root cause:** The `computed_features` block was buried at the bottom of a large JSON prompt, after 40+ event objects. Small models (20B) have a strong recency bias and missed it.

**Fix:**
1. Added `priority_context_keys` to `BaseAgent` — a tuple of context keys (`computed_features`, `state_board`, `synthesis`, `judge_verdict`) that get **hoisted to the top of the prompt**, before the bulky `events` array.
2. Tightened the agents' system prompts to explicitly instruct them to **copy** the authoritative values verbatim from `computed_features`, not recompute them.

**Lesson:** Prompt structure matters as much as prompt content, especially for smaller models.

---

## 6. Limitations

- **Ground truth is our own.** The `sample_ground_truth.json` file was constructed from our best reading of each scenario's story. The organizers' hidden GT may have additional intermediate checkpoints and a different alignment.
- **Small sample.** Three scenarios, one checkpoint each. A 100% result on three examples is encouraging but not statistically meaningful.
- **Debate branch untested live.** The Debate + Judge path is implemented and unit-tested but was not exercised end-to-end.
- **LLM non-determinism.** Runs may vary slightly across invocations. The Judge runs at temperature 0 for stability, but exact reproducibility is not guaranteed.
- **Latency.** Full pipeline on one scenario takes 30–60 seconds on the Groq free tier, dominated by LLM calls. Rate limits occasionally trigger retries (visible in logs as `HTTP 429` followed by backoff).

---

## 7. Reproducibility

```bash
# 1. Run the pipeline on all scenarios
python -m scripts.run_full_pipeline --all

# 2. Score the output against ground truth
python -m scripts.run_evaluation

# 3. Run the full test suite
pytest tests/ -v