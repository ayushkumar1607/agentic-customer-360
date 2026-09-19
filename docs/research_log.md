# Research Log

Every paper, blog, and engineering write-up consulted during the project, with a note on **what was taken from it** and **which specific code decision it informed**.

The PS explicitly weights research at 30%; the goal here is not bibliography length but traceable paper → code mapping.

---

## 1. Multi-Agent Debate

**Liang, Tian et al. — *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate* (EMNLP 2024).**
[arxiv.org/abs/2305.19118](https://arxiv.org/abs/2305.19118)

**What we learned**
- Multiple agents can explicitly defend different hypotheses before an independent judge resolves the disagreement.
- Debate is useful when disagreement itself is informative, not when it's merely noise.
- A judge at low temperature produces more consistent arbitration than majority voting.

**Where it's implemented**
- `src/agents/debate_agent.py` — `RetentionDebateAgent` and `GrowthDebateAgent` argue opposite sides from different model tiers.
- `src/agents/judge_agent.py` — `JudgeAgent` at temperature 0 picks the winning strategy.
- `src/orchestration/pipeline.py` — debate fires **only** when `synthesis.arbitration_needed` is true.

**Design choice:** Debate is expensive; we don't run it as a default stage. The Synthesis Agent's `arbitration_needed` flag gates it.

---

## 2. Multi-Agent Reflection and Critique

**Moncada-Ramirez, Jesus et al. — *Agentic Workflows for Improving Large Language Model Reasoning in Robotic Object-Centered Planning* (Robotics, 2025).**
[doi.org/10.3390/robotics14030024](https://doi.org/10.3390/robotics14030024)

**What we learned**
- Planner-feedback-refinement workflows improve outputs.
- **Excessive reflection can be harmful** when the correct result is a null response.
- A bounded number of refinement iterations is more effective than open-ended looping.

**Where it's implemented**
- `src/agents/critique_agent.py` — a dedicated reviewer that critiques the Action Agent's proposal against policy, cost, tone, and compliance.
- `src/orchestration/pipeline.py` — `MAX_CRITIQUE_ITERATIONS = 2` caps the refinement loop.
- `src/orchestration/no_action_gate.py` — the explicit No Action gate runs **before** the action loop, preventing unnecessary generation when the right answer is to wait.

**Design choice:** Bounded iteration count + explicit null-action gate, exactly as the paper recommends.

---

## 3. Governed Shared Memory for Multi-Agent LLM Systems

**Anonymous / 2026.**
[arxiv.org/abs/2606.24535](https://arxiv.org/abs/2606.24535)

**What we learned**
- Shared state in multi-agent systems needs **isolation**, **scoped access**, and **provenance**.
- Agents should publish structured conclusions, not raw reasoning traces.
- Every write should carry traceability metadata.

**Where it's implemented**
- `src/memory/state_board.py` — `StateBoard` keyed by `customer_id`; every publish carries a `trace_id` and only accepts structured findings.
- `src/agents/base_agent.py` — `BaseAgent.run()` publishes to the State Board via `self.state_board.publish(...)`. Internal reasoning traces never cross the boundary.
- `src/guardrails/rbac.py` — every agent has a scoped role that physically filters events by `source_system`.

**Design choice:** The State Board is the **only** inter-agent channel. Raw LLM reasoning is never shared between agents.

---

## 4. Tripartite Agent Memory

**Lavoie, John; Rana, Ashish et al. — *MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents* (2026).**
[arxiv.org/abs/2604.04853](https://arxiv.org/abs/2604.04853)

**What we learned**
- Working, episodic, and semantic memory serve different roles with different lifecycles.
- Preserving raw historical evidence is important for provenance and post-hoc audit.
- Working memory should be cheap to write and discarded once a case resolves.

**Where it's implemented**
- `src/memory/working_memory.py` — per-case transient memory; discarded on resolution.
- `src/memory/episodic_memory.py` — per-customer long-term history, seeded from `history_seed.jsonl`.
- `src/memory/semantic_memory.py` — ChromaDB-backed cross-customer policy knowledge.
- `src/memory/schemas.py` — `MemoryEntry` retains `source_event_id` and `trace_id` for provenance.

**Design choice:** Three distinct stores with different retention semantics, exactly as the paper describes.

---

## 5. Memory Decay

**Yao, Ashish et al. — *Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation* (2026).**
[arxiv.org/abs/2604.00131](https://arxiv.org/abs/2604.00131)

**What we learned**
- Memories can become **less accessible** over time without being deleted.
- Activation should be a function of importance, age, and reinforcement from recurring evidence.
- A decay-weighted retrieval is preferable to hard expiry because it preserves provenance while reducing influence.

**Where it's implemented**
- `src/memory/decay.py` — implements `A(m, t) = W_base · exp(-λ · (t - t_event)) · (1 + R_reinforce)`.
- `src/memory/episodic_memory.py::retrieve` — ranks memories by activation score, returns top-K above a minimum threshold.
- `src/agents/life_event_agent.py` and `src/agents/action_agent.py` — read decay-weighted episodic hits as context.

**Design choice:** An old churn flag from 18 months ago is preserved but will naturally fall below the activation threshold for new decisions. Exactly the PS's stated concern on page 5.

---

## 6. Complex Event Processing

**Waehner, Kai — *Flink CEP and Agentic AI: Real-Time Pattern Detection as the Foundation for Autonomous Decisions* (2026).**

**What we learned**
- Continuous event streams can be filtered and transformed into meaningful patterns **before** invoking expensive reasoning.
- Event-time processing (not wall-clock) is essential for correctness under out-of-order arrival.
- Rolling windows should aggregate per customer, not globally.

**Where it's implemented**
- `src/streaming/cep_engine.py` — `CEPEngine` with per-customer rolling windows keyed on `event_time`.
- `src/streaming/event_generator.py` — async replay with jitter injection to simulate out-of-order arrival.
- `src/streaming/features.py` — `extract_features` computes rolling aggregates once per customer so LLM agents receive authoritative numbers instead of raw event lists.
- `src/streaming/windows.py` — `WindowSpec` definitions (1h, 24h, 7d, 30d).

**Design choice:** CEP-lite in Python, as the mid-term committed. Full Flink would add deployment complexity without changing the demo semantics. The architectural idea — filter/transform before expensive reasoning — is retained.

---

## 7. Real-Time RAG

**Bushnam, Ganesh (Striim) — *Real-Time RAG: Streaming Vector Embeddings and Low-Latency AI Search* (2024).**

**What we learned**
- CDC-driven incremental updates can keep retrieval synchronized with changing data.
- Full re-indexing is unnecessary and slow; embeddings should be added incrementally.
- Stale embeddings should not silently persist.

**Where it's implemented**
- `src/memory/semantic_memory.py::add` — supports incremental document addition to ChromaDB without re-embedding existing docs.
- `src/memory/semantic_memory.py::query` — cosine-similarity retrieval over the persistent store.

**Design choice:** The system never rebuilds the policy corpus. New policies or CRM notes can be embedded and added live.

---

## 8. Provenance and Agent Tracing

**Strobelt, Hendrik et al. — *From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents* (2026).**
[arxiv.org/abs/2606.04990](https://arxiv.org/abs/2606.04990)

**What we learned**
- Agent executions can be represented as traceable chains covering retrieval, tool calls, and intermediate steps.
- Trust in agent decisions depends on the reviewer's ability to reconstruct the path.
- Trace IDs should propagate across async boundaries.

**Where it's implemented**
- `src/utils/tracing.py` — every stage emits a span with `trace_id`, `span_id`, `duration_ms`, and stage-specific attributes. Uses a `contextvars.ContextVar` so the trace ID flows implicitly through async calls.
- `src/output/inferred_events_writer.py` — writes the final `trace_id` into the checkpoint.
- `src/memory/state_board.py` — every published finding carries its `trace_id`.

**Design choice:** A reviewer can take any checkpoint, pull its `trace_id`, and reconstruct every agent invocation and tool call that contributed to it.

---

## 9. Latent Customer State

**Ding, Qianggang et al. — *ComBodied Agents: A New Paradigm of Human-Centric Agentic AI* (2026).**
[arxiv.org/abs/2608.10915](https://arxiv.org/abs/2608.10915)

**What we learned**
- A user's state can be represented as an **evolving latent belief** updated from multiple observations over time, not as a one-shot classification.
- Belief state should persist and be re-used by downstream reasoning rather than re-derived.

**Where it's implemented**
- `src/agents/life_event_agent.py` — infers a life phase as a **belief with supporting evidence**, not a fact. `LifeEventFindings.needs_sensitive_handling` explicitly captures uncertainty.
- `src/agents/synthesis_agent.py` — reconciles beliefs from multiple agents into one unified customer state.
- `src/memory/state_board.py` — the inferred life-phase belief is stored as a first-class finding that later agents read, rather than re-deriving.

**Design choice:** The PS page 4 explicitly requires this: *"an inferred 'current life phase' is not a one-off classification to be computed and forgotten; it belongs in the sophisticated memory architecture the participants design."* Our Life-Event Agent publishes to the State Board, so later decisions reuse the belief rather than recomputing it.

---

## 10. Composition Contribution

The PS asks explicitly: *"The closest existing approaches already demonstrate individual capabilities required by the PS. Our focus is not to claim that each component is new. The proposed contribution is the composition."*

Our composition layers:
