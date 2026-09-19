"""
Verify that all Phase 2 files are the correct (full) versions.

Run:
    python -m scripts.verify_phase2
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# (module_path, required_attribute)
CHECKS = [
    # streaming
    ("src.streaming.event_generator", "RawEvent"),
    ("src.streaming.event_generator", "stream_scenario"),
    ("src.streaming.event_generator", "resolve_scenario_dir"),
    ("src.streaming.cep_engine", "CEPEngine"),
    ("src.streaming.cep_engine", "WindowState"),
    ("src.streaming.features", "extract_features"),

    # guardrails
    ("src.guardrails.pii", "PIITokenizer"),
    ("src.guardrails.rbac", "RBAC"),
    ("src.guardrails.rbac", "RBACViolation"),
    ("src.guardrails.rbac", "rbac_for"),
    ("src.guardrails.rbac", "ROLE_SCOPES"),
    ("src.guardrails.scanner", "GuardrailScanner"),
    ("src.guardrails.scanner", "HardStop"),

    # memory
    ("src.memory.decay", "activation"),
    ("src.memory.decay", "age_days"),
    ("src.memory.schemas", "MemoryEntry"),
    ("src.memory.schemas", "StateBoardSlot"),
    ("src.memory.working_memory", "WorkingMemory"),
    ("src.memory.episodic_memory", "EpisodicMemory"),
    ("src.memory.semantic_memory", "SemanticMemory"),
    ("src.memory.state_board", "StateBoard"),

    # agents
    ("src.agents.base_agent", "BaseAgent"),

    # output
    ("src.output.schema", "InferredState"),
    ("src.output.schema", "Action"),
    ("src.output.schema", "HITLStatus"),
    ("src.output.schema", "ConfidenceBand"),
    ("src.output.schema", "Checkpoint"),
    ("src.output.schema", "confidence_to_band"),

    # utils
    ("src.utils.logger", "get_logger"),
    ("src.utils.tracing", "span"),
    ("src.utils.tracing", "new_trace_id"),
    ("src.utils.tracing", "set_trace_id"),
    ("src.utils.llm_client", "complete"),
    ("src.utils.llm_client", "complete_json"),

    # config
    ("config.settings", "settings"),
]

# Class-level checks (attributes that must exist on an object)
ATTR_CHECKS = [
    ("config.settings", "settings", "groq_model_fast"),
    ("config.settings", "settings", "groq_model_reason"),
    ("config.settings", "settings", "groq_model_alt"),
    ("config.settings", "settings", "memory_decay_lambda"),
]


def main() -> int:
    fails: list[str] = []

    for mod_path, attr in CHECKS:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            fails.append(f"IMPORT FAIL  {mod_path}: {e}")
            continue
        if not hasattr(mod, attr):
            fails.append(f"MISSING ATTR {mod_path}.{attr}")
        else:
            print(f"OK   {mod_path}.{attr}")

    for mod_path, obj_name, attr in ATTR_CHECKS:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, obj_name)
        except Exception as e:
            fails.append(f"IMPORT FAIL  {mod_path}.{obj_name}: {e}")
            continue
        if not hasattr(obj, attr):
            fails.append(f"MISSING ATTR {mod_path}.{obj_name}.{attr}")
        else:
            print(f"OK   {mod_path}.{obj_name}.{attr}")

    # Also check config files exist
    for p in [
        Path("config/agents.yaml"),
        Path("config/guardrails.yaml"),
        Path(".env"),
    ]:
        if p.exists():
            print(f"OK   file {p}")
        else:
            fails.append(f"MISSING FILE {p}")

    print()
    if fails:
        print(f"=== {len(fails)} FAILURES ===")
        for f in fails:
            print(" -", f)
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())