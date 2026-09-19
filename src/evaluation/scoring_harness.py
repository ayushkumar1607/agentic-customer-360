"""Ground-truth scoring harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import ConfidenceCalibration, confusion
from src.utils.logger import get_logger

log = get_logger(__name__)


class ScoringHarness:
    def __init__(self, *, inferred_path: Path, ground_truth_path: Path) -> None:
        self.inferred_path = Path(inferred_path)
        self.gt_path = Path(ground_truth_path)

    # --- loading -----------------------------------------------------------

    def _load_inferred(self) -> list[dict[str, Any]]:
        if not self.inferred_path.exists():
            raise FileNotFoundError(self.inferred_path)
        out: list[dict[str, Any]] = []
        for line in self.inferred_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        log.info("harness.inferred_loaded", n=len(out))
        return out

    def _load_gt(self) -> dict[str, dict[str, Any]]:
        if not self.gt_path.exists():
            raise FileNotFoundError(self.gt_path)
        data = json.loads(self.gt_path.read_text(encoding="utf-8"))
        log.info("harness.gt_loaded", scenarios=list(data.keys()))
        return data

    # --- matching ----------------------------------------------------------

    @staticmethod
    def _match_by_customer(
        inferred: list[dict[str, Any]],
        gt: dict[str, dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        by_customer: dict[str, dict[str, Any]] = {}
        for cp in inferred:
            cid = cp.get("customer_id")
            if cid:
                by_customer[cid] = cp

        pairs: list[tuple[str, dict, dict]] = []
        for scenario_name, payload in gt.items():
            cid = payload.get("customer_id")
            if not cid:
                continue
            our_cp = by_customer.get(cid)
            if our_cp is None:
                log.warning(
                    "harness.no_inferred_for_customer",
                    customer_id=cid, scenario=scenario_name,
                )
                continue
            gt_cp = payload["checkpoints"][-1]
            pairs.append((scenario_name, our_cp, gt_cp))
        return pairs

    # --- scoring -----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        inferred = self._load_inferred()
        gt = self._load_gt()
        pairs = self._match_by_customer(inferred, gt)

        if not pairs:
            log.error("harness.no_pairs")
            return {"error": "no matched pairs"}

        expected_state, pred_state = [], []
        expected_action, pred_action = [], []
        expected_hitl, pred_hitl = [], []
        per_scenario: list[dict[str, Any]] = []
        conf_correct: dict[str, list[bool]] = {"high": [], "medium": [], "low": []}

        for scenario, ours, target in pairs:
            es = target["expected_state"]
            ea = target["expected_action"]
            eh = target["expected_hitl"]
            ps = ours.get("inferred_state")
            pa = ours.get("action")
            ph = ours.get("hitl_status")
            cb = ours.get("confidence_band", "low")

            expected_state.append(es)
            pred_state.append(ps)
            expected_action.append(ea)
            pred_action.append(pa)
            expected_hitl.append(eh)
            pred_hitl.append(ph)

            state_ok = (es == ps)
            action_ok = (ea == pa)
            hitl_ok = (eh == ph)
            all_ok = state_ok and action_ok and hitl_ok
            if cb in conf_correct:
                conf_correct[cb].append(all_ok)

            per_scenario.append({
                "scenario": scenario,
                "customer_id": ours.get("customer_id"),
                "expected": {"state": es, "action": ea, "hitl": eh},
                "predicted": {"state": ps, "action": pa, "hitl": ph},
                "state_match": state_ok,
                "action_match": action_ok,
                "hitl_match": hitl_ok,
                "all_match": all_ok,
                "confidence_band": cb,
                "notes": ours.get("notes"),
            })

        cm_state = confusion(
            field_name="inferred_state",
            expected=expected_state, predicted=pred_state,
        )
        cm_action = confusion(
            field_name="action",
            expected=expected_action, predicted=pred_action,
        )
        cm_hitl = confusion(
            field_name="hitl_status",
            expected=expected_hitl, predicted=pred_hitl,
        )

        calib = ConfidenceCalibration(
            high_total=len(conf_correct["high"]),
            high_correct=sum(conf_correct["high"]),
            medium_total=len(conf_correct["medium"]),
            medium_correct=sum(conf_correct["medium"]),
            low_total=len(conf_correct["low"]),
            low_correct=sum(conf_correct["low"]),
        )

        return {
            "n_scenarios": len(pairs),
            "per_scenario": per_scenario,
            "state_metrics": {
                "accuracy": round(cm_state.accuracy(), 4),
                "macro_f1": round(cm_state.macro_f1(), 4),
                "per_class": {
                    c: {
                        "precision": round(cm_state.precision(c), 4),
                        "recall": round(cm_state.recall(c), 4),
                        "f1": round(cm_state.f1(c), 4),
                    }
                    for c in cm_state.all_classes()
                },
            },
            "action_metrics": {
                "accuracy": round(cm_action.accuracy(), 4),
                "macro_f1": round(cm_action.macro_f1(), 4),
                "per_class": {
                    c: {
                        "precision": round(cm_action.precision(c), 4),
                        "recall": round(cm_action.recall(c), 4),
                        "f1": round(cm_action.f1(c), 4),
                    }
                    for c in cm_action.all_classes()
                },
            },
            "hitl_metrics": {
                "accuracy": round(cm_hitl.accuracy(), 4),
                "macro_f1": round(cm_hitl.macro_f1(), 4),
            },
            "confidence_calibration": calib.to_dict(),
            "overall_accuracy": round(
                sum(1 for p in per_scenario if p["all_match"]) / len(per_scenario),
                4,
            ),
        }

    # --- reporting ---------------------------------------------------------

    def print_report(self, report: dict[str, Any]) -> None:
        if "error" in report:
            print(f"\nERROR: {report['error']}\n")
            return

        print()
        print("=" * 78)
        print(" SCORING HARNESS — GROUND TRUTH EVALUATION")
        print("=" * 78)
        print(f" Scenarios evaluated: {report['n_scenarios']}")
        print(f" Overall accuracy: {report['overall_accuracy']:.2%}")
        print()
        print(" -- Per-scenario results --")
        for row in report["per_scenario"]:
            marker = "PASS" if row["all_match"] else "FAIL"
            print(f"  [{marker}] {row['scenario']} ({row['customer_id']})")
            print(f"     expected:  state={row['expected']['state']}, "
                  f"action={row['expected']['action']}, "
                  f"hitl={row['expected']['hitl']}")
            print(f"     predicted: state={row['predicted']['state']}, "
                  f"action={row['predicted']['action']}, "
                  f"hitl={row['predicted']['hitl']}")
            print(f"     conf_band={row['confidence_band']}")
        print()
        print(" -- Field accuracy --")
        print(f"  inferred_state : {report['state_metrics']['accuracy']:.2%} "
              f"(macro-F1 {report['state_metrics']['macro_f1']:.3f})")
        print(f"  action         : {report['action_metrics']['accuracy']:.2%} "
              f"(macro-F1 {report['action_metrics']['macro_f1']:.3f})")
        print(f"  hitl_status    : {report['hitl_metrics']['accuracy']:.2%} "
              f"(macro-F1 {report['hitl_metrics']['macro_f1']:.3f})")
        print()
        print(" -- Confidence calibration --")
        for band, stats in report["confidence_calibration"].items():
            print(f"  {band:>6}: n={stats['n']}, accuracy={stats['acc']:.2%}")
        print("=" * 78)
        print()

    @staticmethod
    def dump_json(report: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("harness.report_written", path=str(path))