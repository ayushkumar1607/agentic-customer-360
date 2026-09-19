"""
Run the scoring harness against our inferred-events.jsonl.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --jsonl output/inferred-events.jsonl \
                                     --gt data/ground_truth/sample_ground_truth.json
    python -m scripts.run_evaluation --out output/evaluation_report.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- defensive path setup ---------------------------------------------------
# Ensures `from src.X import Y` works whether invoked as `python -m scripts.X`
# or as `python scripts/run_evaluation.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.scoring_harness import ScoringHarness   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score our inferred-events against a ground-truth file."
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path("output/inferred-events.jsonl"),
        help="Our inferred-events.jsonl (default: output/inferred-events.jsonl)",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=Path("data/ground_truth/sample_ground_truth.json"),
        help="Ground truth JSON (default: data/ground_truth/sample_ground_truth.json)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output/evaluation_report.json"),
        help="Where to write the JSON report",
    )
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: inferred-events file not found: {args.jsonl}", file=sys.stderr)
        print("Run the pipeline first:", file=sys.stderr)
        print("    python -m scripts.run_full_pipeline --all", file=sys.stderr)
        sys.exit(1)

    if not args.gt.exists():
        print(f"ERROR: ground-truth file not found: {args.gt}", file=sys.stderr)
        print("Expected a JSON file with per-scenario expected_state/action/hitl.",
              file=sys.stderr)
        sys.exit(1)

    harness = ScoringHarness(
        inferred_path=args.jsonl,
        ground_truth_path=args.gt,
    )
    report = harness.run()
    harness.print_report(report)
    harness.dump_json(report, args.out)

    print(f"JSON report written to: {args.out}")
    print()


if __name__ == "__main__":
    main()