#!/usr/bin/env python3
"""Validate eval scores against minimum thresholds.

Reads lm-eval results JSON files in PWD and checks that scored metrics meet
the per-task threshold from `thresholds.json` (or a fallback flat threshold).

Usage:
    python3 utils/evals/validate_scores.py
    python3 utils/evals/validate_scores.py --thresholds my_thresholds.json
    python3 utils/evals/validate_scores.py --min-score 0.90
"""

import argparse
import glob
import json
import sys
from pathlib import Path


def load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval scores")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.85,
        help="Fallback minimum score when no threshold config matches (default: 0.85)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to thresholds JSON config (default: utils/evals/thresholds.json)",
    )
    parser.add_argument(
        "--metric-prefix",
        default="exact_match,",
        help="Only check metrics whose name starts with this prefix (default: 'exact_match,')",
    )
    parser.add_argument(
        "--results-glob",
        default="results*.json",
        help="Glob pattern for result files (default: 'results*.json')",
    )
    args = parser.parse_args()

    thresholds: dict[str, float] = {}
    thresholds_path = args.thresholds
    if thresholds_path is None:
        default_path = Path(__file__).parent / "thresholds.json"
        if default_path.exists():
            thresholds_path = str(default_path)
    if thresholds_path:
        try:
            thresholds = load_thresholds(thresholds_path)
            print(f"Loaded thresholds from {thresholds_path}")
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"WARN: could not load thresholds from {thresholds_path}: {e}",
                file=sys.stderr,
            )

    failed = False
    checked = 0

    for f in sorted(glob.glob(args.results_glob)):
        with open(f) as fh:
            data = json.load(fh)
        for task, metrics in data.get("results", {}).items():
            min_score = thresholds.get(task, args.min_score)
            for name, val in metrics.items():
                if not name.startswith(args.metric_prefix) or "stderr" in name:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                checked += 1
                if val < min_score:
                    print(
                        f"FAIL: {task} {name} = {val:.4f} (< {min_score})",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(f"PASS: {task} {name} = {val:.4f} (>= {min_score})")

    if checked == 0:
        print(
            "WARN: no metrics matched prefix '{}'".format(args.metric_prefix),
            file=sys.stderr,
        )

    return 1 if (failed or checked == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
