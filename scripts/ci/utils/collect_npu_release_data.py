#!/usr/bin/env python3
"""Consolidate NPU key-model performance/accuracy metrics into release data.

The multi-node NPU test jobs persist one ``metrics.json`` per case under
``/root/.cache/tests/output/...`` and upload the case directory as a workflow
artifact. This script merges the downloaded artifacts into a single JSON file
and a Markdown table for the release run summary.

Usage:
    python3 scripts/ci/utils/collect_npu_release_data.py \
        --input-dir release-data \
        --output-json release-data.json \
        --output-markdown release-data.md \
        --run-id 12345678 \
        --commit-sha abc123def
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def load_case_metrics(filepath: str) -> dict | None:
    """Load one per-case metrics.json written by dump_metric."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to read {filepath}: {e}")
        return None

    test_case = data.get("test_case")
    if not test_case:
        print(f"Warning: {filepath} has no test_case, skipped")
        return None

    return {
        "test_case": test_case,
        "test_type": data.get("test_type", "unknown"),
        "metrics": data.get("metrics") or {},
        "baselines": data.get("baselines") or {},
    }


def collect_cases(input_dir: str) -> list[dict]:
    """Collect one entry per (test_case, test_type), recursively."""
    cases = {}
    pattern = os.path.join(input_dir, "**", "metrics.json")
    for filepath in sorted(glob.glob(pattern, recursive=True)):
        case = load_case_metrics(filepath)
        if case is None:
            continue
        key = (case["test_case"], case["test_type"])
        if key in cases:
            print(f"Warning: duplicate metrics for {key}, using {filepath}")
        cases[key] = case
    return [cases[key] for key in sorted(cases)]


def render_markdown(cases: list[dict]) -> str:
    """Render a per-metric table for GITHUB_STEP_SUMMARY."""
    lines = ["# NPU Key Model Release Data", ""]
    if not cases:
        lines.append("No metrics were collected.")
        return "\n".join(lines) + "\n"

    lines.append("| Test case | Type | Metric | Value | Baseline |")
    lines.append("| --- | --- | --- | --- | --- |")
    for case in cases:
        metrics = case["metrics"]
        if not metrics:
            lines.append(f"| {case['test_case']} | {case['test_type']} | - | - | - |")
            continue
        for metric_name in sorted(metrics):
            baseline = case["baselines"].get(metric_name, "-")
            lines.append(
                f"| {case['test_case']} | {case['test_type']} | "
                f"{metric_name} | {metrics[metric_name]} | {baseline} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing downloaded metrics artifacts",
    )
    parser.add_argument("--output-json", required=True, help="Consolidated JSON path")
    parser.add_argument(
        "--output-markdown", required=True, help="Markdown summary path"
    )
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--commit-sha", default="", help="Tested commit SHA")
    parser.add_argument("--branch", default="", help="Tested branch or tag")

    args = parser.parse_args()

    cases = collect_cases(args.input_dir)
    payload = {
        "source": "npu-key-model-release",
        "run_id": args.run_id,
        "commit_sha": args.commit_sha,
        "branch": args.branch,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    markdown = render_markdown(cases)
    with open(args.output_markdown, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Collected {len(cases)} case(s) from {args.input_dir}")
    print(f"Wrote {args.output_json} and {args.output_markdown}")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
