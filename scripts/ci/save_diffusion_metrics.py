#!/usr/bin/env python3
"""Collect and save diffusion performance metrics for artifact collection in CI.

This script reads diffusion test results from the pytest stash and saves them
with metadata for the performance dashboard.

Usage:
    python3 scripts/ci/save_diffusion_metrics.py \
        --gpu-config 1-gpu-runner \
        --run-id 12345678 \
        --output test/diffusion-metrics-1gpu.json \
        --results-json test/diffusion-results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def load_diffusion_results(results_file: str) -> list[dict]:
    """Load diffusion performance results from JSON file."""
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return []

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to parse {results_file}: {e}")
        return []


def transform_diffusion_result(result: dict, gpu_config: str) -> dict:
    """Transform a diffusion result to match dashboard expectations.

    Dashboard expects:
    - Separate test_name, class_name
    - Numeric metrics in consistent units
    - Optional modality field
    """
    return {
        "test_name": result.get("test_name"),
        "class_name": result.get("class_name"),
        "modality": result.get("modality", "image"),
        "e2e_ms": result.get("e2e_ms"),
        "avg_denoise_ms": result.get("avg_denoise_ms"),
        "median_denoise_ms": result.get("median_denoise_ms"),
        "stage_metrics": result.get("stage_metrics", {}),
        "sampled_steps": result.get("sampled_steps", {}),
        # Video-specific metrics (if present)
        "frames_per_second": result.get("frames_per_second"),
        "total_frames": result.get("total_frames"),
        "avg_frame_time_ms": result.get("avg_frame_time_ms"),
    }


def group_results_by_class(results: list[dict], gpu_config: str) -> list[dict]:
    """Group diffusion results by test class (suite).

    Returns list with one entry per test class, containing all tests in that class.
    """
    groups = {}

    for result in results:
        class_name = result.get("class_name", "unknown")

        if class_name not in groups:
            groups[class_name] = {
                "gpu_config": gpu_config,
                "test_suite": class_name,
                "tests": [],
            }

        transformed = transform_diffusion_result(result, gpu_config)
        groups[class_name]["tests"].append(transformed)

    return list(groups.values())


def save_metrics(
    gpu_config: str,
    run_id: str,
    output_file: str,
    results_file: str,
) -> bool:
    """Collect diffusion metrics and save to output file."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Load diffusion results
    raw_results = load_diffusion_results(results_file)
    print(f"Loaded {len(raw_results)} diffusion test result(s)")

    # Group by test class
    grouped = group_results_by_class(raw_results, gpu_config)

    # Create metrics structure
    metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "gpu_config": gpu_config,
        "test_type": "diffusion",
        "results": grouped,
    }

    # Ensure output directory exists and write output
    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if not raw_results:
            print(f"Created empty metrics file: {output_file}")
        else:
            print(f"Saved diffusion metrics to: {output_file}")
        return True
    except OSError as e:
        print(f"Error writing metrics file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Collect diffusion performance metrics from test results"
    )
    parser.add_argument(
        "--gpu-config",
        required=True,
        help="GPU configuration (e.g., 1-gpu-runner, 2-gpu-runner)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for metrics JSON",
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to diffusion results JSON file",
    )

    args = parser.parse_args()

    success = save_metrics(
        gpu_config=args.gpu_config,
        run_id=args.run_id,
        output_file=args.output,
        results_file=args.results_json,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
