#!/usr/bin/env python3
"""Collect and save performance metrics from nightly benchmark results.

This script reads benchmark result JSON files from performance profile directories
and saves them with metadata for artifact collection in CI.

Usage:
    python3 scripts/ci/save_metrics.py \
        --gpu-config 8-gpu-h200 \
        --partition 0 \
        --run-id 12345678 \
        --output test/metrics-8gpu-h200-partition-0.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def find_result_files(search_dirs: list[str]) -> list[str]:
    """Find all results_*.json files in the given directories."""
    result_files = set()
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "**/results_*.json")
            result_files.update(glob.glob(pattern, recursive=True))
    return list(result_files)


def parse_result_file(filepath: str) -> list[dict]:
    """Parse a benchmark result JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
        return []


def transform_benchmark_result(result: dict, gpu_config: str, partition: int) -> dict:
    """Transform a benchmark result to the metrics schema."""
    # Handle None values safely for numeric conversions
    latency = result.get("latency")
    last_ttft = result.get("last_ttft")

    return {
        "batch_size": result.get("batch_size"),
        "input_len": result.get("input_len"),
        "output_len": result.get("output_len"),
        "latency_ms": latency * 1000 if latency is not None else None,
        "input_throughput": result.get("input_throughput"),
        "output_throughput": result.get("output_throughput"),
        "overall_throughput": result.get("overall_throughput"),
        "ttft_ms": last_ttft * 1000 if last_ttft is not None else None,
        "acc_length": result.get("acc_length"),
    }


def group_results_by_model(
    results: list[dict], gpu_config: str, partition: int
) -> list[dict]:
    """Group benchmark results by model, variant, and server_args."""
    groups = {}

    for result in results:
        model_path = result.get("model_path", "unknown")
        run_name = result.get("run_name", "default")
        variant = run_name if run_name != "default" else None
        server_args = result.get("server_args")
        # Convert server_args list to tuple for use as dict key (lists are not hashable)
        server_args_key = tuple(server_args) if server_args else None

        key = (model_path, variant, server_args_key)
        if key not in groups:
            groups[key] = {
                "gpu_config": gpu_config,
                "partition": partition,
                "model": model_path,
                "variant": variant,
                "server_args": server_args,
                "benchmarks": [],
            }

        groups[key]["benchmarks"].append(
            transform_benchmark_result(result, gpu_config, partition)
        )

    return list(groups.values())


def save_metrics(
    gpu_config: str,
    partition: int,
    run_id: str,
    output_file: str,
    search_dirs: list[str],
) -> bool:
    """Collect metrics and save to output file."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Find all result files
    result_files = find_result_files(search_dirs)
    print(f"Found {len(result_files)} result file(s)")

    grouped = []
    if not result_files:
        print("No benchmark result files found")
    else:
        # Parse all result files
        all_results = []
        for filepath in sorted(result_files):
            print(f"  Reading: {filepath}")
            results = parse_result_file(filepath)
            all_results.extend(results)
        print(f"Total benchmark results: {len(all_results)}")

        # Group by model/variant
        grouped = group_results_by_model(all_results, gpu_config, partition)

    # Create metrics structure
    metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "gpu_config": gpu_config,
        "partition": partition,
        "results": grouped,
    }

    # Ensure output directory exists and write output
    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if not result_files:
            print(f"Created empty metrics file: {output_file}")
        else:
            print(f"Saved metrics to: {output_file}")
        return True
    except OSError as e:
        print(f"Error writing metrics file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Collect performance metrics from benchmark results"
    )
    parser.add_argument(
        "--gpu-config",
        required=True,
        help="GPU configuration (e.g., 8-gpu-h200, 8-gpu-b200)",
    )
    parser.add_argument(
        "--partition",
        type=int,
        required=True,
        help="Partition number (0, 1, 2, etc.)",
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
        "--search-dir",
        action="append",
        default=[],
        dest="search_dirs",
        help="Directory to search for result files (can be specified multiple times)",
    )

    args = parser.parse_args()

    # Default search directories if none specified
    search_dirs = args.search_dirs or [
        "test/performance_profiles_8_gpu",
        "test/performance_profiles_text_models",
        "test/performance_profiles_vlms",
        "test",
        ".",
    ]

    success = save_metrics(
        gpu_config=args.gpu_config,
        partition=args.partition,
        run_id=args.run_id,
        output_file=args.output,
        search_dirs=search_dirs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
