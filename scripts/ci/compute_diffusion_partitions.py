#!/usr/bin/env python3
"""
Compute dynamic partitions for diffusion CI tests.

This script is designed to run on lightweight CI runners (ubuntu-latest) without
any sglang dependencies. It uses AST parsing to extract test case information
from source files.

Usage:
    python scripts/ci/compute_diffusion_partitions.py --target-time 600
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List

from diffusion_case_parser import (
    BASELINE_REL_PATH,
    RUN_SUITE_REL_PATH,
    TESTCASE_CONFIG_REL_PATH,
    DiffusionCaseInfo,
    DiffusionSuiteInfo,
    collect_diffusion_suites,
)

# Mapping from internal suite name to output matrix name
SUITE_OUTPUT_NAMES = {
    "1-gpu": "1gpu",
    "2-gpu": "2gpu",
}


def compute_partition_count(
    cases: List[DiffusionCaseInfo],
    target_time_seconds: float,
    min_partitions: int = 1,
    max_partitions: int = 10,
) -> int:
    """
    Compute the number of partitions needed based on target time per partition.

    Args:
        cases: List of DiffusionCaseInfo objects
        target_time_seconds: Target time for each partition in seconds
        min_partitions: Minimum number of partitions
        max_partitions: Maximum number of partitions

    Returns:
        Number of partitions needed
    """
    if not cases:
        return 0

    total_time = sum(c.est_time for c in cases)

    if total_time <= 0:
        return min_partitions

    ideal_count = math.ceil(total_time / target_time_seconds)
    return max(min_partitions, min(ideal_count, max_partitions))


def lpt_partition(
    cases: List[DiffusionCaseInfo],
    num_partitions: int,
) -> List[List[DiffusionCaseInfo]]:
    """
    Partition cases using LPT (Longest Processing Time First) algorithm.

    Args:
        cases: List of DiffusionCaseInfo objects
        num_partitions: Number of partitions to create

    Returns:
        List of partitions, each containing DiffusionCaseInfo objects
    """
    if not cases or num_partitions <= 0:
        return []

    # Sort by time descending (LPT heuristic)
    sorted_cases = sorted(cases, key=lambda c: c.est_time, reverse=True)

    # Initialize partitions
    partitions: List[List[DiffusionCaseInfo]] = [[] for _ in range(num_partitions)]
    partition_sums = [0.0] * num_partitions

    # Greedy assignment: assign each case to partition with smallest current sum
    for case in sorted_cases:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append(case)
        partition_sums[min_idx] += case.est_time

    return partitions


def output_github_matrix(suite: str, partition_count: int):
    """
    Output GitHub Actions matrix format to GITHUB_OUTPUT.

    Uses 'include' structure to create paired values instead of Cartesian product.
    Uses compact JSON (no spaces) for consistent string matching in workflow conditions.

    Args:
        suite: Suite name for output (e.g., "1gpu", "2gpu")
        partition_count: Number of partitions
    """
    if partition_count <= 0:
        matrix = {"include": []}
    else:
        matrix = {
            "include": [
                {"part": i, "total": partition_count} for i in range(partition_count)
            ]
        }

    # Use compact JSON (no spaces) for consistent matching in workflow conditions
    matrix_json = json.dumps(matrix, separators=(",", ":"))

    # Output to GITHUB_OUTPUT if running in GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"matrix-{suite}={matrix_json}\n")

    # Also print to stdout for debugging
    print(f"matrix-{suite}={matrix_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute diffusion test partitions for CI"
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=1200.0,
        help="Target time per partition in seconds (default: 1200 = 20 minutes)",
    )
    parser.add_argument(
        "--min-partitions",
        type=int,
        default=2,
        help="Minimum number of partitions (default: 2)",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=10,
        help="Maximum number of partitions (default: 10)",
    )
    args = parser.parse_args()

    # Determine repository root (script is in scripts/ci/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # Load source files
    testcase_config_path = repo_root / TESTCASE_CONFIG_REL_PATH
    baseline_path = repo_root / BASELINE_REL_PATH
    run_suite_path = repo_root / RUN_SUITE_REL_PATH

    if not testcase_config_path.exists():
        print(f"Error: Testcase config not found: {testcase_config_path}")
        sys.exit(1)

    if not run_suite_path.exists():
        print(f"Error: Run suite not found: {run_suite_path}")
        sys.exit(1)

    # Collect all suite information using AST parsing
    suites = collect_diffusion_suites(
        testcase_config_path,
        run_suite_path,
        baseline_path,
    )

    # Print header
    print("=== Diffusion Partition Computation ===")
    print(
        f"Target time per partition: {args.target_time}s ({args.target_time/60:.1f} min)"
    )
    print()

    # Process each suite
    for suite_name, suite_info in suites.items():
        cases = suite_info.cases
        standalone_files = suite_info.standalone_files

        total_time = sum(c.est_time for c in cases)
        parametrized_partitions = compute_partition_count(
            cases,
            args.target_time,
            args.min_partitions,
            args.max_partitions,
        )

        # Add standalone file partitions
        num_standalone = len(standalone_files)
        total_partitions = parametrized_partitions + num_standalone

        # Print suite summary
        display_name = suite_name.upper()
        print(f"{display_name} suite:")
        print(f"  Cases: {len(cases)}")
        print(f"  Total estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Parametrized partitions: {parametrized_partitions}")
        print(f"  Standalone partitions: {num_standalone}")
        print(f"  Total partitions: {total_partitions}")
        print()

        # Show detailed partition assignments
        if parametrized_partitions > 0 and cases:
            partitions = lpt_partition(cases, parametrized_partitions)
            print("  Partition assignments (parametrized):")
            for i, partition in enumerate(partitions):
                partition_time = sum(c.est_time for c in partition)
                print(f"    Partition {i}:")
                print(
                    f"      Estimated time: {partition_time:.1f}s ({partition_time/60:.1f} min)"
                )
                print(f"      Cases ({len(partition)}):")
                for case in partition:
                    print(f"        - {case.case_id} ({case.est_time:.1f}s)")
            print()

        if num_standalone > 0:
            print("  Standalone partitions:")
            for i, filename in enumerate(standalone_files):
                partition_idx = parametrized_partitions + i
                print(f"    Partition {partition_idx}: {filename}")
            print()

        # Output GitHub Actions matrix
        output_name = SUITE_OUTPUT_NAMES.get(suite_name, suite_name.replace("-", ""))
        output_github_matrix(output_name, total_partitions)


if __name__ == "__main__":
    main()
