"""
Compute dynamic partitions for diffusion CI tests.

This script calculates the optimal number of partitions based on estimated test times
and outputs GitHub Actions matrix format for dynamic job creation.

Usage:
    python -m sglang.multimodal_gen.test.compute_partitions --target-time 1200
"""

import argparse
import json
import math
import os
from typing import List

# Import case configurations
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import get_case_est_time


def compute_partition_count(
    cases: List[DiffusionTestCase],
    target_time_seconds: float,
    min_partitions: int = 1,
    max_partitions: int = 10,
) -> int:
    """
    Compute the number of partitions needed based on target time per partition.

    Args:
        cases: List of test cases
        target_time_seconds: Target time for each partition in seconds
        min_partitions: Minimum number of partitions
        max_partitions: Maximum number of partitions

    Returns:
        Number of partitions needed
    """
    if not cases:
        return 0

    total_time = sum(get_case_est_time(case.id) for case in cases)

    if total_time <= 0:
        return min_partitions

    ideal_count = math.ceil(total_time / target_time_seconds)
    return max(min_partitions, min(ideal_count, max_partitions))


def output_github_matrix(suite: str, partition_count: int):
    """
    Output GitHub Actions matrix format to GITHUB_OUTPUT.

    Uses 'include' structure to create paired values instead of Cartesian product.
    Uses compact JSON (no spaces) for consistent string matching in workflow conditions.

    Args:
        suite: Suite name (e.g., "1gpu", "2gpu")
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
        default=600.0,
        help="Target time per partition in seconds (default: 600 = 10 minutes)",
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

    # Define suites
    suites = {
        "1gpu": ONE_GPU_CASES_A + ONE_GPU_CASES_B,
        "2gpu": TWO_GPU_CASES_A + TWO_GPU_CASES_B,
    }

    # Print header
    print("=== Diffusion Partition Computation ===")
    print(
        f"Target time per partition: {args.target_time}s ({args.target_time/60:.1f} min)"
    )
    print()

    # Process each suite
    for suite_name, cases in suites.items():
        total_time = sum(get_case_est_time(case.id) for case in cases)
        num_partitions = compute_partition_count(
            cases,
            args.target_time,
            args.min_partitions,
            args.max_partitions,
        )

        # Print suite summary
        display_name = suite_name.upper().replace("GPU", "-GPU")
        print(f"{display_name} suite:")
        print(f"  Cases: {len(cases)}")
        print(f"  Total estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Partitions: {num_partitions}")
        print()

        # Output GitHub Actions matrix
        output_github_matrix(suite_name, num_partitions)


if __name__ == "__main__":
    main()
