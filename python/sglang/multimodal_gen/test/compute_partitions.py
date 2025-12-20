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
    BASELINE_CONFIG,
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)

DEFAULT_EST_TIME_SECONDS = 300.0  # 5 minutes default for cases without baseline


def get_case_est_time(case_id: str) -> float:
    """
    Get estimated time in seconds from perf_baselines.json.
    Returns default value if case has no baseline.
    """
    scenario = BASELINE_CONFIG.scenarios.get(case_id)
    if scenario is None:
        return DEFAULT_EST_TIME_SECONDS
    return scenario.expected_e2e_ms / 1000.0


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

    Args:
        suite: Suite name (e.g., "1gpu", "2gpu")
        partition_count: Number of partitions
    """
    if partition_count <= 0:
        matrix = {"part": [], "total": []}
    else:
        matrix = {
            "part": list(range(partition_count)),
            "total": [partition_count] * partition_count,
        }

    matrix_json = json.dumps(matrix)

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

    # 1-GPU cases
    one_gpu_cases = ONE_GPU_CASES_A + ONE_GPU_CASES_B
    one_gpu_total_time = sum(get_case_est_time(case.id) for case in one_gpu_cases)
    one_gpu_partitions = compute_partition_count(
        one_gpu_cases,
        args.target_time,
        args.min_partitions,
        args.max_partitions,
    )

    # 2-GPU cases
    two_gpu_cases = TWO_GPU_CASES_A + TWO_GPU_CASES_B
    two_gpu_total_time = sum(get_case_est_time(case.id) for case in two_gpu_cases)
    two_gpu_partitions = compute_partition_count(
        two_gpu_cases,
        args.target_time,
        args.min_partitions,
        args.max_partitions,
    )

    # Print summary
    print("=== Diffusion Partition Computation ===")
    print(
        f"Target time per partition: {args.target_time}s ({args.target_time/60:.1f} min)"
    )
    print()
    print("1-GPU suite:")
    print(f"  Cases: {len(one_gpu_cases)}")
    print(
        f"  Total estimated time: {one_gpu_total_time:.1f}s ({one_gpu_total_time/60:.1f} min)"
    )
    print(f"  Partitions: {one_gpu_partitions}")
    print()
    print("2-GPU suite:")
    print(f"  Cases: {len(two_gpu_cases)}")
    print(
        f"  Total estimated time: {two_gpu_total_time:.1f}s ({two_gpu_total_time/60:.1f} min)"
    )
    print(f"  Partitions: {two_gpu_partitions}")
    print()

    # Output GitHub Actions matrix
    output_github_matrix("1gpu", one_gpu_partitions)
    output_github_matrix("2gpu", two_gpu_partitions)


if __name__ == "__main__":
    main()
