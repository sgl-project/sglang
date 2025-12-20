#!/usr/bin/env python3
"""
Compute dynamic partitions for diffusion CI tests.

This script is designed to run on lightweight CI runners (ubuntu-latest) without
any sglang dependencies. It extracts test case information directly from source
files using regex and JSON parsing.

Usage:
    python scripts/ci/compute_diffusion_partitions.py --target-time 600
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

# Paths relative to repository root
TESTCASE_CONFIG_PATH = "python/sglang/multimodal_gen/test/server/testcase_configs.py"
BASELINE_PATH = "python/sglang/multimodal_gen/test/server/perf_baselines.json"

# Default estimated time for cases without baseline (5 minutes)
DEFAULT_EST_TIME_SECONDS = 300.0

# Mapping from suite name to list variable names in testcase_configs.py
SUITE_LISTS = {
    "1gpu": ["ONE_GPU_CASES_A", "ONE_GPU_CASES_B"],
    "2gpu": ["TWO_GPU_CASES_A", "TWO_GPU_CASES_B"],
}


def find_list_content(content: str, list_name: str) -> str:
    """
    Find the content of a list definition in Python source code.

    Args:
        content: Full source code content
        list_name: Name of the list variable (e.g., "ONE_GPU_CASES_A")

    Returns:
        String content between [ and matching ]
    """
    # Pattern to find the start of list definition
    pattern = rf"{list_name}\s*(?::\s*list\[DiffusionTestCase\])?\s*=\s*\["
    match = re.search(pattern, content)
    if not match:
        return ""

    # Find the matching closing bracket
    start_pos = match.end()
    bracket_count = 1
    pos = start_pos

    while pos < len(content) and bracket_count > 0:
        if content[pos] == "[":
            bracket_count += 1
        elif content[pos] == "]":
            bracket_count -= 1
        pos += 1

    return content[start_pos : pos - 1]


def extract_case_ids_from_list(content: str, list_name: str) -> List[str]:
    """
    Extract case IDs from a list definition in testcase_configs.py.

    Args:
        content: Full source code content
        list_name: Name of the list variable

    Returns:
        List of case ID strings
    """
    list_content = find_list_content(content, list_name)
    if not list_content:
        return []

    # Match DiffusionTestCase("case_id", ...) or DiffusionTestCase('case_id', ...)
    pattern = r'DiffusionTestCase\(\s*["\'](\w+)["\']'
    matches = re.findall(pattern, list_content)
    return matches


def load_baselines(path: Path) -> Dict[str, float]:
    """
    Load performance baselines from JSON file.

    Args:
        path: Path to perf_baselines.json

    Returns:
        Dictionary mapping case_id to estimated time in seconds
    """
    if not path.exists():
        print(f"Warning: Baseline file not found: {path}", file=sys.stderr)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = {}
    scenarios = data.get("scenarios", {})
    for case_id, scenario in scenarios.items():
        expected_e2e_ms = scenario.get("expected_e2e_ms", 0)
        baselines[case_id] = expected_e2e_ms / 1000.0  # Convert to seconds

    return baselines


def get_case_est_time(case_id: str, baselines: Dict[str, float]) -> float:
    """
    Get estimated time in seconds for a test case.

    Args:
        case_id: Test case ID
        baselines: Dictionary of baselines from load_baselines()

    Returns:
        Estimated time in seconds
    """
    return baselines.get(case_id, DEFAULT_EST_TIME_SECONDS)


def compute_partition_count(
    case_ids: List[str],
    baselines: Dict[str, float],
    target_time_seconds: float,
    min_partitions: int = 1,
    max_partitions: int = 10,
) -> int:
    """
    Compute the number of partitions needed based on target time per partition.

    Args:
        case_ids: List of test case IDs
        baselines: Dictionary of baselines from load_baselines()
        target_time_seconds: Target time for each partition in seconds
        min_partitions: Minimum number of partitions
        max_partitions: Maximum number of partitions

    Returns:
        Number of partitions needed
    """
    if not case_ids:
        return 0

    total_time = sum(get_case_est_time(cid, baselines) for cid in case_ids)

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

    # Determine repository root (script is in scripts/ci/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # Load source files
    testcase_config_path = repo_root / TESTCASE_CONFIG_PATH
    baseline_path = repo_root / BASELINE_PATH

    if not testcase_config_path.exists():
        print(f"Error: Testcase config not found: {testcase_config_path}")
        sys.exit(1)

    with open(testcase_config_path, "r", encoding="utf-8") as f:
        testcase_content = f.read()

    baselines = load_baselines(baseline_path)

    # Print header
    print("=== Diffusion Partition Computation ===")
    print(
        f"Target time per partition: {args.target_time}s ({args.target_time/60:.1f} min)"
    )
    print()

    # Process each suite
    for suite_name, list_names in SUITE_LISTS.items():
        # Extract case IDs from all lists for this suite
        case_ids = []
        for list_name in list_names:
            ids = extract_case_ids_from_list(testcase_content, list_name)
            case_ids.extend(ids)

        total_time = sum(get_case_est_time(cid, baselines) for cid in case_ids)
        num_partitions = compute_partition_count(
            case_ids,
            baselines,
            args.target_time,
            args.min_partitions,
            args.max_partitions,
        )

        # Print suite summary
        display_name = suite_name.upper().replace("GPU", "-GPU")
        print(f"{display_name} suite:")
        print(f"  Cases: {len(case_ids)}")
        print(f"  Total estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Partitions: {num_partitions}")
        print()

        # Output GitHub Actions matrix
        output_github_matrix(suite_name, num_partitions)


if __name__ == "__main__":
    main()
