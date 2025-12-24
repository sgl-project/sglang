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
RUN_SUITE_PATH = "python/sglang/multimodal_gen/test/run_suite.py"

# Default estimated time for cases without baseline (5 minutes)
DEFAULT_EST_TIME_SECONDS = 300.0
# Fixed overhead for server startup when estimated_full_test_time_s is not set
STARTUP_OVERHEAD_SECONDS = 120.0

# Mapping from suite name to list variable names in testcase_configs.py
SUITE_LISTS = {
    "1gpu": ["ONE_GPU_CASES"],
    "2gpu": ["TWO_GPU_CASES"],
}

# Mapping from suite name in this script to suite name in run_suite.py
SUITE_NAME_MAP = {
    "1gpu": "1-gpu",
    "2gpu": "2-gpu",
}


def extract_standalone_files_dict_content(content: str) -> str:
    """
    Extract the STANDALONE_FILES dictionary content from run_suite.py.

    Args:
        content: Full source code content of run_suite.py

    Returns:
        String content of the STANDALONE_FILES dictionary
    """
    pattern = r"STANDALONE_FILES\s*=\s*\{"
    match = re.search(pattern, content)
    if not match:
        return ""

    start_pos = match.end()
    brace_count = 1
    pos = start_pos

    while pos < len(content) and brace_count > 0:
        if content[pos] == "{":
            brace_count += 1
        elif content[pos] == "}":
            brace_count -= 1
        pos += 1

    return content[start_pos : pos - 1]


def extract_standalone_files_list(content: str, suite_key: str) -> List[str]:
    """
    Extract the list of standalone files for a suite from run_suite.py.

    Args:
        content: Full source code content of run_suite.py
        suite_key: Suite key as used in run_suite.py (e.g., "1-gpu", "2-gpu")

    Returns:
        List of standalone file names
    """
    dict_content = extract_standalone_files_dict_content(content)
    if not dict_content:
        return []

    suite_pattern = rf'"{suite_key}"\s*:\s*\[([^\]]*)\]'
    suite_match = re.search(suite_pattern, dict_content)
    if not suite_match:
        return []

    list_content = suite_match.group(1).strip()
    if not list_content:
        return []

    # Extract quoted strings (file names)
    file_pattern = r'["\']([^"\']+)["\']'
    files = re.findall(file_pattern, list_content)
    return files


def extract_standalone_files_count(content: str, suite_key: str) -> int:
    """
    Extract the count of standalone files for a suite from run_suite.py.

    Args:
        content: Full source code content of run_suite.py
        suite_key: Suite key as used in run_suite.py (e.g., "1-gpu", "2-gpu")

    Returns:
        Number of standalone files for the suite
    """
    return len(extract_standalone_files_list(content, suite_key))


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
    pattern = "DiffusionTestCase\\(\\s*[\"']([^\"']+)[\"']"

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

        if (
            "estimated_full_test_time_s" in scenario
            and scenario["estimated_full_test_time_s"] is not None
        ):
            baselines[case_id] = scenario["estimated_full_test_time_s"]
        else:

            expected_e2e_ms = scenario.get("expected_e2e_ms", 0)
            baselines[case_id] = expected_e2e_ms / 1000.0 + STARTUP_OVERHEAD_SECONDS

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


def lpt_partition(
    case_ids: List[str],
    baselines: Dict[str, float],
    num_partitions: int,
) -> List[List[tuple]]:
    """
    Partition cases using LPT (Longest Processing Time First) algorithm.

    Args:
        case_ids: List of test case IDs
        baselines: Dictionary of baselines from load_baselines()
        num_partitions: Number of partitions to create

    Returns:
        List of partitions, each containing tuples of (case_id, estimated_time)
    """
    if not case_ids or num_partitions <= 0:
        return []

    # Get estimated time for each case
    cases_with_time = [(cid, get_case_est_time(cid, baselines)) for cid in case_ids]

    # Sort by time descending (LPT heuristic)
    sorted_cases = sorted(cases_with_time, key=lambda x: x[1], reverse=True)

    # Initialize partitions
    partitions: List[List[tuple]] = [[] for _ in range(num_partitions)]
    partition_sums = [0.0] * num_partitions

    # Greedy assignment: assign each case to partition with smallest current sum
    for case_id, est_time in sorted_cases:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append((case_id, est_time))
        partition_sums[min_idx] += est_time

    return partitions


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
    testcase_config_path = repo_root / TESTCASE_CONFIG_PATH
    baseline_path = repo_root / BASELINE_PATH
    run_suite_path = repo_root / RUN_SUITE_PATH

    if not testcase_config_path.exists():
        print(f"Error: Testcase config not found: {testcase_config_path}")
        sys.exit(1)

    if not run_suite_path.exists():
        print(f"Error: Run suite not found: {run_suite_path}")
        sys.exit(1)

    with open(testcase_config_path, "r", encoding="utf-8") as f:
        testcase_content = f.read()

    with open(run_suite_path, "r", encoding="utf-8") as f:
        run_suite_content = f.read()

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
        parametrized_partitions = compute_partition_count(
            case_ids,
            baselines,
            args.target_time,
            args.min_partitions,
            args.max_partitions,
        )

        # Add standalone file partitions (extracted from run_suite.py)
        run_suite_key = SUITE_NAME_MAP.get(suite_name, suite_name)
        num_standalone = extract_standalone_files_count(
            run_suite_content, run_suite_key
        )
        total_partitions = parametrized_partitions + num_standalone

        # Print suite summary
        display_name = suite_name.upper().replace("GPU", "-GPU")
        print(f"{display_name} suite:")
        print(f"  Cases: {len(case_ids)}")
        print(f"  Total estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Parametrized partitions: {parametrized_partitions}")
        print(f"  Standalone partitions: {num_standalone}")
        print(f"  Total partitions: {total_partitions}")
        print()

        # Show detailed partition assignments
        if parametrized_partitions > 0 and case_ids:
            partitions = lpt_partition(case_ids, baselines, parametrized_partitions)
            print("  Partition assignments (parametrized):")
            for i, partition in enumerate(partitions):
                partition_time = sum(t for _, t in partition)
                print(f"    Partition {i}:")
                print(
                    f"      Estimated time: {partition_time:.1f}s ({partition_time/60:.1f} min)"
                )
                print(f"      Cases ({len(partition)}):")
                for case_id, est_time in partition:
                    print(f"        - {case_id} ({est_time:.1f}s)")
            print()

        if num_standalone > 0:
            standalone_files = extract_standalone_files_list(
                run_suite_content, run_suite_key
            )
            print("  Standalone partitions:")
            for i, filename in enumerate(standalone_files):
                partition_idx = parametrized_partitions + i
                print(f"    Partition {partition_idx}: {filename}")
            print()

        # Output GitHub Actions matrix
        output_github_matrix(suite_name, total_partitions)


if __name__ == "__main__":
    main()
