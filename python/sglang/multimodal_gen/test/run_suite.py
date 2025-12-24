"""
Test runner for multimodal_gen that manages test suites and parallel execution.

Uses LPT (Longest Processing Time First) algorithm for load-balanced partitioning
based on estimated test times from perf_baselines.json.

Usage:
    python3 run_suite.py --suite <suite_name> --partition-id <id> --total-partitions <num>

Example:
    python3 run_suite.py --suite 1-gpu --partition-id 0 --total-partitions 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import tabulate
from typing import List

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    ONE_GPU_CASES,
    TWO_GPU_CASES,
    DiffusionTestCase,
)

logger = init_logger(__name__)

# Suite definitions using DiffusionTestCase lists
SUITES = {
    "1-gpu": ONE_GPU_CASES,
    "2-gpu": TWO_GPU_CASES,
}

# Parametrized test files (use case ID filter)
PARAMETRIZED_FILES = {
    "1-gpu": ["test_server_1_gpu.py"],
    "2-gpu": ["test_server_2_gpu.py"],
}

# Standalone test files (each gets its own partition, no filter, run last)
STANDALONE_FILES = {
    "1-gpu": ["test_lora_format_adapter.py",
        # cli test
        "../cli/test_generate_t2i_perf.py"],
    "2-gpu": [],
}

# Default estimated time for cases without baseline (5 minutes)
DEFAULT_EST_TIME_SECONDS = 300.0
# Fixed overhead for server startup when estimated_full_test_time_s is not set
STARTUP_OVERHEAD_SECONDS = 120.0


def get_case_est_time(case_id: str) -> float:
    """
    Get estimated time in seconds from perf_baselines.json.

    Priority:
    1. estimated_full_test_time_s (if set)
    2. expected_e2e_ms / 1000 + STARTUP_OVERHEAD_SECONDS (fallback)
    3. DEFAULT_EST_TIME_SECONDS (if no baseline)
    """
    scenario = BASELINE_CONFIG.scenarios.get(case_id)
    if scenario is None:
        return DEFAULT_EST_TIME_SECONDS

    if scenario.estimated_full_test_time_s is not None:
        return scenario.estimated_full_test_time_s

    return scenario.expected_e2e_ms / 1000.0 + STARTUP_OVERHEAD_SECONDS


def auto_partition(
    cases: List[DiffusionTestCase],
    rank: int,
    size: int,
) -> List[DiffusionTestCase]:
    """
    Partition cases using LPT (Longest Processing Time First) greedy algorithm.

    This algorithm distributes cases across partitions to balance the total
    estimated time in each partition.

    Args:
        cases: List of DiffusionTestCase objects
        rank: Index of the partition to return (0 to size-1)
        size: Total number of partitions

    Returns:
        List of DiffusionTestCase objects assigned to the specified partition
    """
    if not cases or size <= 0:
        return []

    # Get estimated time for each case
    cases_with_time = [(case, get_case_est_time(case.id)) for case in cases]

    # Sort by time descending (LPT heuristic)
    sorted_cases = sorted(cases_with_time, key=lambda x: x[1], reverse=True)

    # Initialize partitions
    partitions: List[List[DiffusionTestCase]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedy assignment: assign each case to partition with smallest current sum
    for case, est_time in sorted_cases:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append(case)
        partition_sums[min_idx] += est_time

    if rank < size:
        return partitions[rank]
    return []


def get_parametrized_files(suite: str, target_dir: Path) -> List[str]:
    """
    Get parametrized test file paths for the specified suite.

    Args:
        suite: Suite name (e.g., "1-gpu", "2-gpu")
        target_dir: Base directory for test files

    Returns:
        List of absolute file paths
    """
    files = PARAMETRIZED_FILES.get(suite, [])
    result = []
    for f in files:
        f_path = target_dir / f
        if f_path.exists():
            result.append(str(f_path))
        else:
            logger.warning(f"Test file {f} not found in {target_dir}")
    return result


def get_standalone_file(suite: str, target_dir: Path, index: int) -> str | None:
    """
    Get a standalone test file path by index.

    Args:
        suite: Suite name (e.g., "1-gpu", "2-gpu")
        target_dir: Base directory for test files
        index: Index of the standalone file (0-based)

    Returns:
        Absolute file path, or None if not found
    """
    files = STANDALONE_FILES.get(suite, [])
    if index < 0 or index >= len(files):
        return None
    f_path = target_dir / files[index]
    if f_path.exists():
        return str(f_path)
    logger.warning(f"Standalone test file {files[index]} not found in {target_dir}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal_gen test suite")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=list(SUITES.keys()),
        help="The test suite to run (e.g., 1-gpu, 2-gpu)",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="Index of the current partition (for parallel execution)",
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="server",
        help="Base directory for tests relative to this script's parent",
    )
    parser.add_argument(
        "-k",
        "--filter",
        type=str,
        default=None,
        help="Pytest filter expression (passed to pytest -k)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (for CI consistency; pytest already continues by default)",
    )
    return parser.parse_args()


def collect_test_items(files, filter_expr=None):
    """Collect test item node IDs from the given files using pytest --collect-only.

    Raises:
        RuntimeError: If pytest collection fails due to errors (e.g., syntax errors,
            import errors, or other collection failures).
    """
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    cmd.extend(files)

    print(f"Collecting tests with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for collection errors
    # pytest exit codes:
    #   0: success
    #   1: tests collected but some had errors during collection
    #   2: test execution interrupted
    #   3: internal error
    #   4: command line usage error
    #   5: no tests collected (may be expected with filters)
    if result.returncode not in (0, 5):
        error_msg = (
            f"pytest --collect-only failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
        )
        if result.stderr:
            error_msg += f"stderr:\n{result.stderr}\n"
        if result.stdout:
            error_msg += f"stdout:\n{result.stdout}\n"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if result.returncode == 5:
        print(
            "No tests were collected (exit code 5). This may be expected with filters."
        )

    # Parse the output to extract test node IDs
    # pytest -q outputs lines like: test_file.py::TestClass::test_method[param]
    test_items = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        # Skip empty lines and summary lines
        if line and "::" in line and not line.startswith(("=", "-", " ")):
            # Handle lines that might have extra info after the test ID
            test_id = line.split()[0] if " " in line else line
            if "::" in test_id:
                test_items.append(test_id)

    print(f"Collected {len(test_items)} test items")
    return test_items


def run_pytest(files, filter_expr=None):
    if not files:
        print("No files to run.")
        return 0

    base_cmd = [sys.executable, "-m", "pytest", "-s", "-v"]

    # Add pytest -k filter if provided
    if filter_expr:
        base_cmd.extend(["-k", filter_expr])

    max_retries = 6
    # retry if the perf assertion failed, for {max_retries} times
    for i in range(max_retries + 1):
        cmd = list(base_cmd)
        if i > 0:
            cmd.append("--last-failed")
        else:
            cmd.extend(files)

        if i > 0:
            print(
                f"Performance assertion failed. Retrying ({i}/{max_retries}) with --last-failed..."
            )

        print(f"Running command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        output_bytes = bytearray()
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            output_bytes.extend(chunk)

        process.wait()
        returncode = process.returncode

        if returncode == 0:
            return 0

        # Exit code 5 means no tests were collected/selected - treat as success
        # when using filters, since some partitions may have all tests filtered out
        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            return 0

        # check if the failure is due to an assertion in test_server_utils.py
        full_output = output_bytes.decode("utf-8", errors="replace")
        is_perf_assertion = (
            "multimodal_gen/test/server/test_server_utils.py" in full_output
            and "AssertionError" in full_output
        )

        is_flaky_ci_assertion = (
            "SafetensorError" in full_output or "FileNotFoundError" in full_output
        )

        is_oom_error = (
            "out of memory" in full_output.lower()
            or "oom killer" in full_output.lower()
        )

        if not (is_perf_assertion or is_flaky_ci_assertion or is_oom_error):
            return returncode

    print(f"Max retry exceeded")
    return returncode


def main():
    args = parse_args()

    # 1. Resolve base path
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    # 2. Calculate partition allocation
    standalone_files = STANDALONE_FILES.get(args.suite, [])
    num_standalone = len(standalone_files)
    parametrized_partitions = args.total_partitions - num_standalone

    if parametrized_partitions < 0:
        print(
            f"Error: total_partitions ({args.total_partitions}) must be >= "
            f"standalone files ({num_standalone})"
        )
        sys.exit(1)

    # 3. Determine partition type and execute
    if args.partition_id < parametrized_partitions:
        # === Parametrized test partition ===
        all_cases = SUITES.get(args.suite, [])

        if not all_cases:
            print(f"No cases found for suite '{args.suite}'.")
            sys.exit(0)

        # Use LPT algorithm to partition cases among parametrized partitions
        my_cases = auto_partition(all_cases, args.partition_id, parametrized_partitions)

        if not my_cases:
            print(
                f"No cases assigned to partition {args.partition_id}. Exiting success."
            )
            sys.exit(0)

        # Print partition info
        print(
            f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions} (parametrized)"
        )
        print(f"Running {len(my_cases)} cases in this partition:")
        total_est_time = 0.0
        for case in my_cases:
            est = get_case_est_time(case.id)
            total_est_time += est
            print(f"  - {case.id} (est: {est:.1f}s)")
        print(
            f"Total estimated time: {total_est_time:.1f}s ({total_est_time/60:.1f} min)"
        )
        print()

        # Build pytest filter expression from case IDs
        case_ids = [case.id for case in my_cases]
        partition_filter = " or ".join([f"[{cid}]" for cid in case_ids])

        # Combine with additional filter if provided
        if args.filter:
            filter_expr = f"({partition_filter}) and ({args.filter})"
        else:
            filter_expr = partition_filter

        # Get parametrized test files
        suite_files = get_parametrized_files(args.suite, target_dir)

        if not suite_files:
            print(f"No valid parametrized test files found for suite '{args.suite}'.")
            sys.exit(0)

        print(f"Test files: {[os.path.basename(f) for f in suite_files]}")
        print(f"Filter expression: {filter_expr}")
        print()

        exit_code = run_pytest(suite_files, filter_expr=filter_expr)

    else:
        # === Standalone test partition ===
        standalone_idx = args.partition_id - parametrized_partitions

        print(
            f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions} (standalone)"
        )

        standalone_file = get_standalone_file(args.suite, target_dir, standalone_idx)

        if not standalone_file:
            print(
                f"No standalone file at index {standalone_idx} for suite '{args.suite}'."
            )
            sys.exit(0)

        print(f"Running standalone test file: {os.path.basename(standalone_file)}")
        print()

        # Run without case ID filter (standalone tests are not parametrized)
        exit_code = run_pytest([standalone_file], filter_expr=args.filter)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
