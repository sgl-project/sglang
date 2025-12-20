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
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import get_case_est_time

logger = init_logger(__name__)

# Suite definitions using DiffusionTestCase lists
SUITES = {
    "1-gpu": ONE_GPU_CASES_A + ONE_GPU_CASES_B,
    "2-gpu": TWO_GPU_CASES_A + TWO_GPU_CASES_B,
}

# Mapping from suite to test files
SUITE_FILES = {
    "1-gpu": [
        "test_server_a.py",
        "test_server_b.py",
        "test_lora_format_adapter.py",
        # cli test
        "../cli/test_generate_t2i_perf.py",
    ],
    "2-gpu": [
        "test_server_2_gpu_a.py",
        "test_server_2_gpu_b.py",
    ],
}


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


def get_suite_files(suite: str, target_dir: Path) -> List[str]:
    """
    Get test file paths for the specified suite.

    Args:
        suite: Suite name (e.g., "1-gpu", "2-gpu")
        target_dir: Base directory for test files

    Returns:
        List of absolute file paths
    """
    files = SUITE_FILES.get(suite, [])
    result = []
    for f in files:
        f_path = target_dir / f
        if f_path.exists():
            result.append(str(f_path))
        else:
            logger.warning(f"Test file {f} not found in {target_dir}")
    return result


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

    # 1. Get all cases for the suite
    all_cases = SUITES.get(args.suite, [])

    if not all_cases:
        print(f"No cases found for suite '{args.suite}'.")
        sys.exit(0)

    # 2. Use LPT algorithm for partitioning
    my_cases = auto_partition(all_cases, args.partition_id, args.total_partitions)

    if not my_cases:
        print(f"No cases assigned to partition {args.partition_id}. Exiting success.")
        sys.exit(0)

    # 3. Print partition info
    print(
        f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions}"
    )
    print(f"Running {len(my_cases)} cases in this partition:")
    total_est_time = 0.0
    for case in my_cases:
        est = get_case_est_time(case.id)
        total_est_time += est
        print(f"  - {case.id} (est: {est:.1f}s)")
    print(f"Total estimated time: {total_est_time:.1f}s ({total_est_time/60:.1f} min)")
    print()

    # 4. Build pytest filter expression from case IDs
    case_ids = [case.id for case in my_cases]
    partition_filter = " or ".join(case_ids)

    # Combine with additional filter if provided
    if args.filter:
        filter_expr = f"({partition_filter}) and ({args.filter})"
    else:
        filter_expr = partition_filter

    # 5. Get test files for the suite
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    suite_files = get_suite_files(args.suite, target_dir)

    if not suite_files:
        print(f"No valid test files found for suite '{args.suite}'.")
        sys.exit(0)

    # 3. collect all test items and partition by items (not files)
    all_test_items = collect_test_items(suite_files_abs, filter_expr=args.filter)

    if not all_test_items:
        print(f"No test items found for suite '{args.suite}'.")
        sys.exit(0)

    # Partition by test items
    my_items = [
        item
        for i, item in enumerate(all_test_items)
        if i % args.total_partitions == args.partition_id
    ]

    # Print test info at beginning (similar to test/run_suite.py pretty_print_tests)
    partition_info = f"{args.partition_id + 1}/{args.total_partitions} (0-based id={args.partition_id})"
    headers = ["Suite", "Partition"]
    rows = [[args.suite, partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
    msg += f"✅ Enabled {len(my_items)} test(s):\n"
    for item in my_items:
        msg += f"  - {item}\n"
    print(msg, flush=True)
    print(
        f"Suite: {args.suite} | Partition: {args.partition_id}/{args.total_partitions}"
    )
    print(f"Selected {len(suite_files_abs)} files:")
    for f in suite_files_abs:
        print(f"  - {os.path.basename(f)}")

    if not my_items:
        print("No items assigned to this partition. Exiting success.")
        sys.exit(0)

    print(f"Running {len(my_items)} items in this shard: {', '.join(my_items)}")

    # 4. execute with the specific test items
    exit_code = run_pytest(my_items)

    # Print tests again at the end for visibility
    msg = "\n" + tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
    msg += f"✅ Executed {len(my_items)} test(s):\n"
    for item in my_items:
        msg += f"  - {item}\n"
    print(msg, flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
