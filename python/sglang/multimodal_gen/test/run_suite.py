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
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    ONE_GPU_CASES,
    TWO_GPU_CASES,
    DiffusionTestCase,
)

logger = init_logger(__name__)

# Suite definitions
SUITES = {
    "1-gpu": ONE_GPU_CASES,
    "2-gpu": TWO_GPU_CASES,
}

# Default estimated time for cases without baseline (5 minutes)
DEFAULT_EST_TIME_SECONDS = 300.0

# Fixed overhead for server startup when estimated_full_test_time_s is not set
STARTUP_OVERHEAD_SECONDS = 120.0

# Parametrized test files (use case ID filter)
PARAMETRIZED_FILES = {
    "1-gpu": ["test_server_1_gpu.py"],
    "2-gpu": ["test_server_2_gpu.py"],
}

# Standalone test files (each gets its own partition, no filter, run last)
# NOTE: This is parsed by diffusion_case_parser.py using AST
STANDALONE_FILES = {
    "1-gpu": [
        "test_lora_format_adapter.py",
        # cli test
        "../cli/test_generate_t2i_perf.py",
    ],
    "2-gpu": [],
}


def get_case_est_time(case_id: str) -> float:
    """
    Get estimated time in seconds from perf_baselines.json.
    Returns default value if case has no baseline.
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

    # Sort by time descending (LPT heuristic)
    sorted_cases = sorted(cases, key=lambda c: get_case_est_time(c.id), reverse=True)

    # Initialize partitions
    partitions: List[List[DiffusionTestCase]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedy assignment: assign each case to partition with smallest current sum
    for case in sorted_cases:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append(case)
        partition_sums[min_idx] += get_case_est_time(case.id)

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


def get_standalone_file(
    standalone_files: List[str], target_dir: Path, index: int
) -> str | None:
    """
    Get a standalone test file path by index.

    Args:
        standalone_files: List of standalone file names
        target_dir: Base directory for test files
        index: Index of the standalone file (0-based)

    Returns:
        Absolute file path, or None if not found
    """
    if index < 0 or index >= len(standalone_files):
        return None
    f_path = target_dir / standalone_files[index]
    if f_path.exists():
        return str(f_path)
    logger.warning(
        f"Standalone test file {standalone_files[index]} not found in {target_dir}"
    )
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal_gen test suite")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=["1-gpu", "2-gpu"],
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


def parse_junit_xml_for_executed_cases(xml_path: str) -> list[str]:
    """
    Parse JUnit XML to extract case IDs that were actually executed (not skipped).

    Returns:
        List of case IDs that were executed (passed or failed, not skipped).
    """
    if not Path(xml_path).exists():
        return []

    executed_cases = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for testcase in root.iter("testcase"):
        # Check if test was skipped
        if testcase.find("skipped") is not None:
            continue

        name = testcase.get("name", "")
        # Extract case ID from test name like "test_diffusion_server[qwen_image_t2i]"
        if "[" in name and "]" in name:
            case_id = name[name.index("[") + 1 : name.index("]")]
            executed_cases.append(case_id)

    return executed_cases


def write_execution_report(
    suite: str,
    partition_id: int,
    total_partitions: int,
    executed_cases: list[str],
    is_standalone: bool = False,
    standalone_file: str | None = None,
) -> str:
    """
    Write execution report to JSON file.

    Returns:
        Path to the generated report file.
    """
    report = {
        "suite": suite,
        "partition_id": partition_id,
        "total_partitions": total_partitions,
        "is_standalone": is_standalone,
        "standalone_file": standalone_file,
        "executed_cases": executed_cases,
    }

    report_filename = f"execution_report_{suite}_{partition_id}.json"
    report_path = Path(__file__).parent / report_filename

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Execution report written to: {report_path}")
    return str(report_path)


def run_pytest(files, filter_expr=None, junit_xml_path=None) -> tuple[int, list[str]]:
    """
    Run pytest with retry logic for flaky tests.

    Returns:
        Tuple of (exit_code, executed_cases) where executed_cases is a list of
        case IDs that were executed across all retry attempts.
    """
    if not files:
        print("No files to run.")
        return (0, [])

    # Accumulate executed cases across all retry attempts
    all_executed_cases: set[str] = set()

    base_cmd = [sys.executable, "-m", "pytest", "-s", "-v"]

    # Add JUnit XML output for coverage tracking
    if junit_xml_path:
        base_cmd.extend(["--junit-xml", junit_xml_path])

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

        # Accumulate executed cases from this run before checking return code
        if junit_xml_path:
            cases_this_run = parse_junit_xml_for_executed_cases(junit_xml_path)
            all_executed_cases.update(cases_this_run)

        if returncode == 0:
            return (0, list(all_executed_cases))

        # Exit code 5 means no tests were collected/selected - treat as success
        # when using filters, since some partitions may have all tests filtered out
        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            return (0, list(all_executed_cases))

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
            return (returncode, list(all_executed_cases))

    logger.info("Max retry exceeded")
    return (returncode, list(all_executed_cases))


def main():
    args = parse_args()

    # 1. Resolve base path
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    # 2. Get test cases from imported configuration
    all_cases = SUITES.get(args.suite)
    if all_cases is None:
        print(f"Unknown suite: {args.suite}")
        sys.exit(1)

    standalone_files = STANDALONE_FILES.get(args.suite, [])

    # 3. Calculate partition allocation
    num_standalone = len(standalone_files)
    parametrized_partitions = args.total_partitions - num_standalone

    if parametrized_partitions < 0:
        print(
            f"Error: total_partitions ({args.total_partitions}) must be >= "
            f"standalone files ({num_standalone})"
        )
        sys.exit(1)

    # 4. Determine partition type and execute
    if args.partition_id < parametrized_partitions:
        # === Parametrized test partition ===
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
            est_time = get_case_est_time(case.id)
            total_est_time += est_time
            print(f"  - {case.id} (est: {est_time:.1f}s)")
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

        junit_xml_path = str(target_dir / "junit_results.xml")
        exit_code, executed_cases = run_pytest(
            suite_files, filter_expr=filter_expr, junit_xml_path=junit_xml_path
        )

        # Generate execution report (executed_cases already collected from run_pytest)
        write_execution_report(
            suite=args.suite,
            partition_id=args.partition_id,
            total_partitions=args.total_partitions,
            executed_cases=executed_cases,
            is_standalone=False,
        )

    else:
        # === Standalone test partition ===
        standalone_idx = args.partition_id - parametrized_partitions

        print(
            f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions} (standalone)"
        )

        standalone_file = get_standalone_file(
            standalone_files, target_dir, standalone_idx
        )

        if not standalone_file:
            print(
                f"No standalone file at index {standalone_idx} for suite '{args.suite}'."
            )
            sys.exit(0)

        print(f"Running standalone test file: {os.path.basename(standalone_file)}")
        print()

        # Run without case ID filter (standalone tests are not parametrized)
        junit_xml_path = str(target_dir / "junit_results.xml")
        exit_code, _ = run_pytest(
            [standalone_file], filter_expr=args.filter, junit_xml_path=junit_xml_path
        )

        # Generate execution report for standalone
        standalone_filename = standalone_files[standalone_idx]
        write_execution_report(
            suite=args.suite,
            partition_id=args.partition_id,
            total_partitions=args.total_partitions,
            executed_cases=[],  # Standalone doesn't have parametrized cases
            is_standalone=True,
            standalone_file=standalone_filename,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
