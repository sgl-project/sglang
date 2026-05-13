"""
Test runner for multimodal_gen MUSA suites that manages partitioned execution.

Usage:
    python3 run_suite_musa.py --suite <suite_name> --partition-id <id> --total-partitions <num>

Example:
    python3 run_suite_musa.py --suite 1-gpu-musa --partition-id 0 --total-partitions 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import tabulate

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SUITES = {
    "1-gpu-musa": [
        "musa/test_server_a_musa.py",
        "musa/test_server_b_musa.py",
    ],
    "2-gpu-musa": [
        "musa/test_server_2_gpu_a_musa.py",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal_gen MUSA test suite")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=list(SUITES.keys()),
        help="The test suite to run (valid names are defined in SUITES)",
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
        help="Continue running remaining tests even if one fails.",
    )
    return parser.parse_args()


def collect_test_items(files, filter_expr=None):
    """Collect test item node IDs from the given files using pytest --collect-only."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    cmd.extend(files)

    print(f"Collecting tests with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

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

    test_items = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line and "::" in line and not line.startswith(("=", "-", " ")):
            test_id = line.split()[0] if " " in line else line
            if "::" in test_id:
                test_items.append(test_id)

    print(f"Collected {len(test_items)} test items")
    return test_items


def run_pytest(files, filter_expr=None, exitfirst=False):
    if not files:
        print("No files to run.")
        return 0

    base_cmd = [sys.executable, "-m", "pytest", "-s", "-v"]
    if exitfirst:
        base_cmd.append("-x")

    if filter_expr:
        base_cmd.extend(["-k", filter_expr])

    max_retries = 6
    for i in range(max_retries + 1):
        cmd = list(base_cmd)
        if i > 0:
            cmd.append("--last-failed")
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

        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            return 0

        full_output = output_bytes.decode("utf-8", errors="replace")
        is_perf_assertion = (
            "multimodal_gen/test/server/test_server_utils.py" in full_output
            and "AssertionError" in full_output
        )
        is_flaky_ci_assertion = (
            "SafetensorError" in full_output
            or "FileNotFoundError" in full_output
            or "TimeoutError" in full_output
        )
        is_oom_error = (
            "out of memory" in full_output.lower()
            or "oom killer" in full_output.lower()
        )

        if not (is_perf_assertion or is_flaky_ci_assertion or is_oom_error):
            return returncode

    print("Max retry exceeded")
    return returncode


def main():
    args = parse_args()

    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    suite_files_rel = SUITES[args.suite]
    suite_files_abs = []
    for rel_path in suite_files_rel:
        abs_path = target_dir / rel_path
        if not abs_path.exists():
            print(f"Warning: Test file {rel_path} not found in {target_dir}. Skipping.")
            continue
        suite_files_abs.append(str(abs_path))

    if not suite_files_abs:
        print(f"No valid test files found for suite '{args.suite}'.")
        sys.exit(0)

    all_test_items = collect_test_items(suite_files_abs, filter_expr=args.filter)
    if not all_test_items:
        print(f"No test items found for suite '{args.suite}'.")
        sys.exit(0)

    my_items = [
        item
        for i, item in enumerate(all_test_items)
        if i % args.total_partitions == args.partition_id
    ]

    partition_info = (
        f"{args.partition_id + 1}/{args.total_partitions} "
        f"(0-based id={args.partition_id})"
    )
    rows = [[args.suite, partition_info]]
    msg = (
        tabulate.tabulate(rows, headers=["Suite", "Partition"], tablefmt="psql") + "\n"
    )
    msg += f"Enabled {len(my_items)} test(s):\n"
    for item in my_items:
        msg += f"  - {item}\n"
    print(msg, flush=True)
    print(
        f"Suite: {args.suite} | Partition: {args.partition_id}/{args.total_partitions}"
    )
    print(f"Selected {len(suite_files_abs)} files:")
    for file_path in suite_files_abs:
        print(f"  - {os.path.basename(file_path)}")

    if not my_items:
        print("No items assigned to this partition. Exiting success.")
        sys.exit(0)

    print(f"Running {len(my_items)} items in this shard: {', '.join(my_items)}")
    exit_code = run_pytest(my_items, exitfirst=not args.continue_on_error)

    msg = (
        "\n"
        + tabulate.tabulate(rows, headers=["Suite", "Partition"], tablefmt="psql")
        + "\n"
    )
    msg += f"Executed {len(my_items)} test(s):\n"
    for item in my_items:
        msg += f"  - {item}\n"
    print(msg, flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
