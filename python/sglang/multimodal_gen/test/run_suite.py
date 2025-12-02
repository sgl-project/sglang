"""
Test runner for multimodal_gen that manages test suites and parallel execution.

Usage:
    python3 run_suite.py --suite <suite_name> --partition-id <id> --total-partitions <num>

Example:
    python3 run_suite.py --suite 1-gpu --partition-id 0 --total-partitions 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SUITES = {
    "1-gpu": [
        "test_server_a.py",
        "test_server_b.py",
        # add new 1-gpu test files here
    ],
    "2-gpu": [
        "test_server_2_gpu_a.py",
        "test_server_2_gpu_b.py",
        # add new 2-gpu test files here
    ],
}


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
    return parser.parse_args()


def run_pytest(files):
    if not files:
        print("No files to run.")
        return 0

    base_cmd = [sys.executable, "-m", "pytest", "-s", "-v", "--log-cli-level=INFO"]

    max_retries = 2
    # retry if the perf assertion failed, for {max_retries} times
    for i in range(max_retries + 1):
        cmd = list(base_cmd)
        if i > 0:
            cmd.append("--last-failed")
        cmd.extend(files)

        if i > 0:
            logger.info(
                f"Performance assertion failed. Retrying ({i}/{max_retries}) with --last-failed..."
            )

        logger.info(f"Running command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                output_lines.append(line)

        returncode = process.poll()

        if returncode == 0:
            return 0

        # check if the failure is due to an assertion in test_server_utils.py
        full_output = "".join(output_lines)
        is_perf_assertion = (
            "multimodal_gen/test/server/test_server_utils.py" in full_output
            and "AssertionError" in full_output
        )

        if not is_perf_assertion:
            return returncode

    return returncode


def main():
    args = parse_args()

    # 1. resolve base path
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    # 2. get files from suite
    suite_files_rel = SUITES[args.suite]

    suite_files_abs = []
    for f_rel in suite_files_rel:
        f_abs = target_dir / f_rel
        if not f_abs.exists():
            print(f"Warning: Test file {f_rel} not found in {target_dir}. Skipping.")
            continue
        suite_files_abs.append(str(f_abs))

    if not suite_files_abs:
        print(f"No valid test files found for suite '{args.suite}'.")
        sys.exit(0)

    # 3. partitioning
    my_files = [
        f
        for i, f in enumerate(suite_files_abs)
        if i % args.total_partitions == args.partition_id
    ]

    print(
        f"Suite: {args.suite} | Partition: {args.partition_id}/{args.total_partitions}"
    )
    print(f"Selected {len(my_files)} files:")
    for f in my_files:
        print(f"  - {os.path.basename(f)}")

    if not my_files:
        print("No files assigned to this partition. Exiting success.")
        sys.exit(0)

    # 4. execute
    exit_code = run_pytest(my_files)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
