"""
Helper script to discover and run multimodal generation tests

This script scans the `multimodal_gen/test/server/` directory for test files matching a pattern and executes a subset of them based on the partition ID

How to add a new test:
1. Create a new test file in `python/sglang/multimodal_gen/test/server/`.
2. Name it matching the pattern `test_server_*.py` (e.g., `test_server_c.py`).
3. The CI will automatically pick it up and distribute it to one of the runners.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal_gen test suite")
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
        "--target-dir",
        type=str,
        default="server",
        help="Sub-directory under multimodal_gen/test to look for tests",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="test_server_*.py",
        help="Glob pattern to match test files",
    )
    return parser.parse_args()


def get_test_files(base_dir, sub_dir, pattern):
    """Find all test files matching the pattern."""
    search_path = os.path.join(base_dir, sub_dir, pattern)
    files = sorted(glob.glob(search_path))
    return files


def main():
    args = parse_args()

    # Determine the absolute path of the test directory
    # Assuming this script is located at python/sglang/multimodal_gen/test/run_suite.py
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent

    # 1. Discover Test Files
    all_files = get_test_files(str(test_root_dir), args.target_dir, args.pattern)

    if not all_files:
        print(
            f"No test files found in {os.path.join(test_root_dir, args.target_dir)} matching {args.pattern}"
        )
        sys.exit(0)  # Exit gracefully if no files found

    print(f"Found {len(all_files)} test files total.")

    # 2. Partitioning (Distribute files across runners)
    # Using simple interleaving: file 0 -> runner 0, file 1 -> runner 1, file 2 -> runner 0...
    my_files = [
        f
        for i, f in enumerate(all_files)
        if i % args.total_partitions == args.partition_id
    ]

    if not my_files:
        print(
            f"Partition {args.partition_id}/{args.total_partitions} has no files to run. Skipping."
        )
        sys.exit(0)

    print(f"Running {len(my_files)} files on this partition:")
    for f in my_files:
        print(f"  - {os.path.basename(f)}")

    # 3. Execute Pytest
    # Construct the pytest command
    # -s: show stdout
    # -v: verbose
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",
        "--log-cli-level=INFO",
    ] + my_files

    print(f"Executing command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
