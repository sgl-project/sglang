"""
Test runner for multimodal_gen MUSA suites that manages partitioned execution.

Usage:
    python3 -m sglang.multimodal_gen.test.server.musa.run_suite --suite <suite_name>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import tabulate

TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from runner.pytest_runner import (  # noqa: E402
    collect_test_items,
    partition_items_by_index,
    run_pytest,
)

SUITES = {
    "1-gpu-musa": [
        "test_server_1_gpu_musa.py",
    ],
    "1-gpu-musa-nightly": [
        "test_server_1_gpu_musa_nightly.py",
    ],
    "2-gpu-musa": [
        "test_server_2_gpu_musa.py",
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
        default=None,
        help=(
            "Base directory for tests relative to multimodal_gen/test. "
            "Defaults to server/musa."
        ),
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


def _resolve_suite_files(
    suite: str, test_root_dir: Path, musa_dir: Path, base_dir: str | None
) -> tuple[Path, list[str]]:
    target_dir = test_root_dir / base_dir if base_dir else musa_dir
    suite_files_rel = SUITES[suite]
    if target_dir == musa_dir:
        return target_dir, suite_files_rel

    musa_rel = musa_dir.relative_to(target_dir)
    return target_dir, [str(musa_rel / rel_path) for rel_path in suite_files_rel]


def main():
    args = parse_args()

    musa_dir = Path(__file__).resolve().parent
    test_root_dir = musa_dir.parent.parent
    target_dir, suite_files_rel = _resolve_suite_files(
        args.suite, test_root_dir, musa_dir, args.base_dir
    )

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

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

    my_items = partition_items_by_index(
        all_test_items, args.partition_id, args.total_partitions
    )

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
    exit_code, _, _ = run_pytest(my_items, exitfirst=not args.continue_on_error)

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
