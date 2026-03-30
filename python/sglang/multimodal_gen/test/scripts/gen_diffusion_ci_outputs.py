#!/usr/bin/env python3
"""
Generate diffusion CI outputs for consistency testing.

This script reuses the CI test code by calling run_suite.py with SGLANG_GEN_GT=1,
ensuring that GT generation uses exactly the same code path as CI tests.

Usage:
    python gen_diffusion_ci_outputs.py --suite 1-gpu --partition-id 0 --total-partitions 2 --out-dir ./output
    python gen_diffusion_ci_outputs.py --suite 1-gpu --case-ids qwen_image_t2i flux_image_t2i --out-dir ./output
"""

import argparse
import os
import sys
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.run_suite import SUITES, collect_test_items, run_pytest

logger = init_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate diffusion CI outputs")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["1-gpu", "2-gpu"],
        required=True,
        help="Test suite to run (1-gpu or 2-gpu)",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        required=False,
        help="Partition ID for matrix partitioning (0-based)",
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        required=False,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other cases if one fails",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        nargs="*",
        required=False,
        help="Specific case IDs to run (space-separated). If provided, only these cases will be run.",
    )

    args = parser.parse_args()

    # Validate partition arguments
    if args.partition_id is not None and args.total_partitions is not None:
        if args.partition_id < 0 or args.partition_id >= args.total_partitions:
            parser.error(f"partition-id must be in range [0, {args.total_partitions})")
    elif args.partition_id is not None or args.total_partitions is not None:
        parser.error(
            "Both --partition-id and --total-partitions must be provided together"
        )

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for GT generation mode
    os.environ["SGLANG_GEN_GT"] = "1"
    os.environ["SGLANG_GT_OUTPUT_DIR"] = str(out_dir.absolute())
    os.environ["SGLANG_SKIP_CONSISTENCY"] = (
        "1"  # Skip consistency checks in GT gen mode
    )

    logger.info(f"GT generation mode enabled")
    logger.info(f"Output directory: {out_dir}")

    # Resolve test files path (same as run_suite.py)
    current_file_path = Path(__file__).resolve()
    test_root_dir = current_file_path.parent.parent  # scripts -> test
    target_dir = test_root_dir / "server"

    # Get files from suite (same as run_suite.py)
    suite_files_rel = SUITES[args.suite]
    suite_files_abs = []
    for f_rel in suite_files_rel:
        f_abs = target_dir / f_rel
        if not f_abs.exists():
            logger.warning(f"Test file {f_rel} not found in {target_dir}. Skipping.")
            continue
        suite_files_abs.append(str(f_abs))

    if not suite_files_abs:
        logger.error(f"No valid test files found for suite '{args.suite}'.")
        sys.exit(1)

    # Build pytest filter for case_ids if provided
    filter_expr = None
    if args.case_ids:
        # pytest parametrized test format: test_diffusion_generation[case_id]
        filters = [f"test_diffusion_generation[{case_id}]" for case_id in args.case_ids]
        filter_expr = " or ".join(filters)
        logger.info(f"Filtering by case IDs: {args.case_ids}")

    # Collect all test items (same as run_suite.py)
    all_test_items = collect_test_items(suite_files_abs, filter_expr=filter_expr)

    if not all_test_items:
        logger.warning(f"No test items found for suite '{args.suite}'.")
        sys.exit(0)

    # Partition by test items (same as run_suite.py)
    partition_id = args.partition_id if args.partition_id is not None else 0
    total_partitions = args.total_partitions if args.total_partitions is not None else 1

    my_items = [
        item
        for i, item in enumerate(all_test_items)
        if i % total_partitions == partition_id
    ]

    logger.info(
        f"Partition {partition_id}/{total_partitions}: "
        f"running {len(my_items)} of {len(all_test_items)} test items"
    )

    if not my_items:
        logger.warning("No items assigned to this partition. Exiting success.")
        sys.exit(0)

    # Run pytest with the specific test items (same as run_suite.py)
    exit_code = run_pytest(my_items)

    if exit_code != 0:
        if args.continue_on_error:
            logger.warning(f"pytest exited with code {exit_code}")
        else:
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
