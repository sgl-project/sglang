#!/usr/bin/env python3
"""
Generate diffusion CI outputs for consistency testing.

This script reuses the CI test code by running pytest with SGLANG_GEN_GT=1,
ensuring that GT generation uses exactly the same code path as CI tests.

Usage:
    python gen_diffusion_ci_outputs.py --suite 1-gpu --partition-id 0 --total-partitions 2 --out-dir ./output
    python gen_diffusion_ci_outputs.py --suite 1-gpu --case-ids qwen_image_t2i flux_image_t2i --out-dir ./output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)

logger = init_logger(__name__)


def _get_cases_for_suite(
    suite: str, case_ids: list[str] | None = None
) -> list[DiffusionTestCase]:
    """Get test cases for the specified suite, optionally filtered by case IDs."""
    if suite == "1-gpu":
        all_cases = ONE_GPU_CASES_A + ONE_GPU_CASES_B
    elif suite == "2-gpu":
        all_cases = TWO_GPU_CASES_A + TWO_GPU_CASES_B
    else:
        raise ValueError(f"Invalid suite: {suite}. Must be '1-gpu' or '2-gpu'")

    # Deduplicate by case.id
    seen: set[str] = set()
    deduplicated: list[DiffusionTestCase] = []
    for c in all_cases:
        if c.id not in seen:
            seen.add(c.id)
            deduplicated.append(c)

    # Filter by case_ids if provided
    if case_ids is not None and len(case_ids) > 0:
        case_id_set = set(case_ids)
        filtered_cases = [c for c in deduplicated if c.id in case_id_set]
        if len(filtered_cases) == 0:
            logger.warning(f"No matching cases found for provided case IDs: {case_ids}")
        missing_ids = case_id_set - {c.id for c in filtered_cases}
        if missing_ids:
            logger.warning(f"Some case IDs not found: {missing_ids}")
        return filtered_cases

    return deduplicated


def _build_pytest_filter(case_ids: list[str]) -> str:
    """Build pytest -k filter expression for specific case IDs."""
    # pytest parametrized test format: test_diffusion_generation[case_id]
    # We need to match any of the case IDs
    filters = [f"test_diffusion_generation[{case_id}]" for case_id in case_ids]
    return " or ".join(filters)


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

    # Get cases
    all_cases = _get_cases_for_suite(
        args.suite, args.case_ids if args.case_ids else None
    )

    # Apply partition filtering if specified
    if args.partition_id is not None and args.total_partitions is not None:
        my_cases = [
            c
            for i, c in enumerate(all_cases)
            if i % args.total_partitions == args.partition_id
        ]
        logger.info(
            f"Partition {args.partition_id}/{args.total_partitions}: "
            f"running {len(my_cases)} of {len(all_cases)} cases"
        )
    else:
        my_cases = all_cases
        logger.info(f"Running {len(my_cases)} cases")

    if len(my_cases) == 0:
        logger.warning("No cases to run")
        return

    # Build pytest filter expression
    case_ids = [case.id for case in my_cases]
    pytest_filter = _build_pytest_filter(case_ids)

    # Determine test files based on suite
    # For 1-gpu: test_server_a.py (ONE_GPU_CASES_A) and test_server_b.py (ONE_GPU_CASES_B)
    # For 2-gpu: test_server_2_gpu_a.py (TWO_GPU_CASES_A) and test_server_2_gpu_b.py (TWO_GPU_CASES_B)
    if args.suite == "1-gpu":
        test_files = [
            "python/sglang/multimodal_gen/test/server/test_server_a.py",
            "python/sglang/multimodal_gen/test/server/test_server_b.py",
        ]
    else:  # 2-gpu
        test_files = [
            "python/sglang/multimodal_gen/test/server/test_server_2_gpu_a.py",
            "python/sglang/multimodal_gen/test/server/test_server_2_gpu_b.py",
        ]

    # Build pytest command
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",  # Don't capture output
        "-v",  # Verbose
        "-k",
        pytest_filter,
    ] + test_files

    # Set environment variables for GT generation mode
    env = os.environ.copy()
    env["SGLANG_GEN_GT"] = "1"
    env["SGLANG_GT_OUTPUT_DIR"] = str(out_dir.absolute())
    env["SGLANG_SKIP_CONSISTENCY"] = "1"  # Skip consistency checks in GT gen mode

    logger.info(f"Running pytest with filter: {pytest_filter}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Command: {' '.join(pytest_cmd)}")

    # Run pytest
    try:
        result = subprocess.run(
            pytest_cmd,
            env=env,
            check=not args.continue_on_error,
        )
        if result.returncode != 0:
            if args.continue_on_error:
                logger.warning(f"pytest exited with code {result.returncode}")
            else:
                sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        if args.continue_on_error:
            logger.error(f"pytest failed: {e}", exc_info=True)
        else:
            raise


if __name__ == "__main__":
    main()
