#!/usr/bin/env python3
"""
Compute dynamic partitions for diffusion CI tests.

This script runs on lightweight CI runners without sglang dependencies and uses
AST parsing to extract parametrized cases plus standalone files from source.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from diffusion_case_parser import (
    BASELINE_REL_PATH,
    RUN_SUITE_REL_PATH,
    DiffusionSuiteInfo,
    collect_diffusion_suites,
    resolve_case_config_path,
)


def _load_partitioning_helpers():
    repo_root = Path(__file__).resolve().parents[4]
    helper_path = repo_root / "python/sglang/multimodal_gen/test/partitioning.py"
    spec = importlib.util.spec_from_file_location(
        "diffusion_test_partitioning", helper_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PartitionItem, module.partition_items_by_lpt


PartitionItem, partition_items_by_lpt = _load_partitioning_helpers()

SUITE_OUTPUT_NAMES = {"1-gpu": "1gpu", "2-gpu": "2gpu", "1-gpu-b200": "b200"}
DEFAULT_STANDALONE_EST_TIME_SECONDS = 300.0


def validate_suite_case_coverage(suites: dict[str, DiffusionSuiteInfo]) -> None:
    """
    Guardrail: dynamic diffusion suites must contain parametrized cases.
    """
    suites_with_no_cases = []
    for suite_name in SUITE_OUTPUT_NAMES:
        suite_info = suites.get(suite_name)
        if suite_info is None:
            print(f"Error: Required suite '{suite_name}' not found in parsed suites.")
            sys.exit(1)
        if len(suite_info.cases) == 0:
            suites_with_no_cases.append(suite_name)

    if suites_with_no_cases:
        joined = ", ".join(suites_with_no_cases)
        print(
            "Error: Parsed zero parametrized cases for diffusion suites: "
            f"{joined}. This usually means run_suite case imports changed but "
            "diffusion parser logic was not updated."
        )
        sys.exit(1)


def compute_partition_count(
    total_time_seconds: float,
    min_time_seconds: float,
    target_time_seconds: float,
    max_time_seconds: float,
    max_partitions: int,
) -> int:
    if total_time_seconds <= 0:
        return 0

    min_partition_count = max(1, math.ceil(total_time_seconds / max_time_seconds))
    max_partition_count = max(1, math.floor(total_time_seconds / min_time_seconds))

    min_partition_count = min(min_partition_count, max_partitions)
    max_partition_count = min(max_partition_count, max_partitions)

    if max_partition_count < min_partition_count:
        fallback_count = math.ceil(total_time_seconds / target_time_seconds)
        return max(1, min(fallback_count, max_partitions))

    preferred_count = math.ceil(total_time_seconds / target_time_seconds)
    preferred_count = max(1, min(preferred_count, max_partitions))
    return max(min_partition_count, min(preferred_count, max_partition_count))


def build_partition_items(
    suite_info: DiffusionSuiteInfo, include_standalone: bool = True
) -> list[PartitionItem]:
    items = [
        PartitionItem(kind="case", item_id=case.case_id, est_time=case.est_time)
        for case in suite_info.cases
    ]
    if not include_standalone:
        return items

    items.extend(
        PartitionItem(
            kind="standalone",
            item_id=standalone_file,
            est_time=suite_info.standalone_est_times.get(
                standalone_file, DEFAULT_STANDALONE_EST_TIME_SECONDS
            ),
            used_fallback_estimate=(
                standalone_file in suite_info.missing_standalone_estimates
            ),
        )
        for standalone_file in suite_info.standalone_files
    )
    return items


def build_matrix(partition_count: int) -> dict:
    if partition_count <= 0:
        return {"include": []}
    return {"include": [{"part": i} for i in range(partition_count)]}


def build_partition_plan(
    suite_name: str,
    partitions: list[list[PartitionItem]],
) -> dict:
    return {
        "suite": suite_name,
        "partition_count": len(partitions),
        "partitions": [
            {
                "part": idx,
                "case_ids": [item.item_id for item in partition if item.kind == "case"],
                "standalone_files": [
                    item.item_id for item in partition if item.kind == "standalone"
                ],
                "missing_standalone_estimates": [
                    item.item_id
                    for item in partition
                    if item.kind == "standalone" and item.used_fallback_estimate
                ],
                "estimated_time": round(sum(item.est_time for item in partition), 1),
            }
            for idx, partition in enumerate(partitions)
        ],
    }


def output_github_value(name: str, value: dict) -> None:
    value_json = json.dumps(value, separators=(",", ":"))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{name}={value_json}\n")
    print(f"{name}={value_json}")


def output_github_scalar(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def print_suite_summary(
    suite_name: str,
    suite_info: DiffusionSuiteInfo,
    partitions: list[list[PartitionItem]],
    include_standalone: bool = True,
) -> None:
    total_time = sum(
        item.est_time
        for item in build_partition_items(
            suite_info, include_standalone=include_standalone
        )
    )
    print(f"{suite_name.upper()} suite:")
    print(f"  Cases: {len(suite_info.cases)}")
    standalone_label = "Standalone files"
    if not include_standalone:
        standalone_label = "Standalone files ignored"
    print(f"  {standalone_label}: {len(suite_info.standalone_files)}")
    print(
        f"  Missing standalone estimates: {len(suite_info.missing_standalone_estimates)}"
    )
    if suite_info.missing_standalone_estimates:
        print(
            f"  Fallback standalone estimate: "
            f"{DEFAULT_STANDALONE_EST_TIME_SECONDS:.1f}s"
        )
        for standalone_file in suite_info.missing_standalone_estimates:
            print(f"    - {standalone_file}")
    print(f"  Total estimated time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Selected partitions: {len(partitions)}")
    print()

    print("  Partition assignments:")
    for idx, partition in enumerate(partitions):
        partition_time = sum(item.est_time for item in partition)
        print(f"    Partition {idx}:")
        print(
            f"      Estimated time: {partition_time:.1f}s ({partition_time/60:.1f} min)"
        )
        for item in partition:
            fallback_suffix = (
                ", fallback estimate"
                if item.kind == "standalone" and item.used_fallback_estimate
                else ""
            )
            print(
                f"      - {item.kind}: {item.item_id} "
                f"({item.est_time:.1f}s{fallback_suffix})"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute diffusion test partitions for CI"
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=1200.0,
        help="Minimum desired partition time in seconds (default: 1200 = 20 minutes)",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=1800.0,
        help="Preferred partition time in seconds (default: 1800 = 30 minutes)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=2400.0,
        help="Maximum desired partition time in seconds (default: 2400 = 40 minutes)",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=10,
        help="Maximum number of partitions (default: 10)",
    )
    parser.add_argument(
        "--parametrized-only",
        action="store_true",
        help="Only partition DiffusionTestCase parametrized cases.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent.parent

    baseline_path = repo_root / BASELINE_REL_PATH
    run_suite_path = repo_root / RUN_SUITE_REL_PATH

    if not run_suite_path.exists():
        print(f"Error: Run suite not found: {run_suite_path}")
        sys.exit(1)
    try:
        case_config_path = resolve_case_config_path(repo_root, run_suite_path)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    suites = collect_diffusion_suites(
        case_config_path,
        run_suite_path,
        baseline_path,
    )
    validate_suite_case_coverage(suites)

    print("=== Diffusion Partition Computation ===")
    print(f"Min partition time: {args.min_time}s ({args.min_time/60:.1f} min)")
    print(f"Target partition time: {args.target_time}s ({args.target_time/60:.1f} min)")
    print(f"Max partition time: {args.max_time}s ({args.max_time/60:.1f} min)")
    print()

    for suite_name, suite_info in suites.items():
        if suite_name not in SUITE_OUTPUT_NAMES:
            continue

        items = build_partition_items(
            suite_info, include_standalone=not args.parametrized_only
        )
        total_time = sum(item.est_time for item in items)
        partition_count = compute_partition_count(
            total_time_seconds=total_time,
            min_time_seconds=args.min_time,
            target_time_seconds=args.target_time,
            max_time_seconds=args.max_time,
            max_partitions=args.max_partitions,
        )
        partitions = partition_items_by_lpt(items, partition_count)

        print_suite_summary(
            suite_name,
            suite_info,
            partitions,
            include_standalone=not args.parametrized_only,
        )

        output_name = SUITE_OUTPUT_NAMES[suite_name]
        output_github_value(f"matrix-{output_name}", build_matrix(partition_count))
        output_github_scalar(f"partition-count-{output_name}", str(partition_count))
        output_github_value(
            f"plan-{output_name}", build_partition_plan(suite_name, partitions)
        )


if __name__ == "__main__":
    main()
