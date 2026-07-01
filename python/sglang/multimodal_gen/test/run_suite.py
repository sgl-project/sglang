"""
Test runner for multimodal_gen that manages test suites and parallel execution.

For diffusion 1-gpu/2-gpu suites, cases are partitioned by estimated runtime
using LPT so each CI shard has a similar total runtime.
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tabulate

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.partitioning import (
    PartitionItem,
    partition_items_by_lpt,
)
from sglang.multimodal_gen.test.runner.pytest_runner import (
    partition_items_by_index,
    run_pytest,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DiffusionTestCase,
)

# TODO: remove duplicated code
if current_platform.is_npu():
    from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import (
        _UPDATE_WEIGHTS_FROM_DISK_TEST_FILE,
        COMPONENT_ACCURACY_SUITES,
        DEFAULT_EST_TIME_SECONDS,
        DEFAULT_STANDALONE_EST_TIME_SECONDS,
        FILE_SUITES,
        PARAMETRIZED_CASE_GROUPS,
        STANDALONE_FILES,
        STARTUP_OVERHEAD_SECONDS,
        SUITES,
    )
else:
    from sglang.multimodal_gen.test.server.gpu_cases import (  # noqa: F401 It is used by ci scripts
        _UPDATE_WEIGHTS_FROM_DISK_TEST_FILE,
        _UPDATE_WEIGHTS_MODEL_PAIR_ENV,
        _UPDATE_WEIGHTS_MODEL_PAIR_IDS,
        COMPONENT_ACCURACY_FILE_NUM_GPUS,
        COMPONENT_ACCURACY_SUITES,
        DEFAULT_EST_TIME_SECONDS,
        DEFAULT_STANDALONE_EST_TIME_SECONDS,
        FILE_SUITES,
        ONE_GPU_CASES,
        PARAMETRIZED_CASE_GROUPS,
        STANDALONE_FILE_EST_TIMES,
        STANDALONE_FILES,
        STARTUP_OVERHEAD_SECONDS,
        STRICT_SUITES,
        SUITES,
        TWO_GPU_CASES,
    )


logger = init_logger(__name__)


@dataclass(frozen=True)
class PartitionAssignment:
    case_ids: list[str]
    standalone_files: list[str]
    estimated_time: float | None = None
    missing_standalone_estimates: list[str] | None = None


def get_case_est_time(case_id: str) -> float:
    scenario = BASELINE_CONFIG.scenarios.get(case_id)
    if scenario is None:
        return DEFAULT_EST_TIME_SECONDS
    if scenario.estimated_full_test_time_s is not None:
        return scenario.estimated_full_test_time_s
    return scenario.expected_e2e_ms / 1000.0 + STARTUP_OVERHEAD_SECONDS


def get_standalone_file_est_time(
    suite: str, standalone_file: str
) -> tuple[float, bool]:
    suite_est_times = STANDALONE_FILE_EST_TIMES.get(suite, {})
    if standalone_file not in suite_est_times:
        return DEFAULT_STANDALONE_EST_TIME_SECONDS, True
    return suite_est_times[standalone_file], False


def get_all_standalone_file_est_times() -> dict[str, dict[str, float]]:
    return copy.deepcopy(STANDALONE_FILE_EST_TIMES)


def validate_standalone_file_est_times() -> dict[str, list[str]]:
    missing_by_suite: dict[str, list[str]] = {}
    for suite, standalone_files in STANDALONE_FILES.items():
        suite_est_times = STANDALONE_FILE_EST_TIMES.get(suite, {})
        missing = [
            standalone_file
            for standalone_file in standalone_files
            if standalone_file not in suite_est_times
        ]
        if missing:
            missing_by_suite[suite] = missing
    return missing_by_suite


def auto_partition(
    cases: list[DiffusionTestCase], rank: int, size: int
) -> list[DiffusionTestCase]:
    if not cases or size <= 0:
        return []

    case_by_id = {case.id: case for case in cases}
    items = [
        PartitionItem(kind="case", item_id=case.id, est_time=get_case_est_time(case.id))
        for case in cases
    ]
    partitions = partition_items_by_lpt(items, size)
    if rank >= len(partitions):
        return []
    return [case_by_id[item.item_id] for item in partitions[rank]]


def get_suite_files_rel(suite: str, parametrized_only: bool = False) -> list[str]:
    if parametrized_only and suite in PARAMETRIZED_CASE_GROUPS:
        return [filename for filename, _ in PARAMETRIZED_CASE_GROUPS[suite]]
    return SUITES[suite]


def _normalize_standalone_key(standalone_file: str) -> str:
    return f"standalone:{standalone_file}"


def parse_partition_plan(
    suite: str,
    partition_id: int,
    total_partitions: int,
    plan_json: str,
) -> PartitionAssignment:
    plan = json.loads(plan_json)
    if plan.get("suite") != suite:
        raise ValueError(
            f"Partition plan suite mismatch: expected {suite!r}, "
            f"got {plan.get('suite')!r}"
        )

    partition_count = plan.get("partition_count")
    if partition_count != total_partitions:
        raise ValueError(
            f"Partition count mismatch for suite {suite!r}: "
            f"plan={partition_count}, matrix={total_partitions}"
        )

    partitions = plan.get("partitions", [])
    selected_partition = None
    for partition in partitions:
        if partition.get("part") == partition_id:
            selected_partition = partition
            break

    if selected_partition is None:
        raise ValueError(
            f"Partition {partition_id} not found in plan for suite {suite!r}"
        )

    return PartitionAssignment(
        case_ids=list(selected_partition.get("case_ids", [])),
        standalone_files=list(selected_partition.get("standalone_files", [])),
        estimated_time=selected_partition.get("estimated_time"),
        missing_standalone_estimates=list(
            selected_partition.get("missing_standalone_estimates", [])
        ),
    )


def _merge_execution_results(
    executed_cases: list[str],
    case_results: dict[str, str],
    new_executed_cases: list[str],
    new_case_results: dict[str, str],
) -> None:
    executed_cases.extend(
        case_id for case_id in new_executed_cases if case_id not in executed_cases
    )
    case_results.update(new_case_results)


def _format_standalone_estimate_snippet(
    suite: str, standalone_file: str, measured_full_test_time_s: float
) -> str:
    return (
        f'"{suite}": {{\n'
        f'    "{standalone_file}": {measured_full_test_time_s:.1f},\n'
        f"}}"
    )


def _print_missing_standalone_estimate_message(
    suite: str,
    standalone_file: str,
    measured_full_test_time_s: float,
) -> None:
    snippet = _format_standalone_estimate_snippet(
        suite, standalone_file, measured_full_test_time_s
    )
    logger.error(
        f'\n{"=" * 60}\n'
        f'Add standalone estimate for suite "{suite}" and file "{standalone_file}":\n\n'
        f"File: python/sglang/multimodal_gen/test/run_suite.py\n\n"
        f"Current partition used fallback estimate: "
        f"{DEFAULT_STANDALONE_EST_TIME_SECONDS:.1f}s\n\n"
        f"{snippet}\n"
        f'{"=" * 60}\n'
    )


def _run_standalone_file(
    suite: str,
    standalone_rel: str,
    target_dir: Path,
    extra_filter: str | None = None,
) -> tuple[int, list[str], dict[str, str], dict]:
    if standalone_rel == _UPDATE_WEIGHTS_FROM_DISK_TEST_FILE:
        _maybe_pin_update_weights_model_pair([standalone_rel])

    est_time, used_fallback_estimate = get_standalone_file_est_time(
        suite, standalone_rel
    )
    standalone_file = _resolve_suite_files(target_dir, [standalone_rel], strict=True)[0]
    junit_xml_path = str(
        target_dir / f"junit_results_{suite}_{Path(standalone_rel).stem}.xml"
    )
    start_time = time.perf_counter()
    exit_code, _, _ = run_pytest(
        [standalone_file],
        filter_expr=extra_filter,
        junit_xml_path=junit_xml_path,
    )
    measured_full_test_time_s = round(time.perf_counter() - start_time, 1)
    standalone_key = _normalize_standalone_key(standalone_rel)
    measurement = {
        "suite": suite,
        "standalone_file": standalone_rel,
        "measured_full_test_time_s": measured_full_test_time_s,
        "used_fallback_estimate": used_fallback_estimate,
        "fallback_estimate_s": DEFAULT_STANDALONE_EST_TIME_SECONDS,
        "had_configured_estimate": not used_fallback_estimate,
        "configured_or_fallback_estimate_s": est_time,
    }
    if used_fallback_estimate:
        _print_missing_standalone_estimate_message(
            suite, standalone_rel, measured_full_test_time_s
        )
    return (
        exit_code,
        [standalone_key],
        {standalone_key: "pass" if exit_code == 0 else "fail"},
        measurement,
    )


def parse_args():
    suite_choices = sorted(set(FILE_SUITES) | set(PARAMETRIZED_CASE_GROUPS))
    parser = argparse.ArgumentParser(description="Run multimodal_gen test suite")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=suite_choices,
        help="The test suite to run.",
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
    parser.add_argument(
        "--partition-plan-json",
        type=str,
        default=None,
        help="Full partition plan JSON for the current suite.",
    )
    return parser.parse_args()


def write_execution_report(
    suite: str,
    partition_id: int,
    total_partitions: int,
    executed_cases: list[str],
    is_standalone: bool = False,
    standalone_file: str | None = None,
    case_results: dict[str, str] | None = None,
    missing_standalone_estimates: list[str] | None = None,
    standalone_measurements: list[dict] | None = None,
) -> str:
    report = {
        "suite": suite,
        "partition_id": partition_id,
        "total_partitions": total_partitions,
        "is_standalone": is_standalone,
        "standalone_file": standalone_file,
        "executed_cases": executed_cases,
        "case_results": case_results or {},
        "missing_standalone_estimates": missing_standalone_estimates or [],
        "standalone_measurements": standalone_measurements or [],
    }

    report_filename = f"execution_report_{suite}_{partition_id}.json"
    report_path = Path(__file__).parent / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Execution report written to: %s", report_path)
    return str(report_path)


def run_component_accuracy_files(files, filter_expr=None, continue_on_error=False):
    exit_code = 0
    for file_path in files:
        file_name = Path(file_path).name
        num_gpus = COMPONENT_ACCURACY_FILE_NUM_GPUS.get(file_name, 1)
        env = None
        if num_gpus > 1:
            env = os.environ.copy()
            # Torch 2.12 bundles NCCL 2.29, which hard-fails NVLS multicast
            # bind errors that NCCL 2.28 used to handle by disabling NVLS and
            # continuing. Component accuracy launches torchrun directly before
            # the diffusion launcher applies its runtime default.
            env.setdefault("NCCL_NVLS_ENABLE", "0")
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                "-m",
                "pytest",
                "-s",
                "-v",
            ]
        else:
            cmd = [sys.executable, "-m", "pytest", "-s", "-v"]

        if filter_expr:
            cmd.extend(["-k", filter_expr])
        cmd.append(file_path)

        print(f"Running command: {' '.join(cmd)}")
        file_exit_code = subprocess.call(cmd, env=env)
        if file_exit_code == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a file. Treating as success."
            )
            file_exit_code = 0
        if file_exit_code != 0 and exit_code == 0:
            exit_code = file_exit_code
        if file_exit_code != 0 and not continue_on_error:
            return file_exit_code
    return exit_code


def _is_in_ci() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true", "yes", "on")


def _maybe_pin_update_weights_model_pair(suite_files_rel: list[str]) -> None:
    if not _is_in_ci():
        return
    if _UPDATE_WEIGHTS_FROM_DISK_TEST_FILE not in suite_files_rel:
        return
    if os.environ.get(_UPDATE_WEIGHTS_MODEL_PAIR_ENV):
        print(
            f"Using preset {_UPDATE_WEIGHTS_MODEL_PAIR_ENV}="
            f"{os.environ[_UPDATE_WEIGHTS_MODEL_PAIR_ENV]}"
        )
        return

    selected_pair = random.choice(_UPDATE_WEIGHTS_MODEL_PAIR_IDS)
    os.environ[_UPDATE_WEIGHTS_MODEL_PAIR_ENV] = selected_pair
    print(f"Selected {_UPDATE_WEIGHTS_MODEL_PAIR_ENV}={selected_pair} for this CI run")


def _resolve_suite_files(
    target_dir: Path, suite_files_rel: list[str], strict: bool
) -> list[str]:
    suite_files_abs = []
    for f_rel in suite_files_rel:
        f_abs = target_dir / f_rel
        if not f_abs.exists():
            msg = f"Test file {f_rel} not found in {target_dir}."
            if strict:
                print(f"Error: {msg}")
                sys.exit(1)
            print(f"Warning: {msg} Skipping.")
            continue
        suite_files_abs.append(str(f_abs))
    return suite_files_abs


def _run_file_suite(args, target_dir: Path) -> int:
    suite_files_rel = FILE_SUITES[args.suite]
    _maybe_pin_update_weights_model_pair(suite_files_rel)
    suite_files_abs = _resolve_suite_files(
        target_dir, suite_files_rel, args.suite in STRICT_SUITES
    )

    if not suite_files_abs:
        print(f"No valid test files found for suite '{args.suite}'.")
        return 1 if args.suite in STRICT_SUITES else 0

    exit_code, _, _ = run_pytest(
        suite_files_abs,
        filter_expr=args.filter,
        junit_xml_path=None,
    )
    return exit_code


def _get_dynamic_suite_cases(suite: str) -> list[DiffusionTestCase]:
    cases = []
    for _, case_group in PARAMETRIZED_CASE_GROUPS[suite]:
        cases.extend(case_group)
    return cases


def _get_parametrized_files_for_case_ids(
    suite: str, case_ids: set[str], target_dir: Path
) -> list[str]:
    files = []
    for filename, case_group in PARAMETRIZED_CASE_GROUPS[suite]:
        if any(case.id in case_ids for case in case_group):
            file_path = target_dir / filename
            if file_path.exists():
                files.append(str(file_path))
            else:
                logger.warning("Test file %s not found in %s", filename, target_dir)
    return files


def _get_standalone_file(target_dir: Path, suite: str, index: int) -> str | None:
    standalone_files = STANDALONE_FILES.get(suite, [])
    if index < 0 or index >= len(standalone_files):
        return None
    file_path = target_dir / standalone_files[index]
    if file_path.exists():
        return str(file_path)
    logger.warning(
        "Standalone test file %s not found in %s", standalone_files[index], target_dir
    )
    return None


def _run_dynamic_suite(args, target_dir: Path) -> int:
    if args.partition_plan_json:
        assignment = parse_partition_plan(
            suite=args.suite,
            partition_id=args.partition_id,
            total_partitions=args.total_partitions,
            plan_json=args.partition_plan_json,
        )

        rows = [[args.suite, f"{args.partition_id + 1}/{args.total_partitions}"]]
        print(tabulate.tabulate(rows, headers=["Suite", "Partition"], tablefmt="psql"))

        total_est_time = 0.0
        executed_cases: list[str] = []
        case_results: dict[str, str] = {}
        missing_standalone_estimates: list[str] = []
        standalone_measurements: list[dict] = []
        overall_exit_code = 0

        if assignment.case_ids:
            case_id_set = set(assignment.case_ids)
            total_est_time += sum(
                get_case_est_time(case_id) for case_id in assignment.case_ids
            )
            suite_files = _get_parametrized_files_for_case_ids(
                args.suite, case_id_set, target_dir
            )
            if not suite_files:
                print(
                    f"No valid parametrized test files found for suite '{args.suite}'."
                )
                return 0

            partition_filter = " or ".join(
                f"[{case_id}]" for case_id in assignment.case_ids
            )
            filter_expr = (
                f"({partition_filter}) and ({args.filter})"
                if args.filter
                else partition_filter
            )

            print(
                f"Running {len(assignment.case_ids)} parametrized cases with estimated total "
                f"{sum(get_case_est_time(case_id) for case_id in assignment.case_ids):.1f}s:"
            )
            for case_id in assignment.case_ids:
                print(f"  - case: {case_id} ({get_case_est_time(case_id):.1f}s)")
            print(f"Test files: {[Path(f).name for f in suite_files]}")
            print(f"Filter expression: {filter_expr}")

            junit_xml_path = str(
                target_dir / f"junit_results_{args.suite}_{args.partition_id}.xml"
            )
            exit_code, new_executed_cases, new_case_results = run_pytest(
                suite_files,
                filter_expr=filter_expr,
                junit_xml_path=junit_xml_path,
            )
            _merge_execution_results(
                executed_cases, case_results, new_executed_cases, new_case_results
            )
            if exit_code != 0 and overall_exit_code == 0:
                overall_exit_code = exit_code
            if exit_code != 0 and not args.continue_on_error:
                write_execution_report(
                    suite=args.suite,
                    partition_id=args.partition_id,
                    total_partitions=args.total_partitions,
                    executed_cases=executed_cases,
                    is_standalone=False,
                    standalone_file=None,
                    case_results=case_results,
                    missing_standalone_estimates=missing_standalone_estimates,
                    standalone_measurements=standalone_measurements,
                )
                return overall_exit_code

        if assignment.standalone_files:
            standalone_estimate = sum(
                get_standalone_file_est_time(args.suite, standalone_file)[0]
                for standalone_file in assignment.standalone_files
            )
            total_est_time += standalone_estimate
            print(
                f"Running {len(assignment.standalone_files)} standalone file(s) with estimated total "
                f"{standalone_estimate:.1f}s:"
            )
            for standalone_file in assignment.standalone_files:
                est_time, used_fallback_estimate = get_standalone_file_est_time(
                    args.suite, standalone_file
                )
                fallback_suffix = (
                    f", fallback estimate {DEFAULT_STANDALONE_EST_TIME_SECONDS:.1f}s"
                    if used_fallback_estimate
                    else ""
                )
                print(
                    f"  - standalone: {standalone_file} "
                    f"({est_time:.1f}s{fallback_suffix})"
                )

            for standalone_file in assignment.standalone_files:
                exit_code, new_executed_cases, new_case_results, measurement = (
                    _run_standalone_file(
                        args.suite,
                        standalone_file,
                        target_dir,
                        extra_filter=args.filter,
                    )
                )
                if measurement["used_fallback_estimate"]:
                    missing_standalone_estimates.append(standalone_file)
                standalone_measurements.append(measurement)
                _merge_execution_results(
                    executed_cases,
                    case_results,
                    new_executed_cases,
                    new_case_results,
                )
                if exit_code != 0 and overall_exit_code == 0:
                    overall_exit_code = exit_code
                if exit_code != 0 and not args.continue_on_error:
                    break

        print(f"Partition estimated total time: {total_est_time:.1f}s")
        write_execution_report(
            suite=args.suite,
            partition_id=args.partition_id,
            total_partitions=args.total_partitions,
            executed_cases=executed_cases,
            is_standalone=False,
            standalone_file=None,
            case_results=case_results,
            missing_standalone_estimates=missing_standalone_estimates,
            standalone_measurements=standalone_measurements,
        )
        return overall_exit_code

    all_cases = _get_dynamic_suite_cases(args.suite)
    standalone_files = STANDALONE_FILES.get(args.suite, [])
    parametrized_partitions = args.total_partitions - len(standalone_files)

    if parametrized_partitions < 0:
        print(
            f"Error: total_partitions ({args.total_partitions}) must be >= "
            f"standalone files ({len(standalone_files)})"
        )
        return 1

    if args.partition_id < parametrized_partitions:
        if not all_cases:
            print(f"No cases found for suite '{args.suite}'.")
            return 0

        my_cases = auto_partition(all_cases, args.partition_id, parametrized_partitions)
        if not my_cases:
            print(
                f"No cases assigned to partition {args.partition_id}. Exiting success."
            )
            write_execution_report(
                suite=args.suite,
                partition_id=args.partition_id,
                total_partitions=args.total_partitions,
                executed_cases=[],
                is_standalone=False,
                missing_standalone_estimates=[],
                standalone_measurements=[],
            )
            return 0

        case_ids = [case.id for case in my_cases]
        case_id_set = set(case_ids)
        total_est_time = sum(get_case_est_time(case.id) for case in my_cases)
        suite_files = _get_parametrized_files_for_case_ids(
            args.suite, case_id_set, target_dir
        )

        if not suite_files:
            print(f"No valid parametrized test files found for suite '{args.suite}'.")
            return 0

        partition_filter = " or ".join(f"[{case_id}]" for case_id in case_ids)
        filter_expr = (
            f"({partition_filter}) and ({args.filter})"
            if args.filter
            else partition_filter
        )

        rows = [[args.suite, f"{args.partition_id + 1}/{args.total_partitions}"]]
        print(tabulate.tabulate(rows, headers=["Suite", "Partition"], tablefmt="psql"))
        print(
            f"Running {len(my_cases)} cases with estimated total "
            f"{total_est_time:.1f}s:"
        )
        for case in my_cases:
            print(f"  - {case.id} ({get_case_est_time(case.id):.1f}s)")
        print(f"Test files: {[Path(f).name for f in suite_files]}")
        print(f"Filter expression: {filter_expr}")

        junit_xml_path = str(
            target_dir / f"junit_results_{args.suite}_{args.partition_id}.xml"
        )
        exit_code, executed_cases, case_results = run_pytest(
            suite_files,
            filter_expr=filter_expr,
            junit_xml_path=junit_xml_path,
        )
        write_execution_report(
            suite=args.suite,
            partition_id=args.partition_id,
            total_partitions=args.total_partitions,
            executed_cases=executed_cases,
            is_standalone=False,
            case_results=case_results,
            missing_standalone_estimates=[],
            standalone_measurements=[],
        )
        return exit_code

    standalone_idx = args.partition_id - parametrized_partitions
    if standalone_idx >= len(standalone_files):
        print(
            f"ERROR: Standalone partition index {standalone_idx} exceeds available "
            f"standalone files ({len(standalone_files)}) for suite '{args.suite}'."
        )
        return 1

    standalone_rel = standalone_files[standalone_idx]
    print(
        f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions} (standalone)"
    )
    print(f"Running standalone test file: {Path(standalone_rel).name}")
    exit_code, executed_cases, case_results, measurement = _run_standalone_file(
        args.suite,
        standalone_rel,
        target_dir,
        extra_filter=args.filter,
    )
    write_execution_report(
        suite=args.suite,
        partition_id=args.partition_id,
        total_partitions=args.total_partitions,
        executed_cases=executed_cases,
        is_standalone=True,
        standalone_file=standalone_rel,
        case_results=case_results,
        missing_standalone_estimates=(
            [standalone_rel] if measurement["used_fallback_estimate"] else []
        ),
        standalone_measurements=[measurement],
    )
    return exit_code


def main():
    args = parse_args()
    validate_standalone_file_est_times()
    test_root_dir = Path(__file__).resolve().parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    if args.suite in COMPONENT_ACCURACY_SUITES:
        suite_files_rel = FILE_SUITES[args.suite]
        suite_files_abs = _resolve_suite_files(
            target_dir, suite_files_rel, args.suite in STRICT_SUITES
        )

        if not suite_files_abs:
            print(f"No valid test files found for suite '{args.suite}'.")
            sys.exit(1 if args.suite in STRICT_SUITES else 0)

        my_files = partition_items_by_index(
            suite_files_abs, args.partition_id, args.total_partitions
        )
        partition_info = (
            f"{args.partition_id + 1}/{args.total_partitions} "
            f"(0-based id={args.partition_id})"
        )
        headers = ["Suite", "Partition"]
        rows = [[args.suite, partition_info]]
        msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
        msg += f"Enabled {len(my_files)} file(s):\n"
        for file_path in my_files:
            msg += f"  - {file_path}\n"
        print(msg, flush=True)
        print(
            f"Suite: {args.suite} | Partition: {args.partition_id}/{args.total_partitions}"
        )
        print(f"Selected {len(suite_files_abs)} files:")
        for f in suite_files_abs:
            print(f"  - {os.path.basename(f)}")

        if not my_files:
            print("No files assigned to this partition. Exiting success.")
            sys.exit(0)

        print(f"Running {len(my_files)} files in this shard: {', '.join(my_files)}")

        exit_code = run_component_accuracy_files(
            my_files,
            filter_expr=args.filter,
            continue_on_error=args.continue_on_error,
        )

        msg = "\n" + tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
        msg += f"Executed {len(my_files)} file(s):\n"
        for file_path in my_files:
            msg += f"  - {file_path}\n"
        print(msg, flush=True)
    elif args.suite in PARAMETRIZED_CASE_GROUPS:
        exit_code = _run_dynamic_suite(args, target_dir)
    else:
        exit_code = _run_file_suite(args, target_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
