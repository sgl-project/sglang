"""
Test runner for multimodal_gen that manages test suites and parallel execution.

For diffusion 1-gpu/2-gpu suites, cases are partitioned by estimated runtime
using LPT so each CI shard has a similar total runtime.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import tabulate

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)

logger = init_logger(__name__)

DEFAULT_EST_TIME_SECONDS = 300.0
STARTUP_OVERHEAD_SECONDS = 120.0

_UPDATE_WEIGHTS_FROM_DISK_TEST_FILE = "test_update_weights_from_disk.py"
_UPDATE_WEIGHTS_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"
_UPDATE_WEIGHTS_MODEL_PAIR_IDS = (
    "FLUX.2-klein-base-4B",
    "Qwen-Image",
)


def _discover_unit_tests() -> list[str]:
    unit_dir = Path(__file__).resolve().parent / "unit"
    if not unit_dir.is_dir():
        return []
    return sorted(
        f"../unit/{f.name}" for f in unit_dir.glob("test_*.py") if f.is_file()
    )


FILE_SUITES = {
    "unit": _discover_unit_tests(),
    "1-gpu-b200": ["test_server_c.py"],
}

suites_ascend = {
    "1-npu": ["ascend/test_server_1_npu.py"],
    "2-npu": ["ascend/test_server_2_npu.py"],
    "8-npu": ["ascend/test_server_8_npu.py"],
}
FILE_SUITES.update(suites_ascend)

PARAMETRIZED_CASE_GROUPS = {
    "1-gpu": [
        ("test_server_a.py", ONE_GPU_CASES_A),
        ("test_server_b.py", ONE_GPU_CASES_B),
    ],
    "2-gpu": [
        ("test_server_2_gpu_a.py", TWO_GPU_CASES_A),
        ("test_server_2_gpu_b.py", TWO_GPU_CASES_B),
    ],
}

# NOTE: This is parsed by diffusion_case_parser.py using AST.
STANDALONE_FILES = {
    "1-gpu": [
        "../cli/test_generate_t2i_perf.py",
        "test_update_weights_from_disk.py",
    ],
    "2-gpu": [],
}

STRICT_SUITES = {"unit"}


def get_case_est_time(case_id: str) -> float:
    scenario = BASELINE_CONFIG.scenarios.get(case_id)
    if scenario is None:
        return DEFAULT_EST_TIME_SECONDS
    if scenario.estimated_full_test_time_s is not None:
        return scenario.estimated_full_test_time_s
    return scenario.expected_e2e_ms / 1000.0 + STARTUP_OVERHEAD_SECONDS


def auto_partition(
    cases: list[DiffusionTestCase], rank: int, size: int
) -> list[DiffusionTestCase]:
    if not cases or size <= 0:
        return []

    sorted_cases = sorted(cases, key=lambda c: get_case_est_time(c.id), reverse=True)
    partitions: list[list[DiffusionTestCase]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    for case in sorted_cases:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append(case)
        partition_sums[min_idx] += get_case_est_time(case.id)

    return partitions[rank] if rank < size else []


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
    return parser.parse_args()


def parse_junit_xml_for_executed_cases(xml_path: str) -> list[str]:
    if not Path(xml_path).exists():
        return []

    executed_cases = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for testcase in root.iter("testcase"):
        if testcase.find("skipped") is not None:
            continue

        name = testcase.get("name", "")
        if "[" in name and "]" in name:
            case_id = name[name.index("[") + 1 : name.index("]")]
            executed_cases.append(case_id)

    return executed_cases


def parse_junit_xml_for_case_results(xml_path: str) -> dict[str, str]:
    if not Path(xml_path).exists():
        return {}

    case_results = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for testcase in root.iter("testcase"):
        if testcase.find("skipped") is not None:
            continue

        name = testcase.get("name", "")
        if "[" not in name or "]" not in name:
            continue

        case_id = name[name.index("[") + 1 : name.index("]")]
        if testcase.find("failure") is not None:
            case_results[case_id] = "fail"
        elif testcase.find("error") is not None:
            case_results[case_id] = "error"
        else:
            case_results[case_id] = "pass"

    return case_results


def write_execution_report(
    suite: str,
    partition_id: int,
    total_partitions: int,
    executed_cases: list[str],
    is_standalone: bool = False,
    standalone_file: str | None = None,
    case_results: dict[str, str] | None = None,
) -> str:
    report = {
        "suite": suite,
        "partition_id": partition_id,
        "total_partitions": total_partitions,
        "is_standalone": is_standalone,
        "standalone_file": standalone_file,
        "executed_cases": executed_cases,
        "case_results": case_results or {},
    }

    report_filename = f"execution_report_{suite}_{partition_id}.json"
    report_path = Path(__file__).parent / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Execution report written to: %s", report_path)
    return str(report_path)


def run_pytest(
    files: list[str],
    filter_expr: str | None = None,
    junit_xml_path: str | None = None,
) -> tuple[int, list[str], dict[str, str]]:
    if not files:
        print("No files to run.")
        return (0, [], {})

    all_executed_cases: set[str] = set()
    all_case_results: dict[str, str] = {}

    base_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",
        "--tb=long",
        "--log-cli=true",
        "--log-cli-level=INFO",
        "--log-cli-format=%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        "--log-cli-date-format=%Y-%m-%d %H:%M:%S",
    ]

    if junit_xml_path:
        base_cmd.extend(["--junit-xml", junit_xml_path])
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

        if junit_xml_path:
            all_executed_cases.update(
                parse_junit_xml_for_executed_cases(junit_xml_path)
            )
            all_case_results.update(parse_junit_xml_for_case_results(junit_xml_path))

        if returncode == 0:
            return (0, list(all_executed_cases), all_case_results)
        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            return (0, list(all_executed_cases), all_case_results)

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
            return (returncode, list(all_executed_cases), all_case_results)

    logger.info("Max retry exceeded")
    return (returncode, list(all_executed_cases), all_case_results)


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
    if standalone_rel == _UPDATE_WEIGHTS_FROM_DISK_TEST_FILE:
        _maybe_pin_update_weights_model_pair([standalone_rel])

    standalone_file = _get_standalone_file(target_dir, args.suite, standalone_idx)
    if not standalone_file:
        print(
            f"ERROR: Standalone test file '{standalone_rel}' not found for suite "
            f"'{args.suite}'."
        )
        return 1

    print(
        f"Suite: {args.suite} | Partition: {args.partition_id + 1}/{args.total_partitions} (standalone)"
    )
    print(f"Running standalone test file: {Path(standalone_file).name}")

    junit_xml_path = str(
        target_dir / f"junit_results_{args.suite}_{args.partition_id}.xml"
    )
    exit_code, _, _ = run_pytest(
        [standalone_file],
        filter_expr=args.filter,
        junit_xml_path=junit_xml_path,
    )
    standalone_key = f"standalone:{standalone_rel}"
    write_execution_report(
        suite=args.suite,
        partition_id=args.partition_id,
        total_partitions=args.total_partitions,
        executed_cases=[],
        is_standalone=True,
        standalone_file=standalone_rel,
        case_results={standalone_key: "pass" if exit_code == 0 else "fail"},
    )
    return exit_code


def main():
    args = parse_args()
    test_root_dir = Path(__file__).resolve().parent
    target_dir = test_root_dir / args.base_dir

    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    if args.suite in PARAMETRIZED_CASE_GROUPS:
        exit_code = _run_dynamic_suite(args, target_dir)
    else:
        exit_code = _run_file_suite(args, target_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
