"""
Test runner for multimodal_gen that manages test suites and parallel execution.

Usage:
    python3 run_suite.py --suite <suite_name> --partition-id <id> --total-partitions <num>

Example:
    python3 run_suite.py --suite 1-gpu --partition-id 0 --total-partitions 4
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import tabulate

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_UPDATE_WEIGHTS_FROM_DISK_TEST_FILE = "test_update_weights_from_disk.py"
_UPDATE_WEIGHTS_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"
_UPDATE_WEIGHTS_MODEL_PAIR_IDS = (
    "FLUX.2-klein-base-4B",
    "Qwen-Image",
)


def _discover_unit_tests() -> list[str]:
    """Auto-discover all test_*.py files in the unit/ directory."""
    unit_dir = Path(__file__).resolve().parent / "unit"
    if not unit_dir.is_dir():
        return []
    return sorted(
        f"../unit/{f.name}" for f in unit_dir.glob("test_*.py") if f.is_file()
    )


SUITES = {
    # no GPU required; safe to run on any CPU-only runner
    # Auto-discovered from test/unit/test_*.py
    "unit": _discover_unit_tests(),
    "1-gpu": [
        "test_server_a.py",
        "test_server_b.py",
        # cli test
        "../cli/test_generate_t2i_perf.py",
        "test_update_weights_from_disk.py",
        # add new 1-gpu test files here
    ],
    "2-gpu": [
        "test_server_2_gpu_a.py",
        "test_server_2_gpu_b.py",
        # add new 2-gpu test files here
    ],
    "1-gpu-b200": [
        "test_server_c.py",
    ],
}

suites_ascend = {
    "1-npu": [
        "ascend/test_server_1_npu.py",
        # add new 1-npu test files here
    ],
    "2-npu": [
        "ascend/test_server_2_npu.py",
        # add new 2-npu test files here
    ],
    "8-npu": [
        "ascend/test_server_8_npu.py",
        # add new 8-npu test files here
    ],
}

SUITES.update(suites_ascend)
STRICT_SUITES = {"unit"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run multimodal_gen test suite")
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
        help="Continue running remaining tests even if one fails (for CI consistency; pytest already continues by default)",
    )
    return parser.parse_args()


def collect_test_items(files, filter_expr=None):
    """Collect test item node IDs from the given files using pytest --collect-only."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    cmd.extend(files)

    filter_note = f" with filter: {filter_expr}" if filter_expr else ""
    print(f"Collecting tests from {len(files)} file(s){filter_note}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for collection errors
    # pytest exit codes:
    #   0: success
    #   1: tests collected but some had errors during collection
    #   2: test execution interrupted
    #   3: internal error
    #   4: command line usage error
    #   5: no tests collected (may be expected with filters)
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

    # Parse the output to extract test node IDs
    # pytest -q outputs lines like: test_file.py::TestClass::test_method[param]
    test_items = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        # Skip empty lines and summary lines
        if line and "::" in line and not line.startswith(("=", "-", " ")):
            # Handle lines that might have extra info after the test ID
            test_id = line.split()[0] if " " in line else line
            if "::" in test_id:
                test_items.append(test_id)

    print(f"Collected {len(test_items)} test items")
    return test_items


def _run_pytest_attempt(cmd: list[str]) -> tuple[int, str]:
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
    return process.returncode, output_bytes.decode("utf-8", errors="replace")


def _extract_collection_line(full_output: str) -> str | None:
    for line in full_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("collected "):
            return stripped
    return None


def _extract_short_test_summary(full_output: str) -> list[str]:
    summary_lines = []
    in_summary = False
    for line in full_output.splitlines():
        stripped = line.strip()
        if "short test summary info" in stripped:
            in_summary = True
            continue
        if not in_summary:
            continue
        if stripped.startswith("="):
            break
        if not stripped or stripped.startswith("!"):
            continue
        summary_lines.append(stripped)
    return summary_lines


def _extract_failure_tail(full_output: str, max_lines: int = 20) -> list[str]:
    summary_lines = _extract_short_test_summary(full_output)
    if summary_lines:
        return summary_lines

    lines = [line.rstrip() for line in full_output.splitlines() if line.strip()]
    return lines[-max_lines:]


def _is_retryable_failure(full_output: str) -> bool:
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
        "out of memory" in full_output.lower() or "oom killer" in full_output.lower()
    )

    return is_perf_assertion or is_flaky_ci_assertion or is_oom_error


def _print_attempt_tail_summary(attempt_reports: list[dict], assigned_count: int) -> None:
    if len(attempt_reports) == 1 and attempt_reports[0]["returncode"] in (0, 5):
        return

    rows = []
    for report in attempt_reports:
        if report["returncode"] in (0, 5):
            result = "success"
        elif report["retryable"]:
            result = "retryable failure"
        else:
            result = "failure"
        rows.append(
            [
                report["attempt"],
                report["mode"],
                result,
                report["collection_line"] or "-",
            ]
        )

    print("\n" + "=" * 32 + " Pytest Tail Summary " + "=" * 32, flush=True)
    print(f"Assigned {assigned_count} test item(s)", flush=True)
    print(
        tabulate.tabulate(
            rows,
            headers=["Attempt", "Mode", "Result", "Collection"],
            tablefmt="psql",
        ),
        flush=True,
    )

    for report in attempt_reports:
        if not report["failure_tail"]:
            continue
        print(f"\nAttempt {report['attempt']} failure summary:", flush=True)
        for line in report["failure_tail"]:
            print(f"  {line}", flush=True)

    print("=" * 84, flush=True)


def run_pytest(files, filter_expr=None):
    if not files:
        print("No files to run.")
        return 0

    base_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        "-v",
        "--tb=short",
        "--no-header",
    ]

    if filter_expr:
        base_cmd.extend(["-k", filter_expr])

    max_retries = 6
    attempt_reports = []

    for i in range(max_retries + 1):
        is_retry = i > 0
        cmd = list(base_cmd)
        if is_retry:
            cmd.append("--last-failed")
        cmd.extend(files)

        mode = "retry failed items" if is_retry else "initial pass"
        print(
            f"Starting pytest attempt {i + 1}/{max_retries + 1}: {mode} "
            f"for {len(files)} assigned item(s)"
        )

        returncode, full_output = _run_pytest_attempt(cmd)
        retryable = returncode not in (0, 5) and _is_retryable_failure(full_output)
        attempt_reports.append(
            {
                "attempt": i + 1,
                "mode": mode,
                "returncode": returncode,
                "retryable": retryable,
                "collection_line": _extract_collection_line(full_output),
                "failure_tail": (
                    _extract_failure_tail(full_output)
                    if returncode not in (0, 5)
                    else []
                ),
            }
        )

        if returncode == 0:
            if is_retry:
                print(f"Recovered retryable failures on attempt {i + 1}.")
            _print_attempt_tail_summary(attempt_reports, len(files))
            return 0

        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            _print_attempt_tail_summary(attempt_reports, len(files))
            return 0

        if not retryable:
            _print_attempt_tail_summary(attempt_reports, len(files))
            return returncode

        if i == max_retries:
            print(f"Max retry exceeded ({max_retries})")
            _print_attempt_tail_summary(attempt_reports, len(files))
            return returncode

        print(
            f"Retryable failure detected on attempt {i + 1}. "
            "Retrying only previously failed items."
        )

    _print_attempt_tail_summary(attempt_reports, len(files))
    return attempt_reports[-1]["returncode"]


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
    _maybe_pin_update_weights_model_pair(suite_files_rel)

    suite_files_abs = []
    for f_rel in suite_files_rel:
        f_abs = target_dir / f_rel
        if not f_abs.exists():
            msg = f"Test file {f_rel} not found in {target_dir}."
            if args.suite in STRICT_SUITES:
                print(f"Error: {msg}")
                sys.exit(1)
            print(f"Warning: {msg} Skipping.")
            continue
        suite_files_abs.append(str(f_abs))

    if not suite_files_abs:
        print(f"No valid test files found for suite '{args.suite}'.")
        sys.exit(1 if args.suite in STRICT_SUITES else 0)

    # 3. collect all test items and partition by items (not files)
    all_test_items = collect_test_items(suite_files_abs, filter_expr=args.filter)

    if not all_test_items:
        print(f"No test items found for suite '{args.suite}'.")
        sys.exit(0)

    # Partition by test items
    my_items = [
        item
        for i, item in enumerate(all_test_items)
        if i % args.total_partitions == args.partition_id
    ]

    # Print test info at beginning (similar to test/run_suite.py pretty_print_tests)
    partition_info = f"{args.partition_id + 1}/{args.total_partitions} (0-based id={args.partition_id})"
    headers = ["Suite", "Partition"]
    rows = [[args.suite, partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
    msg += f"✅ Assigned {len(my_items)} test(s) from {len(suite_files_abs)} file(s):\n"
    for f in suite_files_abs:
        msg += f"  - {os.path.basename(f)}\n"
    print(msg, flush=True)

    if not my_items:
        print("No items assigned to this partition. Exiting success.")
        sys.exit(0)

    print(f"Running shard with {len(my_items)} assigned test item(s)")

    # 4. execute with the specific test items
    exit_code = run_pytest(my_items)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
