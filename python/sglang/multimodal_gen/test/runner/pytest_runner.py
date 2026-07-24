from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import tabulate


def collect_test_items(
    files: Sequence[str], filter_expr: str | None = None
) -> list[str]:
    """Collect pytest node IDs from the given files or node selectors."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if filter_expr:
        cmd.extend(["-k", filter_expr])
    cmd.extend(files)

    filter_note = f" with filter: {filter_expr}" if filter_expr else ""
    print(f"Collecting tests from {len(files)} item(s){filter_note}")
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


def _summary_has_retryable_failure(summary_lines: list[str]) -> bool:
    for line in summary_lines:
        lowered = line.lower()
        if (
            "[performance]" in line
            or "SafetensorError" in line
            or "FileNotFoundError" in line
            or "TimeoutError" in line
            or "out of memory" in lowered
            or "oom killer" in lowered
        ):
            return True
    return False


def _is_consistency_failure(full_output: str) -> bool:
    summary_lines = _extract_short_test_summary(full_output)
    for line in summary_lines:
        if "Consistency check failed for" in line or "GT not found for" in line:
            return True

    return (
        "Consistency check failed for " in full_output
        or "GT not found for " in full_output
        or "--- MISSING GROUND TRUTH DETECTED ---" in full_output
    )


def _is_retryable_failure(full_output: str) -> bool:
    if _is_consistency_failure(full_output):
        return False

    summary_lines = _extract_short_test_summary(full_output)
    is_perf_assertion = (
        "multimodal_gen/test/server/test_server_utils.py" in full_output
        and "AssertionError" in full_output
    )
    is_aggregated_retryable_failure = _summary_has_retryable_failure(summary_lines)

    is_flaky_ci_assertion = (
        "SafetensorError" in full_output
        or "FileNotFoundError" in full_output
        or "TimeoutError" in full_output
    )

    is_oom_error = (
        "out of memory" in full_output.lower() or "oom killer" in full_output.lower()
    )

    return (
        is_perf_assertion
        or is_aggregated_retryable_failure
        or is_flaky_ci_assertion
        or is_oom_error
    )


def _print_attempt_tail_summary(
    attempt_reports: list[dict], assigned_count: int
) -> None:
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


def run_pytest(
    files: Sequence[str],
    filter_expr: str | None = None,
    junit_xml_path: str | None = None,
    exitfirst: bool = False,
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
        "--tb=short",
        "--no-header",
    ]
    if exitfirst:
        base_cmd.append("-x")
    if junit_xml_path:
        base_cmd.extend(["--junit-xml", junit_xml_path])
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

        if junit_xml_path:
            all_executed_cases.update(
                parse_junit_xml_for_executed_cases(junit_xml_path)
            )
            all_case_results.update(parse_junit_xml_for_case_results(junit_xml_path))

        if returncode == 0:
            if is_retry:
                print(f"Recovered retryable failures on attempt {i + 1}.")
            _print_attempt_tail_summary(attempt_reports, len(files))
            return (0, list(all_executed_cases), all_case_results)
        if returncode == 5:
            print(
                "No tests collected (exit code 5). This is expected when filters "
                "deselect all tests in a partition. Treating as success."
            )
            _print_attempt_tail_summary(attempt_reports, len(files))
            return (0, list(all_executed_cases), all_case_results)

        if not retryable:
            _print_attempt_tail_summary(attempt_reports, len(files))
            return (returncode, list(all_executed_cases), all_case_results)

        if i == max_retries:
            print(f"Max retry exceeded ({max_retries})")
            _print_attempt_tail_summary(attempt_reports, len(files))
            return (returncode, list(all_executed_cases), all_case_results)

        print(
            f"Retryable failure detected on attempt {i + 1}. "
            "Retrying only previously failed items."
        )

    _print_attempt_tail_summary(attempt_reports, len(files))
    return (
        attempt_reports[-1]["returncode"],
        list(all_executed_cases),
        all_case_results,
    )


def partition_items_by_index(
    items: Sequence[str], partition_id: int, total_partitions: int
) -> list[str]:
    return [
        item for i, item in enumerate(items) if i % total_partitions == partition_id
    ]
