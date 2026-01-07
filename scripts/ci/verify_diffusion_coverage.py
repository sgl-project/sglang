#!/usr/bin/env python3
"""
Verify 100% coverage of diffusion test cases.

This script checks that all expected test cases were executed across all partitions.
Designed to run in the CI summary job after all partition jobs complete.

Usage:
    python scripts/ci/verify_diffusion_coverage.py --reports-dir <path>

Exit codes:
    0 - All cases executed (100% coverage)
    1 - Missing cases detected (coverage < 100%)
"""

import argparse
import json
import sys
from pathlib import Path

from diffusion_case_parser import (
    BASELINE_REL_PATH,
    RUN_SUITE_REL_PATH,
    TESTCASE_CONFIG_REL_PATH,
    collect_diffusion_suites,
)


def load_execution_reports(reports_dir: Path) -> list[dict]:
    """Load all execution report JSON files from the given directory."""
    reports = []
    for json_file in reports_dir.glob("**/execution_report_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def get_expected_cases(repo_root: Path) -> dict[str, set[str]]:
    """
    Get all expected cases from testcase_configs.py and run_suite.py.

    Returns:
        Dictionary mapping suite name to set of expected case IDs.
        Standalone files are represented as "standalone:<filename>".
    """
    testcase_config_path = repo_root / TESTCASE_CONFIG_REL_PATH
    baseline_path = repo_root / BASELINE_REL_PATH
    run_suite_path = repo_root / RUN_SUITE_REL_PATH

    suites = collect_diffusion_suites(
        testcase_config_path,
        run_suite_path,
        baseline_path,
    )

    expected = {}
    for suite_name, suite_info in suites.items():
        case_ids = set(case.case_id for case in suite_info.cases)
        # Add standalone files as special case IDs
        for standalone_file in suite_info.standalone_files:
            case_ids.add(f"standalone:{standalone_file}")
        expected[suite_name] = case_ids

    return expected


def collect_executed_cases(reports: list[dict]) -> dict[str, set[str]]:
    """
    Collect all executed cases from execution reports.

    Returns:
        Dictionary mapping suite name to set of executed case IDs.
    """
    executed = {}
    for report in reports:
        suite = report["suite"]
        if suite not in executed:
            executed[suite] = set()

        if report["is_standalone"]:
            standalone_file = report["standalone_file"]
            executed[suite].add(f"standalone:{standalone_file}")
        else:
            executed[suite].update(report["executed_cases"])

    return executed


def collect_case_results(reports: list[dict]) -> dict[str, dict[str, str]]:
    """
    Collect case results (pass/fail/error status) from execution reports.

    Returns:
        Dictionary mapping suite name to {case_id: status} dictionary.
    """
    results = {}
    for report in reports:
        suite = report["suite"]
        if suite not in results:
            results[suite] = {}

        # Get case_results from report (empty dict for legacy reports)
        case_results = report.get("case_results", {})
        results[suite].update(case_results)

    return results


def verify_coverage(
    expected: dict[str, set[str]],
    executed: dict[str, set[str]],
) -> tuple[bool, dict[str, set[str]]]:
    """
    Verify that all expected cases were executed.

    Returns:
        Tuple of (is_complete, missing_cases_by_suite)
    """
    missing = {}
    for suite, expected_cases in expected.items():
        executed_cases = executed.get(suite, set())
        suite_missing = expected_cases - executed_cases
        if suite_missing:
            missing[suite] = suite_missing

    return len(missing) == 0, missing


def print_results_summary(
    case_results: dict[str, dict[str, str]],
) -> tuple[int, int, int]:
    """
    Print test results summary and return counts.

    Returns:
        Tuple of (passed_count, failed_count, error_count)
    """
    # Check if we have any results data
    total_results = sum(len(results) for results in case_results.values())
    if total_results == 0:
        print("\nTest Results: No results data available (legacy reports)")
        return (0, 0, 0)

    # Count by status
    passed_count = 0
    failed_count = 0
    error_count = 0
    failed_cases: dict[str, list[str]] = {}

    for suite, results in case_results.items():
        for case_id, status in results.items():
            if status == "pass":
                passed_count += 1
            elif status == "fail":
                failed_count += 1
                if suite not in failed_cases:
                    failed_cases[suite] = []
                failed_cases[suite].append(case_id)
            elif status == "error":
                error_count += 1
                if suite not in failed_cases:
                    failed_cases[suite] = []
                failed_cases[suite].append(f"{case_id} (error)")

    # Print summary
    total = passed_count + failed_count + error_count
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"  Total executed: {total}")
    print(f"  ✅ Passed: {passed_count}")
    print(f"  ❌ Failed: {failed_count}")
    if error_count > 0:
        print(f"  ⚠️  Errors: {error_count}")

    # Print failed cases if any
    if failed_cases:
        print("\nFailed cases:")
        for suite, cases in sorted(failed_cases.items()):
            print(f"  {suite}:")
            for case_id in sorted(cases):
                print(f"    - {case_id}")

    return (passed_count, failed_count, error_count)


def main():
    parser = argparse.ArgumentParser(
        description="Verify 100% coverage of diffusion test cases"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        required=True,
        help="Directory containing execution report JSON files",
    )
    args = parser.parse_args()

    # Determine repository root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    reports_dir = Path(args.reports_dir)

    print("=" * 60)
    print("Diffusion CI Coverage Verification")
    print("=" * 60)

    # Load execution reports
    reports = load_execution_reports(reports_dir)
    print(f"\nLoaded {len(reports)} execution reports")

    if not reports:
        print("\nERROR: No execution reports found!")
        print(f"Expected reports in: {reports_dir}")
        sys.exit(1)

    # Get expected cases
    expected = get_expected_cases(repo_root)
    print("\nExpected cases by suite:")
    for suite, cases in expected.items():
        print(f"  {suite}: {len(cases)} cases")

    # Collect executed cases
    executed = collect_executed_cases(reports)
    print("\nExecuted cases by suite:")
    for suite, cases in executed.items():
        print(f"  {suite}: {len(cases)} cases")

    # Collect case results
    case_results = collect_case_results(reports)

    # Verify coverage
    is_complete, missing = verify_coverage(expected, executed)

    if is_complete:
        print("\n" + "=" * 60)
        print("✅ COVERAGE: 100% - All test cases executed")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ COVERAGE FAILURE: Missing test cases detected")
        print("=" * 60)
        for suite, cases in missing.items():
            print(f"\n{suite.upper()} suite - Missing {len(cases)} case(s):")
            for case_id in sorted(cases):
                print(f"  - {case_id}")

    # Print test results summary
    passed_count, failed_count, error_count = print_results_summary(case_results)

    # Exit with appropriate code
    if not is_complete:
        sys.exit(1)
    elif failed_count > 0 or error_count > 0:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Some tests failed but coverage is complete")
        print("=" * 60)
        sys.exit(0)  # Coverage is complete, failures are visible in results
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
