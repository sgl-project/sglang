#!/usr/bin/env python3
"""
analyze_failure_report.py  (NEW FILE - add to .github/workflows/scripts/)

Cross-reference CI test failures with test recommendations.

Pipeline:
  1. Scan each .txt log file for "short test summary info" block
  2. Extract FAILED and ERROR lines from that block only (no full-file scan)
  3. Read recommended_pytest_paths.txt
  4. Match: exact match + file-level match
  5. Generate a Markdown report

Usage:
  python analyze_failure_report.py --log-dir LOG_DIR --recommendations-file RECOMMENDED.txt [--output report.md]
"""

import argparse
import contextlib
import sys
from pathlib import Path

import regex as re

# ============================================================
#  Utility: strip CI log noise
# ============================================================


def strip_ansi(text):
    """Remove ANSI color codes like \x1b[31m, \x1b[0m, etc."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def strip_timestamp(line):
    """Remove GitHub Actions timestamp prefix: YYYY-MM-DDTHH:MM:SS.fffffffZ"""
    return re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+", "", line)


def clean_line(line):
    """Strip BOM, ANSI codes, and timestamp from one log line."""
    line = line.lstrip("\ufeff")  # UTF-8 BOM marker
    return strip_ansi(strip_timestamp(line)).strip()


# ============================================================
#  Step 1: Extract FAILED and ERROR tests from log files
# ============================================================


FAILED_PATTERN = re.compile(r"^(?:FAILED|ERROR)\s+(tests/\S+?\.py(?:::\S+?)?)\s")
SUMMARY_SEPARATOR_PATTERN = re.compile(r"^=+\s")
CPU_LOG_PATH_PATTERN = re.compile(r"(?:^|-)cpu-\d+card(?:-|$)", re.IGNORECASE)
CPU_FAILURE_LABEL = "cpu-ut"


def extract_failed_from_log(log_path):
    """Extract pytest node IDs from one log's short test summary."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        print(f"::warning:: Cannot read {log_path}: {exc}")
        return []

    failed = []
    in_summary = False
    for line in lines:
        text = clean_line(line)

        if "short test summary info" in text:
            in_summary = True
            continue

        if not in_summary:
            continue

        if SUMMARY_SEPARATOR_PATTERN.match(text):
            in_summary = False
            continue

        match = FAILED_PATTERN.match(text)
        if match:
            failed.append(match.group(1))

    return failed


def is_cpu_log(log_path):
    """Return whether a log belongs to a CPU selected-test artifact."""
    if log_path.stem.lower().endswith("-cpu-ut"):
        return True

    return any(CPU_LOG_PATH_PATTERN.search(part) for part in log_path.parent.parts)


def extract_failed_from_logs(log_dir):
    """
    Scan CPU logs first and represent all CPU failures as one ``cpu-ut`` item.
    Scan all remaining logs with the existing pytest node-ID behavior.
    """
    base = Path(log_dir)
    if not base.is_dir():
        print(f"::warning:: Log directory not found: {log_dir}")
        return []

    # Scan both real .log files (from run_selected_tests.sh) and mock .txt files
    candidates = []
    candidates.extend(base.rglob("*.log"))
    candidates.extend(base.rglob("*.txt"))
    candidates = [
        candidate
        for candidate in sorted(candidates)
        if candidate.suffix != ".txt" or "run-selected-tests" in candidate.name
    ]

    cpu_candidates = []
    regular_candidates = []
    for candidate in candidates:
        target = cpu_candidates if is_cpu_log(candidate) else regular_candidates
        target.append(candidate)

    all_failed = []
    seen = set()

    cpu_failed = False
    for candidate in cpu_candidates:
        if extract_failed_from_log(candidate):
            cpu_failed = True

    if cpu_failed:
        seen.add(CPU_FAILURE_LABEL)
        all_failed.append(CPU_FAILURE_LABEL)

    for candidate in regular_candidates:
        for test_path in extract_failed_from_log(candidate):
            if test_path not in seen:
                seen.add(test_path)
                all_failed.append(test_path)

    return all_failed


# ============================================================
#  Step 2: Read recommendations
# ============================================================


def read_recommended(recommendations_file):
    """
    recommended_pytest_paths.txt contains one pytest path per line, e.g.:
        tests/ops/test_matmul.py::test_bf16
        tests/layers/test_attention.py
    """
    path = Path(recommendations_file)
    if not path.exists():
        print(f"::warning:: Recommendations file not found: {recommendations_file}")
        return []
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff")
    return [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.startswith("ERROR")
    ]


# ============================================================
#  Step 3: Match
# ============================================================


def normalize_test_path(test_path):
    """Return a stable comparison key for a pytest path or node ID."""
    normalized = test_path.strip().replace("\\", "/").removeprefix("./")
    file_path, separator, test_name = normalized.partition("::")
    file_path = file_path.removesuffix(".py")
    if separator:
        test_name = test_name.partition("[")[0]
    return f"{file_path}{separator}{test_name}" if separator else file_path


def match_failed_vs_recommended(failed, recommended):
    """
    Two-level matching:
      Level 1 - File-level: recommended "tests/foo.py" (no function)
                  matches failed "tests/foo.py::anything"
      Level 2 - Exact: "tests/foo.py::test_bar" in both lists

    Returns {"hit": [...], "miss": [...], "untested": [...]}
      hit:       failed AND recommended
      miss:      failed but NOT recommended
      untested:  recommended but NOT in failed list
    """
    recommended_files = {
        normalize_test_path(item) for item in recommended if "::" not in item
    }
    recommended_functions = {
        normalize_test_path(item) for item in recommended if "::" in item
    }

    hit = []
    miss = []

    normalized_failed = {item: normalize_test_path(item) for item in failed}
    for original, normalized in normalized_failed.items():
        failed_file = normalized.split("::", 1)[0]
        if failed_file in recommended_files or normalized in recommended_functions:
            hit.append(original)
        else:
            miss.append(original)

    # Recommended but not failed
    failed_functions = set(normalized_failed.values())
    failed_files = {item.split("::", 1)[0] for item in failed_functions}
    untested = []
    for item in recommended:
        normalized = normalize_test_path(item)
        has_failure = (
            normalized in failed_functions
            if "::" in item
            else normalized in failed_files
        )
        if not has_failure:
            untested.append(item)

    return {"hit": hit, "miss": miss, "untested": untested}


# ============================================================
#  Step 4: Generate Markdown report
# ============================================================


def generate_report(
    failed, recommended, matched, log_dir, recommendations_source="none"
):
    """Produce a Markdown summary table."""
    hit = matched["hit"]
    miss = matched["miss"]
    untested = matched["untested"]

    out = []
    out.append("# Test Failure vs Recommendation Report")
    out.append("")
    out.append(f"**Log source**: `{log_dir}`")
    out.append("")

    # Recommendation source indicator
    if recommendations_source == "output":
        out.append(
            "> **[Source: Workflow Output]** Recommended cases are passed from coverage recommendations outputs"
        )
    elif recommendations_source == "committed":
        out.append(
            "> **[Source: Local File]** Recommended test cases come from a txt file in the repository"
        )
    else:
        out.append("> **[Source: None]** No recommended test cases found")
    out.append("")

    # ================================================================
    #  Section 1: Full Failed Test List
    # ================================================================
    out.append("---")
    out.append("")
    out.append(f"## Failed Test Cases（ {len(failed)} total）")
    out.append("")
    if failed:
        for i, t in enumerate(failed, 1):
            tag = (
                " **[Matched Recommendation]**"
                if t in hit
                else " **[Not Matched Recommendation]**"
            )
            out.append(f"{i}. `{t}`{tag}")
        out.append("")
    else:
        out.append("> No failed test cases")
        out.append("")

    # ================================================================
    #  Section 2: Full Recommended Test List
    # ================================================================
    out.append("---")
    out.append("")
    out.append(f"## Recommended Test Cases（ {len(recommended)} total）")
    out.append("")
    if recommended:
        normalized_failed = {normalize_test_path(item) for item in failed}
        failed_file_set = {item.split("::", 1)[0] for item in normalized_failed}
        for i, item in enumerate(recommended, 1):
            normalized = normalize_test_path(item)
            has_failure = (
                normalized in normalized_failed
                if "::" in item
                else normalized in failed_file_set
            )
            tag = " **[Already Failed]**" if has_failure else ""
            out.append(f"{i}. `{item}`{tag}")
        out.append("")
    else:
        out.append("> No recommended test cases")
        out.append("")

    # ================================================================
    #  Section 3: Core Conclusion
    # ================================================================
    out.append("---")
    out.append("")
    out.append("## Core Conclusion")
    out.append("")
    if not failed:
        out.append(
            "> No failed cases in this CI run; no need to compare against the recommendation list."
        )
    elif len(miss) == 0:
        out.append("> **All failed test cases are within the recommended scope.**")
    else:
        total_failed = len(failed)
        out.append(
            f"> ** {len(miss)}/{total_failed} failed cases are outside the recommended scope.**"
        )
    out.append("")

    # ================================================================
    #  Section 4: Detail table
    # ================================================================
    out.append("| Category | Count |")
    out.append("|---|---|")
    out.append(f"| Failed & Matched Recommendation | {len(hit)} |")
    out.append(f"| Failed but Not Matched Recommendation | {len(miss)} |")
    out.append(f"| Recommended but Not Failed | {len(untested)} |")
    out.append("")

    if hit:
        out.append("## Failed & Matched Recommendation")
        out.append("")
        out.append("| # | Failed test |")
        out.append("|---|---|")
        for i, t in enumerate(hit, 1):
            out.append(f"| {i} | `{t}` |")
        out.append("")

    if miss:
        out.append("## Failed but Not Matched Recommendation")
        out.append("")
        out.append(
            "> Possible causes: uncovered modules, environment issues, flaky tests."
        )
        out.append("")
        for t in miss:
            out.append(f"- `{t}`")
        out.append("")

    if untested:
        out.append("## Recommended but Not Failed")
        out.append("")
        out.append(
            "> These test cases were recommended but did not fail this run (passed or not executed)."
        )
        out.append("")
        for t in untested:
            out.append(f"- `{t}`")
        out.append("")

    if not hit and not miss:
        out.append("## No failed cases")
        out.append("")

    out.append("---")
    out.append("*Generated by analyze_failure_report.py*")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-reference CI test failures with test recommendations"
    )
    parser.add_argument(
        "--log-dir", required=True, help="Directory containing CI .txt log files"
    )
    parser.add_argument(
        "--recommendations-file",
        help="Path to recommended_pytest_paths.txt",
    )
    parser.add_argument(
        "--hitest-file",
        dest="recommendations_file",
        default=argparse.SUPPRESS,
        help="(Deprecated alias) same as --recommendations-file",
    )
    parser.add_argument(
        "--output",
        default="failure_report.md",
        help="Output Markdown report path (default: failure_report.md)",
    )
    parser.add_argument(
        "--recommendations-source",
        default="none",
        choices=["committed", "output", "none"],
        help="Where recommendations came from",
    )
    parser.add_argument(
        "--hitest-source",
        dest="recommendations_source",
        default=argparse.SUPPRESS,
        help="(Deprecated alias) same as --recommendations-source",
    )
    args = parser.parse_args()

    if not args.recommendations_file:
        parser.error("one of --recommendations-file or --hitest-file is required")

    # For Windows console: force UTF-8 if possible
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 50)
    print("Step 1: Extract failed tests from CI logs")
    print("=" * 50)
    failed = extract_failed_from_logs(args.log_dir)
    print(f"Failed: {len(failed)}")

    print()
    print("=" * 50)
    print("Step 2: Read recommendations")
    print("=" * 50)
    recommended = read_recommended(args.recommendations_file)
    print(f"Recommended: {len(recommended)}")

    print()
    print("=" * 50)
    print("Step 3: Match")
    print("=" * 50)
    matched = match_failed_vs_recommended(failed, recommended)
    print(f"Hit (failed + recommended): {len(matched['hit'])}")
    print(f"Miss (failed, not recommended): {len(matched['miss'])}")
    print(f"Untested (recommended, no failure): {len(matched['untested'])}")

    report = generate_report(
        failed, recommended, matched, args.log_dir, args.recommendations_source
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print()
    print(f"Report => {output_path}")
    print()

    # Print report to stdout (safe fallback for Windows encoding)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
