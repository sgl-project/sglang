#!/usr/bin/env python3
"""
analyze_failure_report.py

Cross-reference CI test failures with test recommendations.

Pipeline:
  1. Scan each log file for failed tests using three methods:
     a. TIMINGS JSON block (machine-readable, from ci_utils.py)
     b. ci_utils.py "✗ FAILED:" summary section (structured text)
     c. pytest "short test summary info" block (for pytest-style logs)
  2. Collect diffusion consistency failures from diffusion-failures-* artifacts
     (test_utils.py writes consistency_failures/summary.json per partition).
  3. Read recommended_pytest_paths.txt
  4. Match: exact match + file-level match
  5. Generate a Markdown report

Usage:
  python analyze_failure_report.py --log-dir LOG_DIR --recommendations-file RECOMMENDED.txt \
    [--diffusion-dir DIFFUSION_DIR] [--output report.md]
"""

import argparse
import contextlib
import json
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


# Match pytest-style FAILED/ERROR lines with a repo test path prefix. Diffusion
# tests live under python/sglang/multimodal_gen/test/, everything else under test/.
FAILED_PATTERN = re.compile(
    r"^(?:FAILED|ERROR)\s+((?:test|sglang/multimodal_gen/test)/\S+?\.py(?:::\S+?)?)\s"
)
SUMMARY_SEPARATOR_PATTERN = re.compile(r"^=+\s")
CPU_LOG_PATH_PATTERN = re.compile(r"(?:^|-)cpu-\d+card(?:-|$)", re.IGNORECASE)
CPU_FAILURE_LABEL = "cpu-ut"

# ci_utils.py summary: "✗ FAILED:" section lines like "  /path/to/test/registered/test_xxx.py (exit code 1)".
# Paths are absolute (from os.path.abspath in run_suite.py's glob).
CI_UTILS_FAILED_PATTERN = re.compile(r"^[✗X]\s*FAILED:\s*$")
CI_UTILS_FAILED_LINE_PATTERN = re.compile(r"^\s{2,}(\S+\.py)\s*\(")

# TIMINGS block: machine-readable JSON lines with "passed": false.
TIMINGS_BEGIN_PATTERN = re.compile(r"^=+\s*TIMINGS\s+BEGIN\s*=+")
TIMINGS_END_PATTERN = re.compile(r"^=+\s*TIMINGS\s+END\s*=+")


def _extract_from_timings(lines):
    """Extract failed test file paths from the TIMINGS JSON block (ci_utils.py)."""
    failed = []
    in_timings = False
    for line in lines:
        text = clean_line(line)
        if TIMINGS_BEGIN_PATTERN.search(text):
            in_timings = True
            continue
        if not in_timings:
            continue
        if TIMINGS_END_PATTERN.search(text):
            break
        try:
            entry = json.loads(text)
            if not entry.get("passed", True) and entry.get("file"):
                failed.append(entry["file"])
        except (json.JSONDecodeError, ValueError):
            continue
    return failed


def _extract_from_ci_utils_summary(lines):
    """Extract failed test file paths from ci_utils.py's '✗ FAILED:' summary section."""
    failed = []
    in_failed_section = False
    for line in lines:
        text = clean_line(line)
        if CI_UTILS_FAILED_PATTERN.match(text):
            in_failed_section = True
            continue
        if not in_failed_section:
            continue
        if SUMMARY_SEPARATOR_PATTERN.match(text):
            break
        match = CI_UTILS_FAILED_LINE_PATTERN.match(line)
        if match:
            # Paths in this section are absolute (e.g. "/__w/sglang/sglang/test/registered/test_xxx.py").
            # Strip everything up to and including the "/sglang/" marker to get
            # the repo-relative form (e.g. "test/registered/test_xxx.py").
            path = match.group(1)
            marker = "/sglang/"
            idx = path.rfind(marker)
            if idx >= 0:
                path = path[idx + len(marker) :]
            failed.append(path)
    return failed


def extract_failed_from_log(log_path):
    """Extract failed test paths from one log file.

    Tries three methods in order of reliability:
      1. TIMINGS JSON block (machine-readable, ci_utils.py)
      2. ci_utils.py '✗ FAILED:' summary section (human-readable but structured)
      3. pytest 'short test summary info' block (for pytest-style logs)
    """
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        print(f"::warning:: Cannot read {log_path}: {exc}")
        return []

    # Method 1: TIMINGS block (most reliable, machine-readable).
    failed = _extract_from_timings(lines)
    if failed:
        return failed

    # Method 2: ci_utils.py summary section.
    failed = _extract_from_ci_utils_summary(lines)
    if failed:
        return failed

    # Method 3: pytest-style "short test summary info" block.
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

    # Scan .log files (from NPU test stages) and .txt files (legacy/mock).
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
#  Step 1b: Extract diffusion consistency failures from
#  diffusion-failures-* artifacts
# ============================================================


def _dir_has_files(directory):
    """Return whether a downloaded-artifact directory exists and holds any file."""
    base = Path(directory)
    return base.is_dir() and any(item.is_file() for item in base.rglob("*"))


def extract_diffusion_failures(diffusion_dir):
    """Collect consistency failure records from diffusion-failures-* artifacts.

    Artifacts are downloaded without merge-multiple, so every partition keeps
    its own directory: <diffusion_dir>/<artifact>/consistency_failures/summary.json.
    Each summary is a list of records written by test_utils.py's
    save_consistency_failure_artifact().

    Returns a list of {case_id, num_gpus, metrics, thresholds, artifact} dicts,
    deduplicated by case_id.
    """
    base = Path(diffusion_dir)
    if not base.is_dir():
        print(f"::notice:: Diffusion artifact directory not found: {diffusion_dir}")
        return []

    records = []
    seen = set()
    for summary_path in sorted(base.rglob("summary.json")):
        if summary_path.parent.name != "consistency_failures":
            continue
        try:
            entries = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"::warning:: Cannot read {summary_path}: {exc}")
            continue
        if not isinstance(entries, list):
            continue
        artifact = summary_path.parents[1].name
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("case_id"):
                continue
            case_id = entry["case_id"]
            if case_id in seen:
                continue
            seen.add(case_id)
            records.append(
                {
                    "case_id": case_id,
                    "num_gpus": entry.get("num_gpus"),
                    "metrics": entry.get("metrics", {}),
                    "thresholds": entry.get("thresholds", {}),
                    "artifact": artifact,
                }
            )
    return records


def diffusion_failure_id(record):
    """Failed-list identifier for one diffusion consistency failure record."""
    return f"diffusion-consistency/{record['case_id']}"


# ============================================================
#  Step 2: Read recommendations
# ============================================================


def read_recommended(recommendations_file):
    """
    recommended_pytest_paths.txt contains one pytest path per line, e.g.:
        test/ops/test_matmul.py::test_bf16
        test/layers/test_attention.py
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
      Level 1 - File-level: recommended "test/foo.py" (no function)
                  matches failed "test/foo.py::anything"
      Level 2 - Exact: "test/foo.py::test_bar" in both lists

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


def _format_metric(metrics, thresholds, metric_key, threshold_key):
    """Render one metric as ``value (threshold X)``; n/a when absent."""
    value = metrics.get(metric_key)
    if value is None:
        return "n/a"
    threshold = thresholds.get(threshold_key)
    if threshold is None:
        return f"{value}"
    return f"{value} (threshold {threshold})"


def generate_report(
    failed,
    recommended,
    matched,
    log_dir,
    recommendations_source="none",
    diffusion_records=None,
):
    """Produce a Markdown summary table."""
    hit = matched["hit"]
    miss = matched["miss"]
    untested = matched["untested"]
    diffusion_records = diffusion_records or []

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
    #  Section 1b: Diffusion consistency failures
    # ================================================================
    if diffusion_records:
        out.append("---")
        out.append("")
        out.append(
            f"## Diffusion Consistency Failures（ {len(diffusion_records)} total）"
        )
        out.append("")
        out.append(
            "> From `diffusion-failures-*` artifacts (sglang/multimodal_gen "
            "consistency checks). These cases are outside the coverage "
            "recommendation scope."
        )
        out.append("")
        out.append(
            "| Case | GPUs | min CLIP | min SSIM | min PSNR | max mean abs diff | Artifact |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for record in diffusion_records:
            metrics = record["metrics"]
            thresholds = record["thresholds"]
            out.append(
                f"| `{record['case_id']}` | {record['num_gpus']} "
                f"| {_format_metric(metrics, thresholds, 'min_clip_similarity', 'clip_threshold')} "
                f"| {_format_metric(metrics, thresholds, 'min_ssim', 'ssim_threshold')} "
                f"| {_format_metric(metrics, thresholds, 'min_psnr', 'psnr_threshold')} "
                f"| {_format_metric(metrics, thresholds, 'max_mean_abs_diff', 'mean_abs_diff_threshold')} "
                f"| `{record['artifact']}` |"
            )
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
        "--log-dir", required=True, help="Directory containing CI .log files"
    )
    parser.add_argument(
        "--diffusion-dir",
        default=None,
        help="Directory containing downloaded diffusion-failures-* artifacts",
    )
    parser.add_argument(
        "--recommendations-file",
        help="Path to recommended_pytest_paths.txt",
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
    args = parser.parse_args()

    if not args.recommendations_file:
        parser.error("--recommendations-file is required")

    # For Windows console: force UTF-8 if possible
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 50)
    print("Step 1: Extract failed tests from CI logs")
    print("=" * 50)
    failed = extract_failed_from_logs(args.log_dir)
    diffusion_records = []
    if args.diffusion_dir:
        print()
        print("=" * 50)
        print("Step 1b: Extract diffusion consistency failures")
        print("=" * 50)
        diffusion_records = extract_diffusion_failures(args.diffusion_dir)
        print(f"Diffusion consistency failures: {len(diffusion_records)}")
    for record in diffusion_records:
        failure_id = diffusion_failure_id(record)
        if failure_id not in failed:
            failed.append(failure_id)

    has_log_inputs = _dir_has_files(args.log_dir)
    has_diffusion_inputs = bool(args.diffusion_dir) and _dir_has_files(
        args.diffusion_dir
    )
    if failed:
        print(f"Failed: {len(failed)}")
    elif not has_log_inputs and not has_diffusion_inputs:
        # Empty result from empty input is not a clean-pass signal; say so
        # instead of reporting a misleading 0.
        print("Failed: 0 (no input artifacts found)")
        print(
            "::warning::No input artifacts found (no test logs, no diffusion "
            "failures). This analysis has no data; it does not mean the run passed."
        )
    else:
        print("Failed: 0")

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
        failed,
        recommended,
        matched,
        args.log_dir,
        args.recommendations_source,
        diffusion_records=diffusion_records,
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
