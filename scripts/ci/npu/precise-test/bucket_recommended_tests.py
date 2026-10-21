#!/usr/bin/env python3
"""
Bucket coverage-recommended tests per CI suite.

Runs after test_selector.py has produced recommended_pytest_paths.txt (one
pytest path per line, repo-relative, optional ::case suffix). For each
requested suite, enumerate the suite's registered test files via list_tests.py
and intersect with the recommended files, then write per-suite GitHub Actions
step outputs:

    only_tests_<key>   multi-line list of intersected files (repo-relative);
                       exported non-empty ONLY in precision mode, so callers
                       can reference it without any extra guard expression
    should_run_<key>   'false' only when precision mode is active, the suite
                       enumerated a non-empty registered pool, and this
                       suite's bucket is empty -> the caller skips the whole
                       job; 'true' otherwise (including full-suite runs)

Suite keys replace '-' with '_' because GitHub Actions output names must be
valid shell identifiers. Any failure path yields empty/missing outputs, which
callers treat as "run the full suite".

When --github-summary points at $GITHUB_STEP_SUMMARY, a markdown section is
appended with the recommended test cases, counts, per-suite buckets, and the
files that matched no suite pool (not run by the bucketed a3 suites).

Usage:
    python3 bucket_recommended_tests.py \
        --recommended recommended_pytest_paths.txt \
        --suites base-a-test-npu,base-b-test-1-npu-a3 \
        --precision-mode true \
        --github-output "$GITHUB_OUTPUT" \
        --github-summary "$GITHUB_STEP_SUMMARY"
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Repo layout: this script lives at <repo>/scripts/ci/npu/precise-test/.
SCRIPT_DIR = Path(__file__).resolve().parent
LIST_TESTS = SCRIPT_DIR / "list_tests.py"
REPO_ROOT = SCRIPT_DIR.parents[3]


def normalize_rel(path: str) -> str:
    """Normalize a pytest path entry to a repo-relative posix file path.

    Strips an optional ::case suffix and the leading ./, converts absolute
    paths (any runner topology) to repo-relative.
    """
    p = path.strip().replace("\\", "/").removeprefix("./")
    p = p.split("::", 1)[0]
    if os.path.isabs(p):
        p = os.path.relpath(p, REPO_ROOT).replace(os.sep, "/")
    return p


def enumerate_suite_files(suite: str, work_dir: Path) -> list:
    """List the suite's registered PR-mode test files via list_tests.py."""
    out_file = work_dir / (suite.replace("-", "_") + ".txt")
    subprocess.run(
        [
            sys.executable,
            str(LIST_TESTS),
            "--hw",
            "npu",
            "--suite",
            suite,
            "-o",
            str(out_file),
        ],
        check=True,
    )
    with open(out_file) as f:
        return [line.strip() for line in f if line.strip()]


def write_github_summary(
    summary_file: str,
    raw_entries: list,
    recommended: set,
    all_pool: set,
    buckets: dict,
    suite_pool_sizes: dict,
    precision_mode: bool,
) -> None:
    """Append a markdown section with the recommendation results to the
    GitHub Actions step summary ($GITHUB_STEP_SUMMARY)."""
    unbucketed = sorted(recommended - all_pool)
    lines = []
    lines.append("### Recommended tests from coverage")
    lines.append("")
    if precision_mode:
        lines.append(
            "Precision mode **on** — a3 suites run only the bucketed subset below."
        )
    else:
        lines.append("Precision mode **off** — suites run in full.")
    lines.append("")
    lines.append(
        f"Recommended **{len(raw_entries)}** test cases "
        f"({len(recommended)} unique files); "
        f"{len(unbucketed)} match no suite pool."
    )
    lines.append("")
    lines.append("| Suite | Registered files | Bucketed files | Action |")
    lines.append("|---|---|---|---|")
    for suite, files in buckets.items():
        if precision_mode and suite_pool_sizes[suite] and not files:
            action = "skipped (no recommended test)"
        elif precision_mode and files:
            action = "run subset"
        else:
            action = "run full suite"
        lines.append(
            f"| {suite} | {suite_pool_sizes[suite]} | {len(files)} | {action} |"
        )
    if raw_entries:
        lines.append("")
        lines.append(
            f"<details><summary>Recommended test cases ({len(raw_entries)})</summary>"
        )
        lines.append("")
        lines.append("```")
        lines.extend(raw_entries)
        lines.append("```")
        lines.append("</details>")
    if unbucketed:
        lines.append("")
        lines.append(
            f"<details><summary>Matched no suite pool — not run by a3 suites ({len(unbucketed)})</summary>"
        )
        lines.append("")
        lines.append("```")
        lines.extend(unbucketed)
        lines.append("```")
        lines.append("")
        lines.append(
            "Not registered in any of the 9 bucketed a3 suites — they may still "
            "run elsewhere (e.g. base-a runs its full suite; nightly tests run "
            "in nightly pipelines)."
        )
        lines.append("</details>")
    if any(buckets.values()):
        lines.append("")
        lines.append("<details><summary>Bucketed test files per suite</summary>")
        lines.append("")
        for suite, files in buckets.items():
            lines.append(f"**{suite} ({len(files)})**")
            lines.append("")
            lines.extend(f"- {f}" for f in files)
            lines.append("")
        lines.append("</details>")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary written to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Bucket coverage-recommended tests per CI suite."
    )
    parser.add_argument(
        "--recommended",
        type=str,
        default=str(SCRIPT_DIR / "recommended_pytest_paths.txt"),
        help="recommended_pytest_paths.txt written by test_selector.py",
    )
    parser.add_argument(
        "--suites",
        type=str,
        required=True,
        help="Comma-separated suite names to bucket (e.g. base-a-test-npu,base-b-test-1-npu-a3)",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default="false",
        help="'true' to enable precision mode (run only the recommended tests)",
    )
    parser.add_argument(
        "--github-output",
        type=str,
        default="",
        help="GITHUB_OUTPUT file to write step outputs to (empty = print only)",
    )
    parser.add_argument(
        "--github-summary",
        type=str,
        default="",
        help="GITHUB_STEP_SUMMARY file to append a markdown report to (empty = skip)",
    )
    args = parser.parse_args()

    # Recommended set: file-level, repo-relative. raw_entries keeps the
    # original case-level lines (with optional ::case suffix) for the summary.
    rec_file = Path(args.recommended)
    recommended = set()
    raw_entries = []
    if rec_file.exists():
        with open(rec_file) as f:
            raw_entries = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("ERROR")
            ]
            recommended = {normalize_rel(line) for line in raw_entries}
    else:
        print(f"::warning::Recommended file not found: {rec_file}")
    print(f"Recommended test entries: {len(raw_entries)} ({len(recommended)} files)")

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    work_dir = Path(tempfile.mkdtemp(prefix="bucket_tests_"))

    buckets = {}
    # Registered-pool size per suite. An empty ENUMERATION (0 files) must
    # never let precision mode skip the whole job — that would silently drop
    # the suite from the PR gate — so the output loop falls back to a
    # full-suite run in that case.
    suite_pool_sizes = {}
    all_pool = set()
    for suite in suites:
        suite_files = {normalize_rel(f) for f in enumerate_suite_files(suite, work_dir)}
        suite_pool_sizes[suite] = len(suite_files)
        all_pool |= suite_files
        if not suite_files:
            print(f"::warning::Suite {suite}: enumerated 0 registered test files")
        intersection = sorted(suite_files & recommended)
        buckets[suite] = intersection
        print(
            f"Suite {suite}: {len(suite_files)} files, {len(intersection)} recommended"
        )

    total_count = sum(len(b) for b in buckets.values())
    precision_mode = args.precision_mode == "true" and total_count > 0
    print(f"total_count={total_count} precision_mode={precision_mode}")

    # Outputs first, summary last: a summary failure must never cost the
    # downstream jobs their bucket outputs (they'd fall back to full suites).
    if args.github_output:
        with open(args.github_output, "a") as out:
            for suite, files in buckets.items():
                key = suite.replace("-", "_")
                # Gate bucket contents on precision_mode: a run without precision
                # mode enabled never exports a non-empty bucket, so downstream
                # jobs can reference only_tests_<key> without any extra guard
                # expression.
                bucket_files = files if precision_mode else []
                out.write(f"only_tests_{key}<<EOF\n")
                for line in bucket_files:
                    out.write(line + "\n")
                out.write("EOF\n")
                # 'false' ONLY when precision mode is on, the suite enumerated a
                # non-empty registered pool, and this bucket is empty -> the
                # caller skips the whole suite job. An empty ENUMERATION (0
                # registered files, e.g. a list_tests/run_suite filter drift)
                # falls back to 'true' so the job runs the full suite instead of
                # silently vanishing. Anything else ('true', or a missing output
                # from a failed step) means full-suite run.
                should_run = (
                    "false"
                    if (precision_mode and suite_pool_sizes[suite] and not files)
                    else "true"
                )
                out.write(f"should_run_{key}={should_run}\n")

    if args.github_summary:
        write_github_summary(
            args.github_summary,
            raw_entries,
            recommended,
            all_pool,
            buckets,
            suite_pool_sizes,
            precision_mode,
        )


if __name__ == "__main__":
    main()
