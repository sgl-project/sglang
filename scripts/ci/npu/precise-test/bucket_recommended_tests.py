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

Usage:
    python3 bucket_recommended_tests.py \
        --recommended recommended_pytest_paths.txt \
        --suites base-a-test-npu,base-b-test-1-npu-a3 \
        --precision-mode true \
        --github-output "$GITHUB_OUTPUT"
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
    args = parser.parse_args()

    # Recommended set: file-level, repo-relative.
    rec_file = Path(args.recommended)
    recommended = set()
    if rec_file.exists():
        with open(rec_file) as f:
            recommended = {
                normalize_rel(line)
                for line in f
                if line.strip() and not line.startswith("ERROR")
            }
    else:
        print(f"::warning::Recommended file not found: {rec_file}")
    print(f"Recommended test files: {len(recommended)}")

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    work_dir = Path(tempfile.mkdtemp(prefix="bucket_tests_"))

    buckets = {}
    # Registered-pool size per suite. An empty ENUMERATION (0 files) must
    # never let precision mode skip the whole job — that would silently drop
    # the suite from the PR gate — so the output loop falls back to a
    # full-suite run in that case.
    suite_pool_sizes = {}
    for suite in suites:
        suite_files = {normalize_rel(f) for f in enumerate_suite_files(suite, work_dir)}
        suite_pool_sizes[suite] = len(suite_files)
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

    if not args.github_output:
        return

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


if __name__ == "__main__":
    main()
