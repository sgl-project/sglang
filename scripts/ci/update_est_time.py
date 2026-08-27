#!/usr/bin/env python3
"""Update est_time values in CI test files based on actual execution times.

Fetches logs from recent scheduled PR Test workflow runs on main,
parses per-file elapsed times from successful jobs, computes the 90th
percentile, and updates the est_time literals in test registration calls.

Usage:
    python scripts/ci/update_est_time.py [--dry-run] [--repo OWNER/REPO]
"""

import argparse
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Regex to extract per-file elapsed time from CI logs.
# Matches lines like:
#   filename='/actions-runner/_work/sglang/sglang/test/registered/core/test_x.py', elapsed=120, ...
#   filename='/actions-runner/_work/sglang/sglang/python/sglang/jit_kernel/tests/test_x.py', ...
LOG_PATTERN = re.compile(
    r"filename='[^']*?/sglang/((?:test|python)/[^']+\.py)', elapsed=(\d+),"
)

WORKFLOW_NAME = "PR Test"
MIN_DATA_POINTS = 3
TARGET_DATA_POINTS = 15
MAX_RUNS = 25

# A change is "significant" if |delta| >= this many seconds AND the relative
# change is at least SIGNIFICANT_REL_DELTA. Dual threshold filters out both
# tiny absolute drifts on long tests and small-but-noisy relative swings on
# short tests.
SIGNIFICANT_ABS_DELTA = 30
SIGNIFICANT_REL_DELTA = 0.3


def gh_api(endpoint, paginate=False):
    """Call gh api and return parsed JSON."""
    cmd = ["gh", "api", endpoint]
    if paginate:
        cmd.append("--paginate")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def gh_api_raw(endpoint):
    """Call gh api and return raw bytes (for log downloads)."""
    cmd = ["gh", "api", endpoint]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def get_workflow_id(repo):
    """Find the workflow ID for the PR Test workflow."""
    data = gh_api(f"/repos/{repo}/actions/workflows")
    for wf in data["workflows"]:
        if wf["name"] == WORKFLOW_NAME:
            return wf["id"]
    raise RuntimeError(f"Workflow '{WORKFLOW_NAME}' not found in {repo}")


def get_scheduled_runs(repo, workflow_id):
    """Get completed scheduled runs on main, newest first."""
    data = gh_api(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs"
        f"?branch=main&status=completed&event=schedule&per_page=100"
    )
    return data["workflow_runs"]


def get_successful_jobs(repo, run_id):
    """Get successful jobs for a given run."""
    data = gh_api(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")
    return [j for j in data["jobs"] if j["conclusion"] == "success"]


def job_name_to_suite(job_name):
    """Extract the suite name from a job name.

    Job names look like "stage-c-test-4-gpu-h100 (2)" or "stage-a-test-cpu".
    Strip the partition suffix " (N)" to get the suite name.
    """
    return re.sub(r"\s*\(\d+\)$", "", job_name)


def determine_backend(job_name):
    """Determine backend from job name."""
    name = job_name.lower()
    for backend in ["cpu", "amd", "npu"]:
        if backend in name:
            return backend
    return "cuda"


def parse_job_logs(repo, job_id):
    """Download and parse a job's logs for elapsed times.

    Returns list of (relative_path, elapsed_seconds) tuples.
    """
    try:
        raw = gh_api_raw(f"/repos/{repo}/actions/jobs/{job_id}/logs")
        text = raw.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        return []

    results = []
    for match in LOG_PATTERN.finditer(text):
        rel_path = match.group(1)
        elapsed = int(match.group(2))
        results.append((rel_path, elapsed))
    return results


def collect_timings(repo):
    """Collect per-file elapsed times from recent scheduled CI runs.

    Returns dict mapping (relative_path, suite, backend) -> list of elapsed
    times (newest first).
    """
    workflow_id = get_workflow_id(repo)
    print(f"Found workflow '{WORKFLOW_NAME}' (id={workflow_id})")

    runs = get_scheduled_runs(repo, workflow_id)
    print(f"Found {len(runs)} completed scheduled runs on main")

    # timings[(rel_path, suite, backend)] = [elapsed1, elapsed2, ...]
    timings = defaultdict(list)
    runs_processed = 0

    for run in runs:
        run_id = run["id"]
        jobs = get_successful_jobs(repo, run_id)
        if not jobs:
            continue

        runs_processed += 1
        test_jobs = [
            j
            for j in jobs
            if j["name"] != "check-changes" and "health" not in j["name"].lower()
        ]
        print(
            f"  Run {run_id} ({run['conclusion']}): "
            f"{len(test_jobs)} successful test jobs"
        )

        for job in test_jobs:
            suite = job_name_to_suite(job["name"])
            backend = determine_backend(job["name"])
            entries = parse_job_logs(repo, job["id"])
            for rel_path, elapsed in entries:
                key = (rel_path, suite, backend)
                timings[key].append(elapsed)

        if runs_processed >= MAX_RUNS:
            print(f"  Reached max {MAX_RUNS} runs, stopping collection")
            break

    print(
        f"\nProcessed {runs_processed} runs, "
        f"collected timings for {len(timings)} (file, suite, backend) pairs"
    )
    return timings


def compute_p90(timings):
    """Compute 90th percentile of last TARGET_DATA_POINTS timings for each entry.

    Returns dict mapping (rel_path, suite, backend) -> p90 (int).
    Only includes entries with >= MIN_DATA_POINTS data points.
    """
    p90s = {}
    for key, values in timings.items():
        recent = values[:TARGET_DATA_POINTS]
        if len(recent) < MIN_DATA_POINTS:
            continue
        p90s[key] = round(statistics.quantiles(recent, n=10, method="inclusive")[8])
    return p90s


def update_est_times(p90s, dry_run=False):
    """Update est_time values in source files.

    Each registration call is matched by both the function name and suite,
    so files with multiple registrations for different suites get the correct
    per-suite p90.

    Returns (updated_count, skipped_count, changes) where changes is a list
    of (rel_path, suite, backend, old_val, new_val) for each modified entry.
    """
    updated = 0
    skipped = 0
    changes = []

    # Group p90s by file: {rel_path: [(suite, backend, p90), ...]}
    by_file = defaultdict(list)
    for (rel_path, suite, backend), p90 in p90s.items():
        by_file[rel_path].append((suite, backend, p90))

    for rel_path, entries in sorted(by_file.items()):
        filepath = REPO_ROOT / rel_path
        if not filepath.exists():
            print(f"  SKIP {rel_path}: file not found")
            skipped += 1
            continue

        content = filepath.read_text()
        new_content = content

        for suite, backend, p90 in entries:
            # Match registration calls with this specific backend and suite.
            # Handles: register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-h100")
            pattern = re.compile(
                rf"(register_{backend}_ci\(est_time=)(\d+)"
                rf'(,\s*suite="{re.escape(suite)}")'
            )
            match = pattern.search(new_content)
            if not match:
                continue

            old_val = int(match.group(2))
            if old_val == p90:
                continue

            new_content = pattern.sub(rf"\g<1>{p90}\3", new_content)
            changes.append((rel_path, suite, backend, old_val, p90))
            print(
                f"  {rel_path}: register_{backend}_ci "
                f'suite="{suite}" est_time={old_val} -> {p90}'
            )

        if new_content != content:
            if not dry_run:
                filepath.write_text(new_content)
            updated += 1
        else:
            skipped += 1

    return updated, skipped, changes


def is_significant(old_val, new_val):
    """Return True if the change meets both absolute and relative thresholds."""
    delta = abs(new_val - old_val)
    return delta >= SIGNIFICANT_ABS_DELTA and delta / old_val >= SIGNIFICANT_REL_DELTA


def write_summary(changes, summary_file):
    """Write a markdown summary of significant est_time changes."""
    significant = [c for c in changes if is_significant(c[3], c[4])]
    significant.sort(key=lambda c: abs(c[4] - c[3]), reverse=True)

    lines = []
    if significant:
        lines.append(
            f"### Significant est_time changes "
            f"({len(significant)} of {len(changes)} updates)"
        )
        lines.append("")
        lines.append("| File | Suite | Old (s) | New (s) | Δ |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for rel_path, suite, _backend, old_val, new_val in significant:
            delta = new_val - old_val
            sign = "+" if delta > 0 else ""
            pct = round(delta / old_val * 100)
            lines.append(
                f"| `{Path(rel_path).name}` | `{suite}` | "
                f"{old_val} | {new_val} | {sign}{delta} ({sign}{pct}%) |"
            )
    else:
        lines.append(
            f"_{len(changes)} est_time update(s); none exceeded both "
            f"±{SIGNIFICANT_ABS_DELTA}s and "
            f"±{int(SIGNIFICANT_REL_DELTA * 100)}% thresholds._"
        )

    Path(summary_file).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Update est_time values from CI run data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without modifying files",
    )
    parser.add_argument(
        "--repo",
        default="sgl-project/sglang",
        help="GitHub repository (default: sgl-project/sglang)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Write a markdown summary of significant changes to this path",
    )
    args = parser.parse_args()

    print("Collecting timings from CI logs...")
    timings = collect_timings(args.repo)

    print("\nComputing 90th percentiles...")
    p90s = compute_p90(timings)
    print(f"Computed p90 for {len(p90s)} (file, suite, backend) entries")

    print("\nUpdating est_time values...")
    updated, skipped, changes = update_est_times(p90s, dry_run=args.dry_run)

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {updated} files, skipped {skipped} files")

    if args.summary_file:
        write_summary(changes, args.summary_file)
        print(f"Wrote summary to {args.summary_file}")

    if args.dry_run:
        print("(dry-run mode, no files modified)")


if __name__ == "__main__":
    main()
