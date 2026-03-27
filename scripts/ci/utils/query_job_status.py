#!/usr/bin/env python3
"""
Query GitHub Actions job status for specific jobs or generate runner fleet reports.

Usage:
    # Per-job reports (original mode)
    python scripts/ci/utils/query_job_status.py --job "stage-c-test-large-8-gpu-amd-mi35x"
    python scripts/ci/utils/query_job_status.py --job "stage-c-test-large-8-gpu-amd-mi35x" --hours 48

    # Runner fleet report (cross-workflow runner analytics)
    python scripts/ci/utils/query_job_status.py --runner-report --workflow "pr-test-amd.yml,nightly-test-amd.yml" --hours 24
    python scripts/ci/utils/query_job_status.py --runner-report --workflow "pr-test-amd.yml,nightly-test-amd.yml,pr-test-amd-rocm720.yml,nightly-test-amd-rocm720.yml" --summary

Requirements:
    pip install tabulate
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from tabulate import tabulate
except ImportError:
    print("Please install tabulate: pip install tabulate")
    exit(1)


def check_gh_cli_available() -> bool:
    """Check if gh CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False

        # Check if authenticated
        auth_result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
        )
        if auth_result.returncode != 0:
            print(
                "Error: gh CLI is not authenticated. Please run 'gh auth login' first.",
                file=sys.stderr,
            )
            print(f"Details: {auth_result.stderr}", file=sys.stderr)
            return False

        return True
    except FileNotFoundError:
        print(
            "Error: gh CLI is not installed. Please install it from https://cli.github.com/",
            file=sys.stderr,
        )
        return False


def run_gh_command(args: list[str]) -> dict:
    """Run gh CLI command and return JSON result."""
    try:
        result = subprocess.run(
            ["gh", "api"] + args,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise Exception("gh CLI not found. Please install from https://cli.github.com/")

    if result.returncode != 0:
        raise Exception(f"gh api failed: {result.stderr}")
    return json.loads(result.stdout)


def parse_time(time_str: str) -> Optional[datetime]:
    """Parse ISO timestamp to datetime."""
    if not time_str:
        return None
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def format_time(time_str: str) -> str:
    """Format ISO timestamp to readable format in UTC."""
    if not time_str:
        return "-"
    dt = parse_time(time_str)
    if dt:
        # Ensure UTC
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%m-%d %H:%M")
    return "-"


def get_workflow_runs(repo: str, workflow: str, hours: int = 24) -> list[dict]:
    """Get workflow runs from the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    runs = []
    page = 1
    while True:
        url = f"repos/{repo}/actions/runs?per_page=100&page={page}"
        if workflow:
            url = f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=100&page={page}"

        data = run_gh_command([url])
        page_runs = data.get("workflow_runs", [])

        for run in page_runs:
            created_at = parse_time(run.get("created_at"))
            if created_at and created_at >= since:
                runs.append(run)
            elif created_at and created_at < since:
                return runs

        if len(page_runs) < 100:
            break
        page += 1
        if page > 20:
            break
    return runs


def get_jobs_for_run(repo: str, run_id: int) -> list[dict]:
    """Get all jobs for a workflow run."""
    jobs = []
    page = 1
    while True:
        data = run_gh_command(
            [f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"]
        )
        jobs.extend(data.get("jobs", []))
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
        if page > 5:
            break
    return jobs


def get_pr_number_from_run(run: dict) -> Optional[int]:
    """Extract PR number from run data."""
    # Try to get from pull_requests array
    prs = run.get("pull_requests", [])
    if prs:
        return prs[0].get("number")
    return None


def query_jobs(
    repo: str,
    job_filter: str,
    workflow: str = None,
    hours: int = 24,
    status_filter: str = None,
) -> list[dict]:
    """Query jobs matching the filter."""

    print(f"Fetching workflow runs from last {hours} hours...", file=sys.stderr)
    runs = get_workflow_runs(repo, workflow, hours)
    print(f"Found {len(runs)} workflow runs", file=sys.stderr)

    results = []
    total_runs = len(runs)

    for i, run in enumerate(runs):
        if (i + 1) % 20 == 0:
            print(f"Processing run {i+1}/{total_runs}...", file=sys.stderr)

        try:
            jobs = get_jobs_for_run(repo, run["id"])
        except Exception as e:
            print(
                f"Warning: Failed to get jobs for run {run['id']}: {e}", file=sys.stderr
            )
            continue

        pr_number = get_pr_number_from_run(run)
        branch = run.get("head_branch", "")
        run_status = run.get("status", "unknown")
        run_conclusion = run.get("conclusion") or "-"

        for job in jobs:
            job_name = job.get("name", "")

            # Filter by job name
            # Use prefix matching to avoid e.g. "stage-c-test-large-8-gpu-amd"
            # also matching "stage-c-test-large-8-gpu-amd-mi35x"
            job_name_lower = job_name.lower()
            filter_lower = job_filter.lower()
            if not job_name_lower.startswith(filter_lower):
                continue
            # If there are characters after the filter, ensure it's not a
            # continuation of the base job name (e.g., "-mi35x")
            if len(job_name_lower) > len(filter_lower):
                next_char = job_name_lower[len(filter_lower)]
                if next_char not in (" ", "("):
                    continue

            # Filter by status if specified
            if status_filter and job.get("status") != status_filter:
                continue

            job_status = job.get("status", "unknown")
            runner_name = job.get("runner_name") or "-"

            # Detect stuck/ghost jobs:
            # - Job is in_progress but no runner assigned
            # - Job is in_progress but workflow run is cancelled/completed
            is_stuck = False
            if job_status == "in_progress":
                if runner_name == "-":
                    is_stuck = True
                elif run_status == "completed" and run_conclusion in (
                    "cancelled",
                    "failure",
                ):
                    is_stuck = True

            results.append(
                {
                    "job_name": job_name,
                    "status": job_status,
                    "conclusion": job.get("conclusion") or "-",
                    "created_at": job.get("created_at", ""),
                    "started_at": job.get("started_at", ""),
                    "completed_at": job.get("completed_at", ""),
                    "runner_name": runner_name,
                    "labels": job.get("labels", []),
                    "runner_group_name": job.get("runner_group_name") or "-",
                    "run_id": run["id"],
                    "run_status": run_status,
                    "run_conclusion": run_conclusion,
                    "pr_number": pr_number,
                    "branch": branch,
                    "html_url": job.get("html_url", ""),
                    "is_stuck": is_stuck,
                }
            )

    return results


def query_all_jobs(
    repo: str,
    workflows: list[str],
    hours: int = 24,
) -> list[dict]:
    """Query all jobs across multiple workflows for fleet-level analysis.

    Unlike query_jobs(), this does NOT filter by job name and collects
    everything in a single pass -- ideal for runner-centric analytics.
    Jobs on ubuntu-latest are excluded since those are utility jobs.
    """
    all_runs = []
    for workflow in workflows:
        print(f"Fetching runs for {workflow}...", file=sys.stderr)
        runs = get_workflow_runs(repo, workflow, hours)
        print(f"  Found {len(runs)} runs for {workflow}", file=sys.stderr)
        for run in runs:
            run["_workflow"] = workflow
        all_runs.extend(runs)

    seen_run_ids = set()
    unique_runs = []
    for run in all_runs:
        if run["id"] not in seen_run_ids:
            seen_run_ids.add(run["id"])
            unique_runs.append(run)

    print(f"Total unique workflow runs: {len(unique_runs)}", file=sys.stderr)

    results = []
    total_runs = len(unique_runs)

    for i, run in enumerate(unique_runs):
        if (i + 1) % 20 == 0:
            print(f"Processing run {i+1}/{total_runs}...", file=sys.stderr)

        try:
            jobs = get_jobs_for_run(repo, run["id"])
        except Exception as e:
            print(
                f"Warning: Failed to get jobs for run {run['id']}: {e}", file=sys.stderr
            )
            continue

        pr_number = get_pr_number_from_run(run)
        branch = run.get("head_branch", "")
        run_status = run.get("status", "unknown")
        run_conclusion = run.get("conclusion") or "-"
        workflow_name = run.get("_workflow", "-")

        for job in jobs:
            job_name = job.get("name", "")
            job_status = job.get("status", "unknown")
            runner_name = job.get("runner_name") or "-"
            labels = job.get("labels", [])

            if len(labels) == 1 and labels[0] == "ubuntu-latest":
                continue

            is_stuck = False
            if job_status == "in_progress":
                if runner_name == "-":
                    is_stuck = True
                elif run_status == "completed" and run_conclusion in (
                    "cancelled",
                    "failure",
                ):
                    is_stuck = True

            results.append(
                {
                    "job_name": job_name,
                    "status": job_status,
                    "conclusion": job.get("conclusion") or "-",
                    "created_at": job.get("created_at", ""),
                    "started_at": job.get("started_at", ""),
                    "completed_at": job.get("completed_at", ""),
                    "runner_name": runner_name,
                    "labels": labels,
                    "runner_group_name": job.get("runner_group_name") or "-",
                    "run_id": run["id"],
                    "run_status": run_status,
                    "run_conclusion": run_conclusion,
                    "pr_number": pr_number,
                    "branch": branch,
                    "html_url": job.get("html_url", ""),
                    "is_stuck": is_stuck,
                    "workflow": workflow_name,
                }
            )

    return results


def calculate_duration(started_at: str, completed_at: str) -> str:
    """Calculate duration between start and completion."""
    if not started_at or not completed_at:
        return "-"
    start = parse_time(started_at)
    end = parse_time(completed_at)
    if start and end:
        duration = (end - start).total_seconds()
        if duration < 0:
            return "-"  # Invalid data, skip
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}h{minutes}m"
        return f"{minutes}m{seconds}s"
    return "-"


def calculate_queue_time(
    created_at: str,
    started_at: str,
    status: str = None,
    report_time: datetime = None,
) -> str:
    """
    Calculate queue time between creation and start.

    For queued/waiting jobs that haven't truly started yet, calculate
    queue time as (report_time - created_at) and mark as "still queuing".
    """
    if not created_at:
        return "-"

    created = parse_time(created_at)
    if not created:
        return "-"

    # For queued/waiting jobs, calculate time since creation
    if status in ("queued", "waiting"):
        if report_time:
            queue_seconds = (report_time - created).total_seconds()
        else:
            queue_seconds = (datetime.now(timezone.utc) - created).total_seconds()

        if queue_seconds < 0:
            return "-"

        minutes = int(queue_seconds // 60)
        seconds = int(queue_seconds % 60)
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}h{minutes}m (queuing)"
        return f"{minutes}m{seconds}s (queuing)"

    # For completed/in_progress jobs, calculate actual queue time
    if not started_at:
        return "-"

    started = parse_time(started_at)
    if not started:
        return "-"

    queue_seconds = (started - created).total_seconds()
    if queue_seconds < 0:
        return "-"  # Invalid data

    minutes = int(queue_seconds // 60)
    seconds = int(queue_seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h{minutes}m"
    return f"{minutes}m{seconds}s"


# ---------------------------------------------------------------------------
# Runner fleet analytics functions
# ---------------------------------------------------------------------------


def _format_duration_seconds(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds <= 0:
        return "-"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h{minutes}m"
    return f"{minutes}m{secs}s"


def _get_runner_label(job: dict) -> str:
    """Extract the primary runner label from a job's labels list."""
    labels = job.get("labels", [])
    if not labels:
        return "unknown"
    for label in labels:
        if label.startswith("linux-mi"):
            return label
    return labels[0]


def analyze_concurrency(jobs: list[dict], report_time: datetime = None) -> dict:
    """Analyze concurrent runner usage per runner label.

    Uses an event-sweep algorithm: for each job that ran, create +1 event
    at started_at and -1 event at completed_at, then sweep through sorted
    events tracking the concurrent count.
    """
    if report_time is None:
        report_time = datetime.now(timezone.utc)

    label_jobs: dict[str, list[dict]] = {}
    for job in jobs:
        label = _get_runner_label(job)
        label_jobs.setdefault(label, []).append(job)

    results = {}
    for label in sorted(label_jobs):
        pool_jobs = label_jobs[label]
        events: list[tuple[datetime, int]] = []
        queue_times: list[float] = []
        durations: list[float] = []

        for job in pool_jobs:
            started = parse_time(job.get("started_at", ""))
            completed = parse_time(job.get("completed_at", ""))
            created = parse_time(job.get("created_at", ""))

            if started and completed:
                events.append((started, +1))
                events.append((completed, -1))
                durations.append((completed - started).total_seconds())
            elif started:
                events.append((started, +1))
                events.append((report_time, -1))
                durations.append((report_time - started).total_seconds())

            if created and started:
                qt = (started - created).total_seconds()
                if qt >= 0:
                    queue_times.append(qt)

        if not events:
            results[label] = {
                "peak": 0,
                "avg_concurrent": 0.0,
                "total_jobs": len(pool_jobs),
                "avg_queue_seconds": 0,
                "avg_duration_seconds": 0,
            }
            continue

        events.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        peak = 0
        time_weighted_sum = 0.0
        total_time = 0.0
        prev_time = events[0][0]

        for ts, delta in events:
            if prev_time and concurrent > 0:
                dt = (ts - prev_time).total_seconds()
                time_weighted_sum += concurrent * dt
                total_time += dt
            concurrent += delta
            peak = max(peak, concurrent)
            prev_time = ts

        avg_concurrent = time_weighted_sum / total_time if total_time > 0 else 0
        avg_queue = sum(queue_times) / len(queue_times) if queue_times else 0
        avg_duration = sum(durations) / len(durations) if durations else 0

        results[label] = {
            "peak": peak,
            "avg_concurrent": round(avg_concurrent, 1),
            "total_jobs": len(pool_jobs),
            "avg_queue_seconds": avg_queue,
            "avg_duration_seconds": avg_duration,
        }

    return results


def analyze_busy_periods(jobs: list[dict]) -> list[dict]:
    """Analyze job activity by hour of day (UTC).

    Buckets jobs by the UTC hour they started and computes avg queue time.
    Classifies each hour as Quiet / Moderate / Busy / Peak relative to the
    busiest hour.
    """
    hourly: dict[int, dict] = {
        h: {"jobs_started": 0, "queue_times": []} for h in range(24)
    }

    for job in jobs:
        started = parse_time(job.get("started_at", ""))
        created = parse_time(job.get("created_at", ""))

        if started:
            hour = started.astimezone(timezone.utc).hour
            hourly[hour]["jobs_started"] += 1

            if created:
                qt = (started - created).total_seconds()
                if qt >= 0:
                    hourly[hour]["queue_times"].append(qt)

    max_jobs = max((v["jobs_started"] for v in hourly.values()), default=1) or 1

    results = []
    for hour in range(24):
        data = hourly[hour]
        avg_queue = (
            sum(data["queue_times"]) / len(data["queue_times"])
            if data["queue_times"]
            else 0
        )
        ratio = data["jobs_started"] / max_jobs
        if ratio >= 0.75:
            load = "Peak"
        elif ratio >= 0.5:
            load = "Busy"
        elif ratio >= 0.25:
            load = "Moderate"
        else:
            load = "Quiet"

        results.append(
            {
                "hour": hour,
                "hour_label": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
                "jobs_started": data["jobs_started"],
                "avg_queue_seconds": avg_queue,
                "load": load,
            }
        )

    return results


def analyze_runner_health(jobs: list[dict]) -> list[dict]:
    """Analyze per-runner-hostname health metrics.

    Returns a list sorted by failure rate (descending), showing total jobs,
    success/failure/cancelled counts, avg duration, and the most-failed job
    name per runner.
    """
    runners: dict[str, dict] = {}
    for job in jobs:
        runner = job.get("runner_name", "-")
        if runner == "-":
            continue

        if runner not in runners:
            runners[runner] = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "cancelled": 0,
                "durations": [],
                "failed_jobs": {},
                "labels": set(),
            }

        r = runners[runner]
        r["total"] += 1

        for lbl in job.get("labels", []):
            r["labels"].add(lbl)

        conclusion = job.get("conclusion", "-")
        if conclusion == "success":
            r["success"] += 1
        elif conclusion == "failure":
            r["failure"] += 1
            jn = job.get("job_name", "unknown")
            r["failed_jobs"][jn] = r["failed_jobs"].get(jn, 0) + 1
        elif conclusion in ("cancelled", "timed_out"):
            r["cancelled"] += 1

        started = parse_time(job.get("started_at", ""))
        completed = parse_time(job.get("completed_at", ""))
        if started and completed:
            r["durations"].append((completed - started).total_seconds())

    results = []
    for runner_name, data in runners.items():
        fail_rate = data["failure"] / data["total"] * 100 if data["total"] > 0 else 0
        avg_dur = (
            sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0
        )

        if data["failed_jobs"]:
            most_failed = max(data["failed_jobs"].items(), key=lambda x: x[1])
            most_failed_str = f"{most_failed[0]} ({most_failed[1]})"
        else:
            most_failed_str = "-"

        results.append(
            {
                "runner_name": runner_name,
                "total": data["total"],
                "success": data["success"],
                "failure": data["failure"],
                "cancelled": data["cancelled"],
                "fail_rate": round(fail_rate, 1),
                "avg_duration_seconds": avg_dur,
                "most_failed_job": most_failed_str,
                "labels": sorted(data["labels"]),
            }
        )

    results.sort(key=lambda x: (-x["fail_rate"], -x["total"]))
    return results


def analyze_queue_distribution(jobs: list[dict]) -> dict:
    """Analyze queue time distribution with percentile stats.

    Returns bucket counts and median/P90/P99 queue times.
    """
    queue_times: list[float] = []
    for job in jobs:
        created = parse_time(job.get("created_at", ""))
        started = parse_time(job.get("started_at", ""))
        if created and started:
            qt = (started - created).total_seconds()
            if qt >= 0:
                queue_times.append(qt)

    if not queue_times:
        return {"buckets": [], "median": 0, "p90": 0, "p99": 0, "total": 0}

    queue_times.sort()

    def percentile(data: list[float], p: int) -> float:
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]

    bucket_defs = [
        ("< 1 min", 0, 60),
        ("1-5 min", 60, 300),
        ("5-15 min", 300, 900),
        ("15-30 min", 900, 1800),
        ("30-60 min", 1800, 3600),
        ("> 60 min", 3600, float("inf")),
    ]

    total = len(queue_times)
    buckets = []
    for label, lo, hi in bucket_defs:
        count = sum(1 for qt in queue_times if lo <= qt < hi)
        pct = count / total * 100 if total > 0 else 0
        buckets.append({"range": label, "count": count, "percentage": round(pct, 1)})

    return {
        "buckets": buckets,
        "median": percentile(queue_times, 50),
        "p90": percentile(queue_times, 90),
        "p99": percentile(queue_times, 99),
        "total": total,
    }


def process_results(
    results: list[dict], repo: str, report_time: datetime = None
) -> dict:
    """
    Process raw results into structured data for presentation.
    Returns a dictionary containing:
    - status_summary: dict of job_name -> status counts
    - sorted_results: list of results sorted by created_at descending
    - active_jobs: list of in_progress/queued/waiting jobs (excluding stuck)
    - stuck_jobs: list of stuck/ghost jobs
    - failed_jobs: list of failed jobs
    - processed_jobs: list of jobs with calculated fields (queue_time, duration, etc.)
    """
    if report_time is None:
        report_time = datetime.now(timezone.utc)

    if not results:
        return {
            "status_summary": {},
            "sorted_results": [],
            "active_jobs": [],
            "stuck_jobs": [],
            "failed_jobs": [],
            "processed_jobs": [],
        }

    # Group by job name for summary
    status_summary = {}
    for r in results:
        job_name = r["job_name"]
        status = r["status"]
        conclusion = r.get("conclusion", "-")
        is_stuck = r.get("is_stuck", False)
        if job_name not in status_summary:
            status_summary[job_name] = {
                "in_progress": 0,
                "queued": 0,
                "waiting": 0,
                "stuck": 0,
                "success": 0,
                "failure": 0,
                "cancelled": 0,
            }
        if is_stuck:
            status_summary[job_name]["stuck"] += 1
        elif status == "completed":
            # For completed jobs, count by conclusion
            if conclusion == "success":
                status_summary[job_name]["success"] += 1
            elif conclusion == "failure":
                status_summary[job_name]["failure"] += 1
            elif conclusion in ("cancelled", "timed_out", "action_required"):
                status_summary[job_name]["cancelled"] += 1
        elif status in status_summary[job_name]:
            status_summary[job_name][status] += 1

    # Sort by created_at descending
    sorted_results = sorted(results, key=lambda x: x["created_at"], reverse=True)

    # Filter into categories (mutually exclusive)
    active_jobs = [
        r
        for r in results
        if r.get("status") in ("in_progress", "queued", "waiting")
        and not r.get("is_stuck", False)
    ]
    stuck_jobs = [r for r in results if r.get("is_stuck", False)]
    # Only include jobs with conclusion "failure"
    # Exclude stuck jobs to avoid double-counting
    failed_jobs = [
        r
        for r in results
        if r.get("conclusion", "-") == "failure" and not r.get("is_stuck", False)
    ]

    # Process jobs with calculated fields
    processed_jobs = []
    for r in sorted_results:
        processed = r.copy()
        processed["created_formatted"] = format_time(r["created_at"])
        processed["started_formatted"] = format_time(r["started_at"])
        processed["queue_time"] = calculate_queue_time(
            r["created_at"], r["started_at"], r["status"], report_time
        )
        processed["duration"] = calculate_duration(r["started_at"], r["completed_at"])
        # Use the job's html_url for direct link to the specific job
        processed["url"] = (
            r.get("html_url") or f"https://github.com/{repo}/actions/runs/{r['run_id']}"
        )

        if r["pr_number"]:
            processed["pr_info"] = f"PR#{r['pr_number']}"
        else:
            processed["pr_info"] = r["branch"] if r["branch"] else "-"

        # Status display with stuck marker
        if r.get("is_stuck", False):
            processed["status_display"] = f"STUCK ({r['status']})"
        else:
            processed["status_display"] = r["status"]

        processed_jobs.append(processed)

    return {
        "status_summary": status_summary,
        "sorted_results": sorted_results,
        "active_jobs": active_jobs,
        "stuck_jobs": stuck_jobs,
        "failed_jobs": failed_jobs,
        "processed_jobs": processed_jobs,
    }


def print_table(
    results: list[dict], repo: str, generated_time: str, report_time: datetime = None
):
    """Print results as a formatted table using tabulate."""
    print("")
    print(f"Report generated: {generated_time} UTC")
    print("Note: All times are in UTC")
    print("")

    if not results:
        print("No jobs found matching the filter.")
        return

    # Process data
    data = process_results(results, repo, report_time)
    status_summary = data["status_summary"]
    processed_jobs = data["processed_jobs"]
    active_jobs = data["active_jobs"]
    stuck_jobs = data["stuck_jobs"]

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY BY JOB NAME")
    print("=" * 100)

    summary_data = []
    for job_name, counts in sorted(status_summary.items()):
        summary_data.append(
            [
                job_name,
                counts["in_progress"],
                counts["queued"],
                counts["waiting"],
                counts["stuck"],
                counts["success"],
                counts["failure"],
                counts["cancelled"],
            ]
        )

    print(
        tabulate(
            summary_data,
            headers=[
                "Job Name",
                "Running",
                "Queued",
                "Waiting",
                "Stuck",
                "Success",
                "Failure",
                "Cancelled",
            ],
            tablefmt="grid",
        )
    )

    # Print detailed table
    print("\n" + "=" * 100)
    print("DETAILED JOB LIST")
    print("=" * 100)

    detail_data = []
    for p in processed_jobs:
        detail_data.append(
            [
                p["job_name"],
                p["status_display"],
                p["conclusion"],
                p["created_formatted"],
                p["started_formatted"],
                p["queue_time"],
                p["duration"],
                p["runner_name"] or "-",
                p["pr_info"],
                p["run_id"],
            ]
        )

    print(
        tabulate(
            detail_data,
            headers=[
                "Job Name",
                "Status",
                "Conclusion",
                "Created",
                "Started",
                "Queue",
                "Duration",
                "Runner",
                "PR/Branch",
                "Run ID",
            ],
            tablefmt="grid",
        )
    )

    # Print links for active jobs (use processed_jobs for correct queue_time)
    if active_jobs:
        print("\n" + "=" * 100)
        print("ACTIVE JOB LINKS")
        print("=" * 100)

        link_data = []
        for r in active_jobs:
            # Find the corresponding processed job to get pre-calculated fields
            p = next(
                (
                    p
                    for p in processed_jobs
                    if p["run_id"] == r["run_id"] and p["job_name"] == r["job_name"]
                ),
                None,
            )
            if p:
                link_data.append(
                    [
                        p["job_name"],
                        p["status"],
                        p["queue_time"],
                        p["pr_info"],
                        p["runner_name"] or "-",
                        p["url"],
                    ]
                )

        print(
            tabulate(
                link_data,
                headers=["Job Name", "Status", "Queue", "PR/Branch", "Runner", "URL"],
                tablefmt="simple",
            )
        )

    # Print stuck jobs (use processed_jobs for correct data)
    if stuck_jobs:
        print("\n" + "=" * 100)
        print("STUCK/GHOST JOBS (in_progress but no runner or workflow cancelled)")
        print("=" * 100)

        stuck_data = []
        for r in stuck_jobs:
            # Find the corresponding processed job
            p = next(
                (
                    p
                    for p in processed_jobs
                    if p["run_id"] == r["run_id"] and p["job_name"] == r["job_name"]
                ),
                None,
            )
            if p:
                run_info = f"{r.get('run_status', '-')}/{r.get('run_conclusion', '-')}"
                stuck_data.append(
                    [
                        p["job_name"],
                        p["status"],
                        run_info,
                        p["pr_info"],
                        p["runner_name"] or "-",
                        p["url"],
                    ]
                )

        print(
            tabulate(
                stuck_data,
                headers=[
                    "Job Name",
                    "Job Status",
                    "Run Status/Conclusion",
                    "PR/Branch",
                    "Runner",
                    "URL",
                ],
                tablefmt="simple",
            )
        )


def format_markdown(
    results: list[dict],
    repo: str,
    job_filter: str,
    hours: int,
    generated_time: str,
    report_time: datetime = None,
) -> str:
    """Format results as markdown for GitHub Actions summary."""
    lines = []

    # Header
    lines.append(f"# Job Status Report: `{job_filter}`")
    lines.append("")
    lines.append(f"**Time window:** Last {hours} hours")
    lines.append(f"**Generated:** {generated_time} UTC")
    lines.append(f"**Total jobs found:** {len(results)}")
    lines.append("")
    lines.append("> **Note:** All times are displayed in UTC")
    lines.append("")

    if not results:
        lines.append("> No jobs found matching the filter.")
        return "\n".join(lines)

    # Process data using shared function
    data = process_results(results, repo, report_time)
    status_summary = data["status_summary"]
    processed_jobs = data["processed_jobs"]
    active_jobs = data["active_jobs"]
    stuck_jobs = data["stuck_jobs"]
    failed_jobs = data["failed_jobs"]

    # Summary table
    lines.append("## Summary by Job Name")
    lines.append("")
    lines.append(
        "> **Status meanings:** Running = executing, Queued = waiting for runner, Waiting = waiting for dependent jobs, Stuck = ghost job, Cancelled = cancelled/timed_out"
    )
    lines.append("")
    lines.append(
        "| Job Name | Running | Queued | Waiting | Stuck | Success | Failure | Cancelled |"
    )
    lines.append(
        "|----------|---------|--------|---------|-------|---------|---------|-----------|"
    )

    for job_name, counts in sorted(status_summary.items()):
        running = f"**{counts['in_progress']}**" if counts["in_progress"] > 0 else "0"
        queued = f"**{counts['queued']}**" if counts["queued"] > 0 else "0"
        waiting = f"**{counts['waiting']}**" if counts["waiting"] > 0 else "0"
        stuck = f"**{counts['stuck']}**" if counts["stuck"] > 0 else "0"
        success = str(counts["success"])
        failure = f"**{counts['failure']}**" if counts["failure"] > 0 else "0"
        cancelled = str(counts["cancelled"])
        lines.append(
            f"| `{job_name}` | {running} | {queued} | {waiting} | {stuck} | {success} | {failure} | {cancelled} |"
        )

    lines.append("")

    # Active jobs section
    if active_jobs:
        lines.append("## Active Jobs")
        lines.append("")
        lines.append(
            "| Status | Job Name | Created | Started | Queue | PR/Branch | Runner | Link |"
        )
        lines.append(
            "|--------|----------|---------|---------|-------|-----------|--------|------|"
        )

        for r in sorted(
            active_jobs, key=lambda x: (x["status"], x["created_at"]), reverse=True
        ):
            # Find the processed version for this job
            p = next(
                (
                    p
                    for p in processed_jobs
                    if p["run_id"] == r["run_id"] and p["job_name"] == r["job_name"]
                ),
                None,
            )
            if p:
                lines.append(
                    f"| {p['status']} | `{p['job_name']}` | {p['created_formatted']} | {p['started_formatted']} | {p['queue_time']} | {p['pr_info']} | `{p['runner_name'] or '-'}` | [View]({p['url']}) |"
                )

        lines.append("")

    # Stuck/Ghost jobs section
    if stuck_jobs:
        lines.append("## Stuck/Ghost Jobs")
        lines.append("")
        lines.append(
            "> Jobs showing `in_progress` but have no runner assigned or workflow run is cancelled"
        )
        lines.append("")
        lines.append(
            "| Job Status | Run Status | Job Name | PR/Branch | Runner | Link |"
        )
        lines.append(
            "|------------|------------|----------|-----------|--------|------|"
        )

        for r in sorted(stuck_jobs, key=lambda x: x["created_at"], reverse=True):
            p = next(
                (
                    p
                    for p in processed_jobs
                    if p["run_id"] == r["run_id"] and p["job_name"] == r["job_name"]
                ),
                None,
            )
            if p:
                run_info = f"{r.get('run_status', '-')}/{r.get('run_conclusion', '-')}"
                lines.append(
                    f"| {p['status']} | {run_info} | `{p['job_name']}` | {p['pr_info']} | `{p['runner_name'] or '-'}` | [View]({p['url']}) |"
                )

        lines.append("")

    # Failed jobs section (before All Jobs)
    if failed_jobs:
        lines.append(f"## Failed Jobs ({len(failed_jobs)} total)")
        lines.append("")
        lines.append(
            "| Conclusion | Job Name | Created | Started | Queue | Duration | Runner | PR/Branch | Link |"
        )
        lines.append(
            "|------------|----------|---------|---------|-------|----------|--------|-----------|------|"
        )

        for r in sorted(failed_jobs, key=lambda x: x["created_at"], reverse=True):
            p = next(
                (
                    p
                    for p in processed_jobs
                    if p["run_id"] == r["run_id"] and p["job_name"] == r["job_name"]
                ),
                None,
            )
            if p:
                lines.append(
                    f"| {p['conclusion']} | `{p['job_name']}` | {p['created_formatted']} | {p['started_formatted']} | {p['queue_time']} | {p['duration']} | `{p['runner_name'] or '-'}` | {p['pr_info']} | [View]({p['url']}) |"
                )

        lines.append("")

    # Detailed table (all jobs) - collapsible
    lines.append("<details>")
    lines.append(
        f"<summary><strong>All Jobs ({len(results)} total)</strong> - Click to expand</summary>"
    )
    lines.append("")
    lines.append(
        "| Job Name | Status | Conclusion | Created | Started | Queue | Duration | Runner | PR/Branch | Link |"
    )
    lines.append(
        "|----------|--------|------------|---------|---------|-------|----------|--------|-----------|------|"
    )

    for p in processed_jobs:
        # Mark stuck jobs in markdown with bold
        if p.get("is_stuck", False):
            status_display = f"**STUCK** ({p['status']})"
        else:
            status_display = p["status"]

        lines.append(
            f"| `{p['job_name']}` | {status_display} | {p['conclusion']} | {p['created_formatted']} | {p['started_formatted']} | {p['queue_time']} | {p['duration']} | `{p['runner_name'] or '-'}` | {p['pr_info']} | [View]({p['url']}) |"
        )

    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def format_runner_report_markdown(
    jobs: list[dict],
    workflows: list[str],
    hours: int,
    generated_time: str,
    report_time: datetime = None,
) -> str:
    """Format runner fleet analytics as markdown for GitHub Actions summary."""
    if report_time is None:
        report_time = datetime.now(timezone.utc)

    lines: list[str] = []

    # Header
    lines.append("# CI Runner Fleet Report")
    lines.append("")
    lines.append(f"**Workflows:** {', '.join(f'`{w}`' for w in workflows)}")
    lines.append(f"**Time window:** Last {hours} hours")
    lines.append(f"**Generated:** {generated_time} UTC")
    lines.append(f"**Total jobs analyzed:** {len(jobs)}")
    lines.append("")
    lines.append("> All times are in UTC. Jobs on `ubuntu-latest` are excluded.")
    lines.append("")

    if not jobs:
        lines.append("> No self-hosted runner jobs found in the time window.")
        return "\n".join(lines)

    # --- Fleet Overview ---
    unique_runners = set(
        j["runner_name"] for j in jobs if j.get("runner_name", "-") != "-"
    )
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    lines.append("## Fleet Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total unique runners seen | {len(unique_runners)} |")
    lines.append(f"| Total jobs analyzed | {len(jobs)} |")
    lines.append(f"| Completed jobs | {len(completed_jobs)} |")
    lines.append(f"| Time window | {hours}h |")
    lines.append("")

    # --- Concurrency by Runner Label ---
    concurrency = analyze_concurrency(jobs, report_time)
    if concurrency:
        lines.append("## Concurrency by Runner Label")
        lines.append("")
        lines.append(
            "| Runner Label | Peak Concurrent | Avg Concurrent | Total Jobs | Avg Queue | Avg Duration |"
        )
        lines.append(
            "|-------------|----------------|---------------|-----------|-----------|-------------|"
        )
        for label in sorted(concurrency, key=lambda k: -concurrency[k]["peak"]):
            c = concurrency[label]
            lines.append(
                f"| `{label}` | **{c['peak']}** | {c['avg_concurrent']} "
                f"| {c['total_jobs']} "
                f"| {_format_duration_seconds(c['avg_queue_seconds'])} "
                f"| {_format_duration_seconds(c['avg_duration_seconds'])} |"
            )
        lines.append("")

    # --- Busy Periods ---
    busy_periods = analyze_busy_periods(jobs)
    if busy_periods:
        lines.append("## Busy Periods (UTC)")
        lines.append("")
        lines.append("| Hour (UTC) | Jobs Started | Avg Queue Time | Load |")
        lines.append("|-----------|-------------|---------------|------|")
        for bp in busy_periods:
            if bp["jobs_started"] == 0:
                continue
            load_display = (
                f"**{bp['load']}**" if bp["load"] in ("Peak", "Busy") else bp["load"]
            )
            lines.append(
                f"| {bp['hour_label']} | {bp['jobs_started']} "
                f"| {_format_duration_seconds(bp['avg_queue_seconds'])} "
                f"| {load_display} |"
            )
        lines.append("")

        peak_hours = [bp for bp in busy_periods if bp["load"] == "Peak"]
        quiet_hours = [
            bp
            for bp in busy_periods
            if bp["load"] == "Quiet" and bp["jobs_started"] > 0
        ]
        if peak_hours:
            labels = ", ".join(bp["hour_label"] for bp in peak_hours)
            lines.append(f"> **Peak hours:** {labels}")
            lines.append("")
        if quiet_hours:
            labels = ", ".join(bp["hour_label"] for bp in quiet_hours)
            lines.append(f"> **Quiet hours:** {labels}")
            lines.append("")

    # --- Runner Health ---
    runner_health = analyze_runner_health(jobs)
    if runner_health:
        lines.append("## Runner Health (sorted by failure rate)")
        lines.append("")

        has_failures = any(r["failure"] > 0 for r in runner_health)
        if has_failures:
            lines.append(
                "| Runner Hostname | Labels | Total | Success | Failure | Cancelled | Fail Rate | Avg Duration | Most Failed Job |"
            )
            lines.append(
                "|----------------|--------|-------|---------|---------|-----------|-----------|-------------|----------------|"
            )
        else:
            lines.append(
                "| Runner Hostname | Labels | Total | Success | Failure | Cancelled | Fail Rate | Avg Duration |"
            )
            lines.append(
                "|----------------|--------|-------|---------|---------|-----------|-----------|-------------|"
            )

        for r in runner_health:
            fail_rate_str = (
                f"**{r['fail_rate']}%**" if r["fail_rate"] > 0 else f"{r['fail_rate']}%"
            )
            labels_str = ", ".join(f"`{l}`" for l in r["labels"][:2])
            dur_str = _format_duration_seconds(r["avg_duration_seconds"])
            if has_failures:
                lines.append(
                    f"| `{r['runner_name']}` | {labels_str} | {r['total']} "
                    f"| {r['success']} | {r['failure']} | {r['cancelled']} "
                    f"| {fail_rate_str} | {dur_str} | {r['most_failed_job']} |"
                )
            else:
                lines.append(
                    f"| `{r['runner_name']}` | {labels_str} | {r['total']} "
                    f"| {r['success']} | {r['failure']} | {r['cancelled']} "
                    f"| {fail_rate_str} | {dur_str} |"
                )
        lines.append("")

    # --- Queue Time Distribution ---
    queue_dist = analyze_queue_distribution(jobs)
    if queue_dist["total"] > 0:
        lines.append("## Queue Time Distribution")
        lines.append("")
        lines.append("| Queue Time Range | Count | Percentage |")
        lines.append("|-----------------|-------|------------|")
        for b in queue_dist["buckets"]:
            bar = "#" * int(b["percentage"] / 3)
            lines.append(f"| {b['range']} | {b['count']} | {b['percentage']}% {bar} |")
        lines.append("")
        lines.append(
            f"> **Median:** {_format_duration_seconds(queue_dist['median'])} "
            f"| **P90:** {_format_duration_seconds(queue_dist['p90'])} "
            f"| **P99:** {_format_duration_seconds(queue_dist['p99'])}"
        )
        lines.append("")

    # --- Failed Jobs Detail (collapsible) ---
    failed_jobs = [
        j
        for j in jobs
        if j.get("conclusion") == "failure" and not j.get("is_stuck", False)
    ]
    if failed_jobs:
        lines.append("<details>")
        lines.append(
            f"<summary><strong>Failed Jobs ({len(failed_jobs)} total)</strong> - Click to expand</summary>"
        )
        lines.append("")
        lines.append(
            "| Job Name | Runner | Workflow | Queue | Duration | PR/Branch | Link |"
        )
        lines.append(
            "|----------|--------|---------|-------|----------|-----------|------|"
        )
        for j in sorted(failed_jobs, key=lambda x: x["created_at"], reverse=True):
            queue = calculate_queue_time(
                j["created_at"], j["started_at"], j["status"], report_time
            )
            dur = calculate_duration(j["started_at"], j["completed_at"])
            pr_info = (
                f"PR#{j['pr_number']}" if j.get("pr_number") else j.get("branch", "-")
            )
            url = j.get("html_url", "")
            wf = j.get("workflow", "-")
            lines.append(
                f"| `{j['job_name']}` | `{j['runner_name']}` | `{wf}` "
                f"| {queue} | {dur} | {pr_info} | [View]({url}) |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # --- Stuck Jobs ---
    stuck_jobs = [j for j in jobs if j.get("is_stuck", False)]
    if stuck_jobs:
        lines.append("## Stuck/Ghost Jobs")
        lines.append("")
        lines.append(
            "> Jobs showing `in_progress` but have no runner assigned or workflow run is cancelled"
        )
        lines.append("")
        lines.append(
            "| Job Name | Job Status | Run Status | Runner | Workflow | Link |"
        )
        lines.append("|----------|-----------|-----------|--------|---------|------|")
        for j in sorted(stuck_jobs, key=lambda x: x["created_at"], reverse=True):
            run_info = f"{j.get('run_status', '-')}/{j.get('run_conclusion', '-')}"
            url = j.get("html_url", "")
            wf = j.get("workflow", "-")
            lines.append(
                f"| `{j['job_name']}` | {j['status']} | {run_info} "
                f"| `{j['runner_name']}` | `{wf}` | [View]({url}) |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    # Check gh CLI availability before proceeding
    if not check_gh_cli_available():
        sys.exit(1)

    # Capture the time when the command is run (both datetime and formatted string)
    report_time = datetime.now(timezone.utc)
    report_generated_time = report_time.strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Query GitHub Actions job status")
    parser.add_argument(
        "--repo",
        default="sgl-project/sglang",
        help="GitHub repo (default: sgl-project/sglang)",
    )
    parser.add_argument(
        "--job",
        required=False,
        default=None,
        help="Job name filter (required unless --runner-report is used)",
    )
    parser.add_argument(
        "--workflow",
        default="pr-test-amd.yml",
        help="Workflow file name, or comma-separated list for --runner-report (default: pr-test-amd.yml)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Time window in hours (default: 24)",
    )
    parser.add_argument(
        "--status",
        choices=["in_progress", "queued", "completed", "waiting"],
        help="Filter by job status",
    )
    parser.add_argument(
        "--output",
        choices=["table", "csv", "json", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write markdown output to GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Write output to file",
    )
    parser.add_argument(
        "--runner-report",
        action="store_true",
        help="Generate runner fleet analytics report across all jobs (no --job filter needed)",
    )
    args = parser.parse_args()

    if not args.runner_report and not args.job:
        parser.error("--job is required unless --runner-report is specified")

    # --- Runner fleet report mode ---
    if args.runner_report:
        workflows = [w.strip() for w in args.workflow.split(",") if w.strip()]
        jobs = query_all_jobs(args.repo, workflows, args.hours)

        md_content = format_runner_report_markdown(
            jobs, workflows, args.hours, report_generated_time, report_time
        )

        print(md_content)

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(md_content)
            print(f"\nOutput written to {args.output_file}", file=sys.stderr)

        if args.summary:
            summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_file:
                with open(summary_file, "a") as f:
                    f.write(md_content)
                    f.write("\n")
                print("Summary written to GITHUB_STEP_SUMMARY", file=sys.stderr)
            else:
                print(
                    "Warning: GITHUB_STEP_SUMMARY not set, markdown printed above.",
                    file=sys.stderr,
                )
        return

    # --- Original per-job report mode ---
    results = query_jobs(
        args.repo,
        args.job,
        args.workflow,
        args.hours,
        args.status,
    )

    output_content = None

    if args.output == "table":
        print_table(results, args.repo, report_generated_time, report_time)
    elif args.output == "csv":
        lines = [
            "job_name,status,is_stuck,conclusion,created_at,started_at,queue_time,duration,runner,run_status,run_conclusion,pr_number,branch,url"
        ]
        for r in sorted(results, key=lambda x: x["created_at"], reverse=True):
            queue_time = calculate_queue_time(
                r["created_at"], r["started_at"], r["status"], report_time
            )
            duration = calculate_duration(r["started_at"], r["completed_at"])
            is_stuck = "true" if r.get("is_stuck", False) else "false"
            lines.append(
                f'"{r["job_name"]}",{r["status"]},{is_stuck},{r["conclusion"]},{r["created_at"]},{r["started_at"]},{queue_time},{duration},{r["runner_name"]},{r.get("run_status", "-")},{r.get("run_conclusion", "-")},{r["pr_number"] or ""},{r["branch"]},{r["html_url"]}'
            )
        output_content = "\n".join(lines)
        print(output_content)
    elif args.output == "json":
        json_results = []
        for r in sorted(results, key=lambda x: x["created_at"], reverse=True):
            r_copy = r.copy()
            r_copy["queue_time"] = calculate_queue_time(
                r["created_at"], r["started_at"], r["status"], report_time
            )
            r_copy["duration"] = calculate_duration(r["started_at"], r["completed_at"])
            r_copy["created_at_formatted"] = format_time(r["created_at"])
            r_copy["started_at_formatted"] = format_time(r["started_at"])
            json_results.append(r_copy)
        output_content = json.dumps(json_results, indent=2)
        print(output_content)
    elif args.output == "markdown":
        output_content = format_markdown(
            results, args.repo, args.job, args.hours, report_generated_time, report_time
        )
        print(output_content)

    if args.output_file and output_content:
        with open(args.output_file, "w") as f:
            f.write(output_content)
        print(f"\nOutput written to {args.output_file}", file=sys.stderr)

    if args.summary:
        md_content = format_markdown(
            results, args.repo, args.job, args.hours, report_generated_time, report_time
        )
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(md_content)
                f.write("\n")
            print("Summary written to GITHUB_STEP_SUMMARY", file=sys.stderr)
        else:
            print(
                "Warning: GITHUB_STEP_SUMMARY not set, printing markdown instead:",
                file=sys.stderr,
            )
            print(md_content)


if __name__ == "__main__":
    main()
