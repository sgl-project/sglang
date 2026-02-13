#!/usr/bin/env python3
"""
Runner Utilization Report

Analyzes GitHub Actions job data to calculate runner utilization metrics.
Reports idle time, active time, and utilization percentage per runner label.
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

# Labels to skip when grouping runners (GitHub default labels)
DEFAULT_LABELS_TO_IGNORE = {"self-hosted", "Linux", "X64", "ARM64"}
GITHUB_HOSTED_LABELS = {"ubuntu-latest", "ubuntu-22.04", "ubuntu-24.04"}


def run_gh_command(args: list[str]) -> dict:
    """Run gh CLI command and return JSON result."""
    result = subprocess.run(
        ["gh", "api"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(f"gh api failed: {result.stderr}")
    return json.loads(result.stdout)


def get_workflow_runs(repo: str, hours: int = 24) -> list[dict]:
    """Get workflow runs from the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    runs = []
    page = 1
    while True:
        data = run_gh_command(
            [
                f"repos/{repo}/actions/runs?per_page=100&page={page}",
            ]
        )
        page_runs = data.get("workflow_runs", [])

        # Filter by time
        for run in page_runs:
            created_at = parse_time(run.get("created_at"))
            if created_at and created_at >= since:
                runs.append(run)
            elif created_at and created_at < since:
                # Runs are ordered by created_at desc, so we can stop
                return runs

        if len(page_runs) < 100:
            break
        page += 1
        if page > 20:  # Safety limit
            break
    return runs


def get_jobs_for_run(repo: str, run_id: int) -> list[dict]:
    """Get all jobs for a workflow run."""
    jobs = []
    page = 1
    while True:
        data = run_gh_command(
            [
                f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}",
            ]
        )
        jobs.extend(data.get("jobs", []))
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
        if page > 5:  # Safety limit
            break
    return jobs


def get_runners(repo: str, online_only: bool = True) -> list[dict]:
    """Get all self-hosted runners with pagination. Returns empty if no permission."""
    try:
        all_runners = []
        page = 1
        while True:
            data = run_gh_command(
                [f"repos/{repo}/actions/runners?per_page=100&page={page}"]
            )
            runners = data.get("runners", [])
            all_runners.extend(runners)
            if len(runners) < 100:
                break
            page += 1
            if page > 10:  # Safety limit
                break
        if online_only:
            all_runners = [r for r in all_runners if r.get("status") == "online"]
        return all_runners
    except Exception as e:
        print(f"Warning: Cannot access runners API (need admin): {e}")
        return []


def parse_time(time_str: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    if not time_str:
        return None
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


# Known runner counts per label (fallback when API unavailable)
KNOWN_RUNNER_COUNTS = {
    "1-gpu-5090": 16,
    "h200": 8,
    "h20": 4,
    "b200": 4,
    "amd": 8,
    "github-hosted": 20,  # GitHub hosted runners (variable)
    "other": 10,
}


def calculate_concurrency_metrics(
    jobs: list[dict],
    window_start: datetime,
    window_end: datetime,
    num_runners: int,
) -> dict:
    """
    Calculate concurrency metrics using a sweep line algorithm.

    Tracks:
    - Peak concurrent runners in use
    - Average concurrent runners over time
    - Time at saturation (all runners busy)
    - Queue depth when runners are saturated
    """
    if not jobs:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_seconds": 0,
            "saturation_pct": 0.0,
            "peak_queue": 0,
        }

    window_seconds = (window_end - window_start).total_seconds()
    if window_seconds <= 0:
        return {
            "peak_concurrent": 0,
            "avg_concurrent": 0.0,
            "saturation_seconds": 0,
            "saturation_pct": 0.0,
            "peak_queue": 0,
        }

    # Create events for running jobs: +1 at start, -1 at end
    running_events = []
    for job in jobs:
        start = job["start"]
        end = job["end"]
        # Clamp to window
        if end < window_start or start > window_end:
            continue
        clamped_start = max(start, window_start)
        clamped_end = min(end, window_end)
        running_events.append((clamped_start, 1, "start"))  # +1 for start
        running_events.append((clamped_end, -1, "end"))  # -1 for end

    # Create events for queue tracking (jobs created but not started)
    queue_events = []
    for job in jobs:
        created_at = job.get("created_at")
        started_at = job["start"]
        if created_at and created_at < started_at:
            # Clamp to window
            if started_at < window_start or created_at > window_end:
                continue
            clamped_created = max(created_at, window_start)
            clamped_started = min(started_at, window_end)
            queue_events.append((clamped_created, 1, "queued"))
            queue_events.append((clamped_started, -1, "dequeued"))

    # Sort running events: by time, then ends before starts at same time
    running_events.sort(key=lambda e: (e[0], e[1] == 1))

    # Process running events to get concurrency metrics
    current_running = 0
    peak_running = 0
    prev_time = window_start
    total_running_seconds = 0.0
    saturation_seconds = 0.0

    for event_time, delta, _ in running_events:
        # Accumulate time at previous concurrency level
        time_delta = (event_time - prev_time).total_seconds()
        if time_delta > 0:
            total_running_seconds += current_running * time_delta
            if current_running >= num_runners:
                saturation_seconds += time_delta

        # Update concurrency
        current_running += delta
        peak_running = max(peak_running, current_running)
        prev_time = event_time

    # Handle remaining time after last event
    if prev_time < window_end:
        time_delta = (window_end - prev_time).total_seconds()
        total_running_seconds += current_running * time_delta
        if current_running >= num_runners:
            saturation_seconds += time_delta

    # Sort queue events and calculate peak queue depth
    queue_events.sort(key=lambda e: (e[0], e[1] == 1))
    current_queued = 0
    peak_queue = 0

    for _, delta, _ in queue_events:
        current_queued += delta
        peak_queue = max(peak_queue, current_queued)

    avg_concurrent = total_running_seconds / window_seconds if window_seconds > 0 else 0

    return {
        "peak_concurrent": peak_running,
        "avg_concurrent": avg_concurrent,
        "saturation_seconds": saturation_seconds,
        "saturation_pct": (
            (saturation_seconds / window_seconds * 100) if window_seconds > 0 else 0
        ),
        "peak_queue": peak_queue,
    }


def calculate_utilization(repo: str, hours: int = 24, runner_filter: str = None):
    """Calculate runner utilization metrics."""

    print(f"Fetching workflow runs from last {hours} hours...")
    runs = get_workflow_runs(repo, hours)
    print(f"Found {len(runs)} workflow runs")

    # Try to get online runners from API
    print("Fetching online runners...")
    runners = get_runners(repo, online_only=True)

    # Build label -> set of online runner names from API
    api_label_runners = defaultdict(set)
    if runners:
        for runner in runners:
            for label in runner.get("labels", []):
                label_name = label.get("name", "")
                if label_name not in DEFAULT_LABELS_TO_IGNORE:
                    api_label_runners[label_name].add(runner["name"])
        print(f"Got {len(runners)} online runners from API")
    else:
        print("No runner API access, will use observed runners from job data")

    # Track runners seen in jobs (for labels not in API or when API unavailable)
    job_label_runners = defaultdict(set)
    label_jobs = defaultdict(list)  # label -> list of job_info

    # Fetch jobs for all runs in parallel
    total_runs = len(runs)
    print(f"Fetching jobs for {total_runs} runs in parallel...")

    def fetch_jobs_for_run(run):
        """Fetch jobs for a single run, returning (run_id, jobs) or (run_id, None) on error."""
        try:
            return (run["id"], get_jobs_for_run(repo, run["id"]))
        except Exception:
            return (run["id"], None)

    all_jobs = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_jobs_for_run, run) for run in runs]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"Fetched jobs for {completed}/{total_runs} runs...")
            run_id, jobs = future.result()
            if jobs:
                all_jobs.extend(jobs)

    print(f"Processing {len(all_jobs)} jobs...")

    for job in all_jobs:
        runner_name = job.get("runner_name")
        if not runner_name:
            continue

        created_at = parse_time(job.get("created_at"))
        started_at = parse_time(job.get("started_at"))
        completed_at = parse_time(job.get("completed_at"))

        if not started_at or not completed_at:
            continue

        duration = (completed_at - started_at).total_seconds()
        queue_time = (started_at - created_at).total_seconds() if created_at else 0
        job_info = {
            "start": started_at,
            "end": completed_at,
            "created_at": created_at,
            "duration": duration,
            "queue_time": queue_time,
            "job_name": job["name"],
            "runner_name": runner_name,
        }

        # Use job labels directly (available in job data)
        job_labels = job.get("labels", [])
        for label in job_labels:
            # Skip generic labels
            if label in DEFAULT_LABELS_TO_IGNORE | GITHUB_HOSTED_LABELS:
                continue
            job_label_runners[label].add(runner_name)
            label_jobs[label].append(job_info)

    # Merge API runners and job-observed runners
    # Prefer API count (online runners) when available
    all_labels = set(api_label_runners.keys()) | set(job_label_runners.keys())

    # Filter labels if specified
    if runner_filter:
        all_labels = {lbl for lbl in all_labels if runner_filter in lbl}

    print(f"Tracking {len(all_labels)} runner labels: {sorted(all_labels)}")

    # Calculate metrics per label
    window_seconds = hours * 3600
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)

    results = []

    for label in sorted(all_labels):
        # Use API runner count if available, otherwise use job-observed count
        if label in api_label_runners and api_label_runners[label]:
            num_runners = len(api_label_runners[label])
        elif label in job_label_runners:
            num_runners = len(job_label_runners[label])
        else:
            num_runners = KNOWN_RUNNER_COUNTS.get(label, 1)

        total_capacity_seconds = window_seconds * num_runners

        jobs = label_jobs.get(label, [])
        total_active_seconds = sum(j["duration"] for j in jobs)

        utilization = (
            (total_active_seconds / total_capacity_seconds * 100)
            if total_capacity_seconds > 0
            else 0
        )
        idle_seconds = total_capacity_seconds - total_active_seconds

        # Calculate queue time metrics
        queue_times = [j["queue_time"] for j in jobs if j["queue_time"] > 0]
        avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
        max_queue_time = max(queue_times) if queue_times else 0

        # Calculate concurrency metrics
        # First pass: get peak concurrent to determine effective capacity
        concurrency_initial = calculate_concurrency_metrics(
            jobs, window_start, window_end, num_runners
        )

        # Use observed peak as effective capacity if lower than API count
        # This handles cases where not all runners are active all the time
        effective_runners = min(num_runners, concurrency_initial["peak_concurrent"])
        if effective_runners < num_runners and effective_runners > 0:
            # Recalculate with effective capacity for accurate saturation
            concurrency = calculate_concurrency_metrics(
                jobs, window_start, window_end, effective_runners
            )
        else:
            concurrency = concurrency_initial
            effective_runners = num_runners

        results.append(
            {
                "label": label,
                "num_runners": num_runners,
                "effective_runners": effective_runners,
                "num_jobs": len(jobs),
                "total_active_hours": total_active_seconds / 3600,
                "total_idle_hours": idle_seconds / 3600,
                "total_capacity_hours": total_capacity_seconds / 3600,
                "utilization_pct": utilization,
                "avg_queue_min": avg_queue_time / 60,
                "max_queue_min": max_queue_time / 60,
                # Concurrency metrics
                "peak_concurrent": concurrency_initial["peak_concurrent"],
                "avg_concurrent": concurrency["avg_concurrent"],
                "saturation_hours": concurrency["saturation_seconds"] / 3600,
                "saturation_pct": concurrency["saturation_pct"],
                "peak_queue": concurrency["peak_queue"],
            }
        )

    return results


def format_report(results: list[dict], hours: int) -> str:
    """Format results as markdown report."""
    lines = [
        "# Runner Utilization Report",
        "",
        f"**Time window:** Last {hours} hours",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Concurrency Analysis",
        "",
        "| Label | Runners (API/Effective) | Peak Concurrent | Avg Concurrent | Saturation Time | Peak Queue |",
        "|-------|-------------------------|-----------------|----------------|-----------------|------------|",
    ]

    for r in results:
        effective = r["effective_runners"]
        avg_pct = (r["avg_concurrent"] / effective * 100) if effective > 0 else 0
        runner_str = (
            f"{r['num_runners']}/{effective}"
            if effective != r["num_runners"]
            else str(r["num_runners"])
        )
        lines.append(
            f"| {r['label']} | {runner_str} | "
            f"{r['peak_concurrent']} | "
            f"{r['avg_concurrent']:.1f} ({avg_pct:.0f}%) | "
            f"{r['saturation_hours']:.1f}h ({r['saturation_pct']:.0f}%) | "
            f"{r['peak_queue']} jobs |"
        )

    # Add recommendations section
    lines.extend(["", "## Recommendations", ""])
    has_recommendations = False
    for r in results:
        label = r["label"]
        saturation_pct = r["saturation_pct"]
        peak_queue = r["peak_queue"]
        effective = r["effective_runners"]
        avg_pct = (r["avg_concurrent"] / effective * 100) if effective > 0 else 0

        if saturation_pct > 50 or peak_queue > 5:
            lines.append(
                f"⚠️ **{label}**: High saturation ({saturation_pct:.0f}%) "
                f"with queue buildup ({peak_queue} jobs). Consider adding runners."
            )
            has_recommendations = True
        elif saturation_pct > 20 or peak_queue > 0:
            lines.append(
                f"📊 **{label}**: Moderate saturation ({saturation_pct:.0f}%), "
                f"peak queue {peak_queue} jobs. Monitor for trends."
            )
            has_recommendations = True
        elif avg_pct < 30 and r["num_jobs"] > 0:
            lines.append(
                f"💡 **{label}**: Low average utilization ({avg_pct:.0f}%). "
                f"Runner pool may be oversized."
            )
            has_recommendations = True
        else:
            lines.append(f"✓ **{label}**: Healthy utilization with minimal queueing.")

    if not has_recommendations and results:
        lines.append("All runner pools have healthy utilization.")

    # Add summary table
    lines.extend(
        [
            "",
            "## Summary by Runner Label",
            "",
            "| Label | Runners | Jobs | Active (hrs) | Utilization | Avg Queue | Max Queue |",
            "|-------|---------|------|--------------|-------------|-----------|-----------|",
        ]
    )

    for r in results:
        utilization_bar = "█" * int(r["utilization_pct"] / 10) + "░" * (
            10 - int(r["utilization_pct"] / 10)
        )
        lines.append(
            f"| {r['label']} | {r['num_runners']} | {r['num_jobs']} | "
            f"{r['total_active_hours']:.1f} | "
            f"{r['utilization_pct']:.1f}% {utilization_bar} | "
            f"{r['avg_queue_min']:.1f}m | {r['max_queue_min']:.1f}m |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate runner utilization report")
    parser.add_argument("--repo", default="sgl-project/sglang", help="GitHub repo")
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    parser.add_argument(
        "--filter", type=str, help="Filter runner labels (e.g., '5090', 'h200')"
    )
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    results = calculate_utilization(args.repo, args.hours, args.filter)
    report = format_report(results, args.hours)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Also write to GITHUB_STEP_SUMMARY if available
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report)


if __name__ == "__main__":
    main()
