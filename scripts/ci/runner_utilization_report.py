#!/usr/bin/env python3
"""
Runner Utilization Report

Analyzes GitHub Actions job data to calculate runner utilization metrics.
Reports idle time, active time, and utilization percentage per runner label.
"""

import argparse
import os
import subprocess
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict


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
        data = run_gh_command([
            f"repos/{repo}/actions/runs?per_page=100&page={page}",
        ])
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
        data = run_gh_command([
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}",
        ])
        jobs.extend(data.get("jobs", []))
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
        if page > 5:  # Safety limit
            break
    return jobs


def get_runners(repo: str) -> list[dict]:
    """Get all self-hosted runners. Returns empty if no permission."""
    try:
        data = run_gh_command([f"repos/{repo}/actions/runners?per_page=100"])
        return data.get("runners", [])
    except Exception as e:
        print(f"Warning: Cannot access runners API (need admin): {e}")
        return []


def parse_time(time_str: str) -> datetime:
    """Parse ISO timestamp to datetime."""
    if not time_str:
        return None
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def infer_runner_label(runner_name: str) -> str:
    """Infer runner label from runner name."""
    name_lower = runner_name.lower()
    if "5090" in name_lower:
        return "1-gpu-5090"
    elif "h200" in name_lower:
        return "h200"
    elif "h20" in name_lower and "h200" not in name_lower:
        return "h20"
    elif "b200" in name_lower:
        return "b200"
    elif "mi" in name_lower or "amd" in name_lower:
        return "amd"
    elif "ubuntu" in name_lower or "github" in name_lower.replace("-", ""):
        return "github-hosted"
    else:
        return "other"


# Known runner counts per label (update as needed)
KNOWN_RUNNER_COUNTS = {
    "1-gpu-5090": 16,
    "h200": 8,
    "h20": 4,
    "b200": 4,
    "amd": 8,
    "github-hosted": 20,  # GitHub hosted runners (variable)
    "other": 10,
}


def calculate_utilization(repo: str, hours: int = 24, runner_filter: str = None):
    """Calculate runner utilization metrics."""

    print(f"Fetching workflow runs from last {hours} hours...")
    runs = get_workflow_runs(repo, hours)
    print(f"Found {len(runs)} workflow runs")

    # Try to get runners (may fail without admin permission)
    print("Fetching runners...")
    runners = get_runners(repo)

    # Group runners by label from API if available
    label_runners = defaultdict(set)
    if runners:
        for runner in runners:
            for label in runner.get("labels", []):
                label_name = label.get("name", "")
                if label_name not in ["self-hosted", "Linux", "X64"]:
                    label_runners[label_name].add(runner["name"])
        print(f"Got {len(runners)} runners from API")
    else:
        print("No runner API access, will infer labels from job data")

    # Collect job data per runner
    runner_jobs = defaultdict(list)  # runner_name -> list of job_info
    label_jobs = defaultdict(list)   # label -> list of job_info

    total_runs = len(runs)
    for i, run in enumerate(runs):
        if (i + 1) % 50 == 0:
            print(f"Processing run {i+1}/{total_runs}...")

        try:
            jobs = get_jobs_for_run(repo, run["id"])
        except Exception as e:
            continue

        for job in jobs:
            runner_name = job.get("runner_name")
            if not runner_name:
                continue

            started_at = parse_time(job.get("started_at"))
            completed_at = parse_time(job.get("completed_at"))

            if not started_at or not completed_at:
                continue

            duration = (completed_at - started_at).total_seconds()
            job_info = {
                "start": started_at,
                "end": completed_at,
                "duration": duration,
                "job_name": job["name"],
                "runner_name": runner_name,
            }

            runner_jobs[runner_name].append(job_info)

            # Use job labels directly (available in job data)
            job_labels = job.get("labels", [])
            for label in job_labels:
                # Skip generic labels
                if label in ["self-hosted", "Linux", "X64", "ubuntu-latest", "ubuntu-22.04", "ubuntu-24.04"]:
                    continue
                label_runners[label].add(runner_name)
                label_jobs[label].append(job_info)

    # Filter labels if specified
    if runner_filter:
        label_runners = {k: v for k, v in label_runners.items() if runner_filter in k}

    print(f"Tracking {len(label_runners)} runner labels: {list(label_runners.keys())}")

    # Calculate metrics per label
    now = datetime.now(timezone.utc)
    window_seconds = hours * 3600

    results = []

    for label in sorted(label_runners.keys()):
        runners_seen = label_runners[label]
        # Use known count if available, otherwise use observed count
        num_runners = KNOWN_RUNNER_COUNTS.get(label, len(runners_seen))
        total_capacity_seconds = window_seconds * num_runners

        jobs = label_jobs.get(label, [])
        total_active_seconds = sum(j["duration"] for j in jobs)

        utilization = (total_active_seconds / total_capacity_seconds * 100) if total_capacity_seconds > 0 else 0
        idle_seconds = total_capacity_seconds - total_active_seconds

        results.append({
            "label": label,
            "num_runners": num_runners,
            "num_jobs": len(jobs),
            "total_active_hours": total_active_seconds / 3600,
            "total_idle_hours": idle_seconds / 3600,
            "total_capacity_hours": total_capacity_seconds / 3600,
            "utilization_pct": utilization,
        })

    return results


def format_report(results: list[dict], hours: int) -> str:
    """Format results as markdown report."""
    lines = [
        f"# Runner Utilization Report",
        f"",
        f"**Time window:** Last {hours} hours",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"## Summary by Runner Label",
        f"",
        f"| Label | Runners | Jobs | Active (hrs) | Idle (hrs) | Utilization |",
        f"|-------|---------|------|--------------|------------|-------------|",
    ]

    for r in results:
        utilization_bar = "█" * int(r["utilization_pct"] / 10) + "░" * (10 - int(r["utilization_pct"] / 10))
        lines.append(
            f"| {r['label']} | {r['num_runners']} | {r['num_jobs']} | "
            f"{r['total_active_hours']:.1f} | {r['total_idle_hours']:.1f} | "
            f"{r['utilization_pct']:.1f}% {utilization_bar} |"
        )

    # Add interpretation
    lines.extend([
        f"",
        f"## Interpretation",
        f"",
        f"- **High utilization (>80%)**: Consider adding more runners",
        f"- **Low utilization (<20%)**: Runners may be over-provisioned",
        f"- **Idle time**: Time when runners were available but no jobs were queued",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate runner utilization report")
    parser.add_argument("--repo", default="sgl-project/sglang", help="GitHub repo")
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    parser.add_argument("--filter", type=str, help="Filter runner labels (e.g., '5090', 'h200')")
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
