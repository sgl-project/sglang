#!/usr/bin/env python3
"""
Query GitHub Actions job status for specific jobs.

Usage:
    python scripts/ci/query_job_status.py --job "stage-c-test-large-8-gpu-amd-mi35x"
    python scripts/ci/query_job_status.py --job "stage-c-test-large-8-gpu-amd-mi35x" --hours 48
    python scripts/ci/query_job_status.py --job "AMD" --workflow pr-test-amd.yml

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
        required=True,
        help="Job name filter (e.g., 'stage-c-test-large-8-gpu-amd-mi35x')",
    )
    parser.add_argument(
        "--workflow",
        default="pr-test-amd.yml",
        help="Workflow file name (default: pr-test-amd.yml)",
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
    args = parser.parse_args()

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
        # Add calculated fields to JSON output for consistency
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

    # Write to file if specified
    if args.output_file and output_content:
        with open(args.output_file, "w") as f:
            f.write(output_content)
        print(f"\nOutput written to {args.output_file}", file=sys.stderr)

    # Write to GITHUB_STEP_SUMMARY if requested
    if args.summary:
        md_content = format_markdown(
            results, args.repo, args.job, args.hours, report_generated_time, report_time
        )
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(md_content)
                f.write("\n")
            print(f"Summary written to GITHUB_STEP_SUMMARY", file=sys.stderr)
        else:
            print(
                "Warning: GITHUB_STEP_SUMMARY not set, printing markdown instead:",
                file=sys.stderr,
            )
            print(md_content)


if __name__ == "__main__":
    main()
