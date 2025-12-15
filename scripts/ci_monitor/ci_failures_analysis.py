"""
SGLang CI Consecutive Failures Analyzer

Monitors GitHub Actions workflows for consecutive test failures and runner issues.
Detects failure streaks, tracks job health, identifies problematic runners, and generates alerts.

Features:
- Analyzes all jobs in PR Test workflow (excluding administrative jobs)
- Tracks consecutive failure streaks for each job
- Monitors runner health and failure rates
- Identifies whether failures are code-related or infrastructure-related
- Generates detailed reports with actionable recommendations

Usage:
    python ci_failures_analysis.py --token <GITHUB_TOKEN> --limit 100
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests


class SGLangFailuresAnalyzer:
    """Analyzes consecutive failures in GitHub Actions workflows."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Failures-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Jobs to EXCLUDE from analysis (administrative/setup jobs, not actual tests)
        self.excluded_jobs = [
            "check-changes",
            "pr-test-finish",  # Nvidia workflow teardown
            "pr-test-amd-finish",  # AMD workflow teardown
            "call-gate",
            "pr-gate",
            "check-all-jobs",
        ]

    def get_recent_runs(
        self,
        limit: int = 500,
        workflow_filter: List[str] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Fetch recent workflow runs from GitHub API using workflow file names.

        Args:
            limit: Number of runs to fetch per workflow
            workflow_filter: List of workflow filenames
            filters: Optional dict of API filters (e.g., {"event": "schedule"}, {"branch": "main"})
        """
        filter_desc = f"workflows: {', '.join(workflow_filter)}"
        if filters:
            filter_desc += f", filters: {filters}"

        print(f"Fetching {limit} runs per workflow ({filter_desc})...")

        all_runs = []

        for workflow_file in workflow_filter:
            print(f"Fetching runs for {workflow_file}...")

            # Use workflow filename directly - much simpler!
            url = f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_file}/runs"
            params = {"per_page": min(limit, 100), "status": "completed"}

            # Apply any additional filters
            if filters:
                params.update(filters)

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                runs = data.get("workflow_runs", [])
                print(f"  Found {len(runs)} runs for {workflow_file}")
                all_runs.extend(runs[:limit])

            except requests.exceptions.RequestException as e:
                print(f"Error fetching runs for {workflow_file}: {e}")
                continue

        print(f"Collected {len(all_runs)} total runs")
        return all_runs

    def get_jobs_for_run(self, run_id: int) -> List[Dict]:
        """Get all jobs for a specific workflow run, handling pagination."""
        try:
            all_jobs = []
            url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            params = {"per_page": 100}  # Max per page

            while url:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                jobs = data.get("jobs", [])
                all_jobs.extend(jobs)

                # Check for next page in Link header
                link_header = response.headers.get("Link", "")
                next_url = None
                if link_header:
                    links = link_header.split(", ")
                    for link in links:
                        if 'rel="next"' in link:
                            next_url = link.split(";")[0].strip("<>")
                            break
                url = next_url
                params = {}  # Clear params for subsequent requests (URL has them)

            return all_jobs
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs for run {run_id}: {e}")
            return []

    def analyze_runner_health(
        self, runs: List[Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """
        Analyze runner health by tracking failures per runner and consecutive failure streaks.

        Returns:
            Tuple of (runner_stats, runner_instance_data, runner_streak_data, runner_instance_streak_data)
            - runner_stats: Overall stats per runner (failure rate, total jobs, etc.)
            - runner_instance_data: Per-instance breakdown of failures
            - runner_streak_data: Consecutive failure streaks per runner label
            - runner_instance_streak_data: Consecutive failure streaks per runner instance
        """
        print("\nAnalyzing runner health and consecutive failures...")

        # Sort runs by created_at (oldest first)
        sorted_runs = sorted(runs, key=lambda x: x.get("created_at", ""))

        # Track runner statistics (overall)
        runner_total_jobs: Dict[str, int] = defaultdict(int)
        runner_failed_jobs: Dict[str, int] = defaultdict(int)
        runner_job_failures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        runner_job_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Track queue times per runner instance (can aggregate for runner labels if needed)
        runner_instance_queue_times: Dict[str, List[float]] = defaultdict(list)

        # Track individual runner instances (runner_name + runner_id)
        runner_instance_stats: Dict[str, Dict] = defaultdict(
            lambda: {"total_jobs": 0, "failed_jobs": 0, "jobs_failed": defaultdict(int)}
        )

        # Track consecutive failures per runner (by labels)
        runner_current_streak: Dict[str, int] = defaultdict(int)
        runner_max_streak: Dict[str, int] = defaultdict(int)
        runner_first_failure_in_streak: Dict[str, Optional[Dict]] = {}
        runner_last_failure_in_streak: Dict[str, Optional[Dict]] = {}
        runner_recovery_info: Dict[str, Optional[Dict]] = {}

        # Track consecutive failures per runner instance
        runner_instance_current_streak: Dict[str, int] = defaultdict(int)
        runner_instance_max_streak: Dict[str, int] = defaultdict(int)
        runner_instance_first_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_last_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_recovery: Dict[str, Optional[Dict]] = {}

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed} for runner analysis: #{run.get('run_number')}"
                )

            head_commit = run.get("head_commit") or {}
            run_info = {
                "run_number": run.get("run_number"),
                "run_id": run.get("id"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": head_commit.get("author", {}).get("name", "Unknown"),
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            # Get jobs for this run
            jobs = self.get_jobs_for_run(run.get("id"))

            # Track whether each runner had at least one failure in this run
            runner_had_failure: Dict[str, bool] = defaultdict(bool)
            runner_had_success: Dict[str, bool] = defaultdict(bool)
            runner_instance_had_failure: Dict[str, bool] = defaultdict(bool)
            runner_instance_had_success: Dict[str, bool] = defaultdict(bool)
            # Track first failed job for each runner in this run (for linking)
            runner_first_failed_job: Dict[str, Dict] = {}
            runner_instance_first_failed_job: Dict[str, Dict] = {}

            for job in jobs:
                job_name = job.get("name", "")

                # Skip excluded jobs (administrative/setup jobs)
                if any(
                    job_name.startswith(excluded) for excluded in self.excluded_jobs
                ):
                    continue

                # Extract runner information
                # GitHub API might use different fields for runner info
                runner_name = (
                    job.get("runner_name")
                    or job.get("runner", {}).get("name")
                    or "unknown"
                )
                runner_id = job.get("runner_id") or job.get("runner", {}).get("id")

                # Get runner labels (from runs-on field in workflow)
                runner_labels = job.get("labels", [])
                runner_labels_str = (
                    ", ".join(runner_labels) if runner_labels else "unknown"
                )

                # Skip jobs without runner information (likely skipped/queued jobs)
                if not runner_labels_str or runner_labels_str == "unknown":
                    continue

                # Track by runner labels (primary identifier)
                # Use labels as the key since they're more informative than runner_name
                runner_key = runner_labels_str
                runner_total_jobs[runner_key] += 1
                runner_job_totals[runner_key][job_name] += 1

                # Track by specific runner instance
                if runner_id:
                    runner_instance_key = f"{runner_labels_str}_{runner_id}"
                    runner_instance_stats[runner_instance_key]["total_jobs"] += 1
                    # Store runner name for reference
                    runner_instance_stats[runner_instance_key][
                        "runner_name"
                    ] = runner_name

                    # Calculate queue time (time from created to started) per instance
                    created_at = job.get("created_at")
                    started_at = job.get("started_at")
                    if created_at and started_at:
                        try:
                            from datetime import datetime

                            created_time = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                            started_time = datetime.fromisoformat(
                                started_at.replace("Z", "+00:00")
                            )
                            queue_time_seconds = (
                                started_time - created_time
                            ).total_seconds()
                            if queue_time_seconds >= 0:  # Sanity check
                                runner_instance_queue_times[runner_instance_key].append(
                                    queue_time_seconds
                                )
                        except (ValueError, AttributeError):
                            pass  # Skip if timestamp parsing fails

                conclusion = job.get("conclusion")

                if conclusion == "failure":
                    # Failure detected
                    runner_failed_jobs[runner_key] += 1
                    runner_job_failures[runner_key][job_name] += 1
                    runner_had_failure[runner_key] = True

                    # Track first failed job for this runner in this run (for linking)
                    if runner_key not in runner_first_failed_job:
                        runner_first_failed_job[runner_key] = {
                            "job_id": job.get("id"),
                            "job_url": job.get("html_url", run_info["url"]),
                            "job_name": job_name,
                        }

                    if runner_id:
                        runner_instance_stats[runner_instance_key]["failed_jobs"] += 1
                        runner_instance_stats[runner_instance_key]["jobs_failed"][
                            job_name
                        ] += 1
                        runner_instance_had_failure[runner_instance_key] = True

                        # Track first failed job for this runner instance in this run
                        if runner_instance_key not in runner_instance_first_failed_job:
                            runner_instance_first_failed_job[runner_instance_key] = {
                                "job_id": job.get("id"),
                                "job_url": job.get("html_url", run_info["url"]),
                                "job_name": job_name,
                            }

                elif conclusion == "success":
                    runner_had_success[runner_key] = True
                    if runner_id:
                        runner_instance_had_success[runner_instance_key] = True

            # Update consecutive failure streaks based on run-level results
            # A runner is considered "failing" if it had at least one failure in the run
            for runner_key in set(
                list(runner_had_failure.keys()) + list(runner_had_success.keys())
            ):
                if runner_had_failure[runner_key]:
                    runner_current_streak[runner_key] += 1
                    failure_info = {
                        **run_info,
                        "runner_key": runner_key,
                    }

                    # Include job URL if we have it
                    if runner_key in runner_first_failed_job:
                        failure_info.update(runner_first_failed_job[runner_key])

                    # Track if this is the first failure in a new streak
                    if runner_current_streak[runner_key] == 1:
                        runner_first_failure_in_streak[runner_key] = failure_info
                    # Always update last failure to the most recent one
                    runner_last_failure_in_streak[runner_key] = failure_info

                    # Update max streak
                    if (
                        runner_current_streak[runner_key]
                        > runner_max_streak[runner_key]
                    ):
                        runner_max_streak[runner_key] = runner_current_streak[
                            runner_key
                        ]

                elif runner_had_success[runner_key]:
                    # Success - streak broken
                    if runner_current_streak[runner_key] > 0:
                        runner_recovery_info[runner_key] = {
                            **run_info,
                            "runner_key": runner_key,
                            "streak_length": runner_current_streak[runner_key],
                        }

                    runner_current_streak[runner_key] = 0
                    runner_first_failure_in_streak[runner_key] = None
                    runner_last_failure_in_streak[runner_key] = None

            # Update instance streaks
            for runner_instance_key in set(
                list(runner_instance_had_failure.keys())
                + list(runner_instance_had_success.keys())
            ):
                if runner_instance_had_failure[runner_instance_key]:
                    runner_instance_current_streak[runner_instance_key] += 1

                    if runner_instance_current_streak[runner_instance_key] == 1:
                        failure_info = {
                            **run_info,
                            "runner_instance": runner_instance_key,
                        }
                        # Include job URL if we have it
                        if runner_instance_key in runner_instance_first_failed_job:
                            failure_info.update(
                                runner_instance_first_failed_job[runner_instance_key]
                            )
                        runner_instance_first_failure[runner_instance_key] = (
                            failure_info
                        )

                    # Always update last failure to the most recent one
                    failure_info = {
                        **run_info,
                        "runner_instance": runner_instance_key,
                    }
                    # Include job URL if we have it
                    if runner_instance_key in runner_instance_first_failed_job:
                        failure_info.update(
                            runner_instance_first_failed_job[runner_instance_key]
                        )
                    runner_instance_last_failure[runner_instance_key] = failure_info

                    if (
                        runner_instance_current_streak[runner_instance_key]
                        > runner_instance_max_streak[runner_instance_key]
                    ):
                        runner_instance_max_streak[runner_instance_key] = (
                            runner_instance_current_streak[runner_instance_key]
                        )

                elif runner_instance_had_success[runner_instance_key]:
                    if runner_instance_current_streak[runner_instance_key] > 0:
                        runner_instance_recovery[runner_instance_key] = {
                            **run_info,
                            "runner_instance": runner_instance_key,
                            "streak_length": runner_instance_current_streak[
                                runner_instance_key
                            ],
                        }

                    runner_instance_current_streak[runner_instance_key] = 0
                    runner_instance_first_failure[runner_instance_key] = None
                    runner_instance_last_failure[runner_instance_key] = None

            time.sleep(0.05)

        # Build final runner stats
        runner_stats = {}
        for runner_key in runner_total_jobs.keys():
            total = runner_total_jobs[runner_key]
            failed = runner_failed_jobs[runner_key]
            failure_rate = (failed / total * 100) if total > 0 else 0

            # Calculate queue time statistics by aggregating from runner instances
            # Find all instances that match this runner label
            aggregated_queue_times = []
            for instance_key, queue_times in runner_instance_queue_times.items():
                # Extract the labels part from "labels_id"
                instance_labels = (
                    instance_key.rsplit("_", 1)[0]
                    if "_" in instance_key
                    else instance_key
                )
                if instance_labels == runner_key:
                    aggregated_queue_times.extend(queue_times)

            avg_queue_time = (
                sum(aggregated_queue_times) / len(aggregated_queue_times)
                if aggregated_queue_times
                else 0
            )
            p90_queue_time = 0
            if aggregated_queue_times:
                sorted_queue_times = sorted(aggregated_queue_times)
                p90_index = int(len(sorted_queue_times) * 0.9)
                p90_queue_time = (
                    sorted_queue_times[p90_index]
                    if p90_index < len(sorted_queue_times)
                    else sorted_queue_times[-1]
                )

            runner_stats[runner_key] = {
                "total_jobs": total,
                "failed_jobs": failed,
                "failure_rate": failure_rate,
                "unique_jobs_with_failures": len(runner_job_failures[runner_key]),
                "jobs_failed": dict(runner_job_failures[runner_key]),
                "jobs_total": dict(runner_job_totals[runner_key]),
                "avg_queue_time_seconds": avg_queue_time,
                "p90_queue_time_seconds": p90_queue_time,
                "queue_time_samples": len(aggregated_queue_times),
            }

        # Convert runner instance stats to regular dicts with queue time stats
        runner_instance_data = {}
        for instance_key, stats in runner_instance_stats.items():
            # Calculate queue time statistics for this instance
            queue_times = runner_instance_queue_times[instance_key]
            avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
            p90_queue_time = 0
            if queue_times:
                sorted_queue_times = sorted(queue_times)
                p90_index = int(len(sorted_queue_times) * 0.9)
                p90_queue_time = (
                    sorted_queue_times[p90_index]
                    if p90_index < len(sorted_queue_times)
                    else sorted_queue_times[-1]
                )

            runner_instance_data[instance_key] = {
                "total_jobs": stats["total_jobs"],
                "failed_jobs": stats["failed_jobs"],
                "failure_rate": (
                    stats["failed_jobs"] / stats["total_jobs"] * 100
                    if stats["total_jobs"] > 0
                    else 0
                ),
                "jobs_failed": dict(stats["jobs_failed"]),
                "runner_name": stats.get("runner_name", "unknown"),
                "avg_queue_time_seconds": avg_queue_time,
                "p90_queue_time_seconds": p90_queue_time,
                "queue_time_samples": len(queue_times),
            }

        # Build runner streak data
        runner_streak_data = {}
        for runner_key in runner_total_jobs.keys():
            runner_streak_data[runner_key] = {
                "current_streak": runner_current_streak[runner_key],
                "max_streak": runner_max_streak[runner_key],
                "total_failures": runner_failed_jobs[runner_key],
                "total_jobs": runner_total_jobs[runner_key],
                "failure_rate": (
                    runner_failed_jobs[runner_key] / runner_total_jobs[runner_key] * 100
                    if runner_total_jobs[runner_key] > 0
                    else 0
                ),
                "jobs_failed": dict(runner_job_failures[runner_key]),
                "first_failure_in_streak": runner_first_failure_in_streak.get(
                    runner_key
                ),
                "last_failure_in_streak": runner_last_failure_in_streak.get(runner_key),
                "recovery_info": runner_recovery_info.get(runner_key),
            }

        # Build runner instance streak data
        runner_instance_streak_data = {}
        for instance_key in runner_instance_stats.keys():
            runner_instance_streak_data[instance_key] = {
                "current_streak": runner_instance_current_streak[instance_key],
                "max_streak": runner_instance_max_streak[instance_key],
                "total_failures": runner_instance_stats[instance_key]["failed_jobs"],
                "total_jobs": runner_instance_stats[instance_key]["total_jobs"],
                "failure_rate": (
                    runner_instance_stats[instance_key]["failed_jobs"]
                    / runner_instance_stats[instance_key]["total_jobs"]
                    * 100
                    if runner_instance_stats[instance_key]["total_jobs"] > 0
                    else 0
                ),
                "runner_name": runner_instance_stats[instance_key].get(
                    "runner_name", "unknown"
                ),
                "jobs_failed": dict(runner_instance_stats[instance_key]["jobs_failed"]),
                "first_failure_in_streak": runner_instance_first_failure.get(
                    instance_key
                ),
                "last_failure_in_streak": runner_instance_last_failure.get(
                    instance_key
                ),
                "recovery_info": runner_instance_recovery.get(instance_key),
            }

        return (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        )

    def analyze_consecutive_failures(
        self, runs: List[Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Analyze consecutive failures for each job.

        "Current Streak" = consecutive failures ending at the most recent run (NOW)
        If the most recent run succeeded, current streak = 0 (streak is broken)
        "Max Streak" = the longest consecutive failure streak seen in the analyzed period

        Returns:
            Tuple of (job_streak_data, job_current_streaks)
        """
        print("\nAnalyzing consecutive failures...")

        # Sort runs by created_at (oldest first) to track streaks chronologically
        sorted_runs = sorted(runs, key=lambda x: x.get("created_at", ""))

        # Track current streak for each job
        job_current_streak: Dict[str, int] = defaultdict(int)
        job_max_streak: Dict[str, int] = defaultdict(int)
        job_total_failures: Dict[str, int] = defaultdict(int)
        job_total_runs: Dict[str, int] = defaultdict(int)
        job_first_failure_in_streak: Dict[str, Optional[Dict]] = {}
        job_last_failure_in_streak: Dict[str, Optional[Dict]] = {}
        job_recovery_info: Dict[str, Optional[Dict]] = {}
        job_recent_runs: Dict[str, List[Dict]] = defaultdict(list)  # Track last 5 runs

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed}: #{run.get('run_number')}"
                )

            head_commit = run.get("head_commit") or {}
            run_info = {
                "run_number": run.get("run_number"),
                "run_id": run.get("id"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": head_commit.get("author", {}).get("name", "Unknown"),
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            # Get jobs for this run
            jobs = self.get_jobs_for_run(run.get("id"))

            for job in jobs:
                job_name = job.get("name", "")

                # Skip excluded jobs (administrative/setup jobs)
                if any(
                    job_name.startswith(excluded) for excluded in self.excluded_jobs
                ):
                    continue

                job_total_runs[job_name] += 1
                conclusion = job.get("conclusion")

                if conclusion == "failure":
                    # Failure detected
                    job_total_failures[job_name] += 1
                    job_current_streak[job_name] += 1

                    # Track if this is the first failure in a new streak
                    if job_current_streak[job_name] == 1:
                        job_first_failure_in_streak[job_name] = {
                            **run_info,
                            "job_name": job_name,
                            "job_id": job.get("id"),
                            "job_url": job.get("html_url", run_info["url"]),
                            "conclusion": conclusion,
                        }

                    # Always update last failure to the most recent one
                    job_last_failure_in_streak[job_name] = {
                        **run_info,
                        "job_name": job_name,
                        "job_id": job.get("id"),
                        "job_url": job.get("html_url", run_info["url"]),
                        "conclusion": conclusion,
                    }

                    # Update max streak
                    if job_current_streak[job_name] > job_max_streak[job_name]:
                        job_max_streak[job_name] = job_current_streak[job_name]

                elif conclusion == "success":
                    # Success - streak broken
                    if job_current_streak[job_name] > 0:
                        # Record recovery
                        job_recovery_info[job_name] = {
                            **run_info,
                            "job_name": job_name,
                            "streak_length": job_current_streak[job_name],
                        }

                    job_current_streak[job_name] = 0
                    job_first_failure_in_streak[job_name] = None
                    job_last_failure_in_streak[job_name] = None

                # Track recent runs (last 5 for each job)
                run_attempt = job.get("run_attempt", 1)

                # Create status emoji with superscript if retry attempt > 1
                if conclusion == "success":
                    status = "✅"
                elif conclusion == "failure":
                    status = "❌"
                else:
                    status = "⚪"

                # Add superscript for retry attempts (2+ only)
                if run_attempt > 1:
                    superscript_map = {
                        "2": "²",
                        "3": "³",
                        "4": "⁴",
                        "5": "⁵",
                        "6": "⁶",
                        "7": "⁷",
                        "8": "⁸",
                        "9": "⁹",
                    }
                    status += superscript_map.get(str(run_attempt), f"^{run_attempt}")

                job_recent_runs[job_name].append(
                    {
                        "run_number": run_info["run_number"],
                        "job_url": job.get("html_url", run_info["url"]),
                        "conclusion": conclusion,
                        "status": status,
                        "run_attempt": run_attempt,
                    }
                )

            time.sleep(0.05)

        # Build final results
        job_streak_data = {}
        for job_name in job_current_streak.keys():
            # Get last 5 runs (most recent first)
            recent_runs = job_recent_runs.get(job_name, [])[-5:][
                ::-1
            ]  # Last 5, reversed

            job_streak_data[job_name] = {
                "current_streak": job_current_streak[job_name],
                "max_streak": job_max_streak[job_name],
                "total_failures": job_total_failures[job_name],
                "total_runs": job_total_runs[job_name],
                "failure_rate": (
                    job_total_failures[job_name] / job_total_runs[job_name] * 100
                    if job_total_runs[job_name] > 0
                    else 0
                ),
                "first_failure_in_streak": job_first_failure_in_streak.get(job_name),
                "last_failure_in_streak": job_last_failure_in_streak.get(job_name),
                "recovery_info": job_recovery_info.get(job_name),
                "recent_runs": recent_runs,  # Last 5 runs with status emoji
            }

        return job_streak_data, job_current_streak

    # print statements here mainly for local testing
    def generate_failure_report(
        self,
        # Scheduled runs (9 workflows)
        pr_test_nvidia_scheduled_data: Dict[str, Dict],
        pr_test_amd_scheduled_data: Dict[str, Dict],
        pr_test_xeon_scheduled_data: Dict[str, Dict],
        pr_test_xpu_scheduled_data: Dict[str, Dict],
        pr_test_npu_scheduled_data: Dict[str, Dict],
        nightly_nvidia_scheduled_data: Dict[str, Dict],
        nightly_amd_scheduled_data: Dict[str, Dict],
        nightly_intel_scheduled_data: Dict[str, Dict],
        nightly_npu_scheduled_data: Dict[str, Dict],
        # General runs (9 workflows)
        pr_test_nvidia_general_data: Dict[str, Dict],
        pr_test_amd_general_data: Dict[str, Dict],
        pr_test_xeon_general_data: Dict[str, Dict],
        pr_test_xpu_general_data: Dict[str, Dict],
        pr_test_npu_general_data: Dict[str, Dict],
        nightly_nvidia_general_data: Dict[str, Dict],
        nightly_amd_general_data: Dict[str, Dict],
        nightly_intel_general_data: Dict[str, Dict],
        nightly_npu_general_data: Dict[str, Dict],
        # Runners
        runner_stats: Optional[Dict[str, Dict]] = None,
        runner_instance_data: Optional[Dict[str, Dict]] = None,
        runner_streak_data: Optional[Dict[str, Dict]] = None,
        runner_instance_streak_data: Optional[Dict[str, Dict]] = None,
        # Config
        output_file: Optional[str] = None,
        pr_test_scheduled_limit: int = 12,
        nightly_scheduled_limit: int = 6,
        general_limit: int = 100,
    ):
        """Generate detailed failure analysis report."""
        print("\n" + "=" * 80)
        print("SGLang Consecutive Failures Analysis Report")
        print("=" * 80)

        # Combine all general data for summary stats
        combined_general_data = {
            **pr_test_nvidia_general_data,
            **pr_test_amd_general_data,
            **pr_test_xeon_general_data,
            **pr_test_xpu_general_data,
            **pr_test_npu_general_data,
            **nightly_nvidia_general_data,
            **nightly_amd_general_data,
            **nightly_intel_general_data,
            **nightly_npu_general_data,
        }

        # Sort jobs by current streak (descending)
        sorted_jobs = sorted(
            combined_general_data.items(),
            key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
            reverse=True,
        )

        # Summary Statistics
        print("\n## Summary Statistics")
        print(f"Total (unique) jobs analyzed: {len(sorted_jobs)}")
        print(
            f"Jobs with Active Failure Streaks: {sum(1 for j in sorted_jobs if j[1]['current_streak'] > 0)}"
        )

        if runner_stats:
            print(f"Total Runners Analyzed: {len(runner_stats)}")

        # Queue Time Summary
        if runner_stats:
            all_avg_queue_times = []
            all_p90_queue_times = []
            for stats in runner_stats.values():
                if stats["queue_time_samples"] > 0:
                    all_avg_queue_times.append(stats["avg_queue_time_seconds"])
                    all_p90_queue_times.append(stats["p90_queue_time_seconds"])

            if all_avg_queue_times:
                overall_avg = sum(all_avg_queue_times) / len(all_avg_queue_times)
                overall_p90 = sum(all_p90_queue_times) / len(all_p90_queue_times)
                print("\n## Queue Time Summary")
                print(
                    f"Average Queue Time (across all runners): {overall_avg / 60:.1f} minutes ({overall_avg:.0f}s)"
                )
                print(
                    f"P90 Queue Time (across all runners): {overall_p90 / 60:.1f} minutes ({overall_p90:.0f}s)"
                )

        # Helper function to print job section
        def print_job_section(
            title: str, data: Dict[str, Dict], color_failures: bool = False
        ):
            sorted_data = sorted(
                data.items(),
                key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                reverse=True,
            )
            broken = [(name, d) for name, d in sorted_data if d["current_streak"] >= 2]
            recently_failed = [
                (name, d)
                for name, d in sorted_data
                if d["current_streak"] < 2 and d["total_failures"] > 0
            ]

            # Always show section header
            print("\n" + "=" * 130)
            if broken:
                print(f"## {title} ({len(broken)} jobs with active streaks)")
                print("=" * 130)
                print(
                    f"\n{'Job Name':<40} {'Current':<8} {'Max':<6} {'Runs':<6} {'First':<13} {'Last':<13} {'Recent History':<30}"
                )
                print("-" * 130)
                for job_name, d in broken[:15]:
                    display_name = (
                        job_name if len(job_name) <= 38 else job_name[:35] + "..."
                    )

                    first_failure = d.get("first_failure_in_streak")
                    first_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )

                    last_failure = d.get("last_failure_in_streak")
                    last_str = (
                        f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                    )

                    # Recent history (last 5 runs as emoji)
                    recent_runs = d.get("recent_runs", [])
                    history_str = (
                        " ".join([r["status"] for r in recent_runs])
                        if recent_runs
                        else "N/A"
                    )

                    # Color red if color_failures is True (for critical sections)
                    if color_failures:
                        print(
                            f"\033[91m{display_name:<40}\033[0m {d['current_streak']:<8} {d['max_streak']:<6} {d['total_runs']:<6} {first_str:<13} {last_str:<13} {history_str:<30}"
                        )
                    else:
                        print(
                            f"{display_name:<40} {d['current_streak']:<8} {d['max_streak']:<6} {d['total_runs']:<6} {first_str:<13} {last_str:<13} {history_str:<30}"
                        )
            else:
                print(f"## {title}")
                print("=" * 130)
                print("\n✅ No jobs with active failure streaks (streak >= 2)")

            # Show recently failed jobs in a collapsed section (terminal doesn't support collapse, so just show as separate section)
            if recently_failed:
                print(
                    f"\n   Recently failed jobs (no active streak): {len(recently_failed)} jobs"
                )
                print(
                    f"   {'Job Name':<38} {'Failures':<12} {'Fail Rate':<12} {'Total Runs':<12} {'Recent History (last 5)':<30}"
                )
                print("   " + "-" * 120)
                for job_name, d in recently_failed[:10]:
                    display_name = (
                        job_name if len(job_name) <= 36 else job_name[:33] + "..."
                    )
                    recent_runs = d.get("recent_runs", [])
                    history_str = (
                        " ".join([r["status"] for r in recent_runs])
                        if recent_runs
                        else "N/A"
                    )
                    print(
                        f"   {display_name:<38} {d['total_failures']:<12} {d['failure_rate']:.1f}%{'':<7} {d['total_runs']:<12} {history_str:<30}"
                    )

        # ========== SCHEDULED/MAIN BRANCH RUNS (9 sections) ==========
        print("\n" + "█" * 130)
        print("SCHEDULED RUNS (Main Branch)")
        print("█" * 130)

        # PR Tests - Scheduled (5 workflows)
        print_job_section(
            f"1. PR Test NVIDIA - Scheduled (latest {pr_test_scheduled_limit} runs)",
            pr_test_nvidia_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"2. PR Test AMD - Scheduled (latest {pr_test_scheduled_limit} runs)",
            pr_test_amd_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"3. PR Test Xeon - Scheduled (latest {pr_test_scheduled_limit} runs)",
            pr_test_xeon_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"4. PR Test XPU - Scheduled (latest {pr_test_scheduled_limit} runs)",
            pr_test_xpu_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"5. PR Test NPU - Scheduled (latest {pr_test_scheduled_limit} runs)",
            pr_test_npu_scheduled_data,
            color_failures=True,
        )

        # Nightly Tests - Scheduled (4 workflows)
        print_job_section(
            f"6. Nightly NVIDIA - Scheduled (latest {nightly_scheduled_limit} runs)",
            nightly_nvidia_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"7. Nightly AMD - Scheduled (latest {nightly_scheduled_limit} runs)",
            nightly_amd_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"8. Nightly Intel - Scheduled (latest {nightly_scheduled_limit} runs)",
            nightly_intel_scheduled_data,
            color_failures=True,
        )
        print_job_section(
            f"9. Nightly NPU - Scheduled (latest {nightly_scheduled_limit} runs)",
            nightly_npu_scheduled_data,
            color_failures=True,
        )

        # ========== GENERAL RUNS (9 sections) ==========
        print("\n" + "█" * 130)
        print("GENERAL RUNS (All Branches)")
        print("█" * 130)

        # PR Tests - General (5 workflows)
        print_job_section(
            f"10. PR Test NVIDIA - General (latest {general_limit} runs)",
            pr_test_nvidia_general_data,
            color_failures=False,
        )
        print_job_section(
            f"11. PR Test AMD - General (latest {general_limit} runs)",
            pr_test_amd_general_data,
            color_failures=False,
        )
        print_job_section(
            f"12. PR Test Xeon - General (latest {general_limit} runs)",
            pr_test_xeon_general_data,
            color_failures=False,
        )
        print_job_section(
            f"13. PR Test XPU - General (latest {general_limit} runs)",
            pr_test_xpu_general_data,
            color_failures=False,
        )
        print_job_section(
            f"14. PR Test NPU - General (latest {general_limit} runs)",
            pr_test_npu_general_data,
            color_failures=False,
        )

        # Nightly Tests - General (4 workflows)
        print_job_section(
            f"15. Nightly NVIDIA - General (latest {general_limit} runs)",
            nightly_nvidia_general_data,
            color_failures=False,
        )
        print_job_section(
            f"16. Nightly AMD - General (latest {general_limit} runs)",
            nightly_amd_general_data,
            color_failures=False,
        )
        print_job_section(
            f"17. Nightly Intel - General (latest {general_limit} runs)",
            nightly_intel_general_data,
            color_failures=False,
        )
        print_job_section(
            f"18. Nightly NPU - General (latest {general_limit} runs)",
            nightly_npu_general_data,
            color_failures=False,
        )

        # ========== RUNNERS ==========
        print("\n" + "█" * 130)
        print("RUNNER HEALTH")
        print("█" * 130)

        # 5. Workers (at the very bottom) - Use machine names from runner instances (streak >= 2)
        if runner_instance_data and runner_instance_streak_data:
            # Combine instance stats with streak data and sort by consecutive failures first
            combined_data = []
            for instance_key, stats in runner_instance_data.items():
                streak_data = runner_instance_streak_data.get(instance_key, {})
                combined_data.append(
                    {
                        "runner_name": stats.get("runner_name", "unknown"),
                        "instance_key": instance_key,
                        "current_streak": streak_data.get("current_streak", 0),
                        "max_streak": streak_data.get("max_streak", 0),
                        "failure_rate": stats["failure_rate"],
                        "total_jobs": stats["total_jobs"],
                        "unique_jobs": len(stats.get("jobs_failed", {})),
                        "avg_queue": stats.get("avg_queue_time_seconds", 0),
                        "p90_queue": stats.get("p90_queue_time_seconds", 0),
                        "queue_samples": stats.get("queue_time_samples", 0),
                        "first_failure": streak_data.get("first_failure_in_streak"),
                        "last_failure": streak_data.get("last_failure_in_streak"),
                    }
                )

            # Sort by current streak (descending), then max streak, then failure rate
            sorted_runners = sorted(
                combined_data,
                key=lambda x: (x["current_streak"], x["max_streak"], x["failure_rate"]),
                reverse=True,
            )

            # Only show runners with streak >= 2
            runners_with_issues = [
                r for r in sorted_runners if r["current_streak"] >= 2
            ]

            # Always show section header
            print("\n" + "=" * 140)
            print("## 5. Top 15 Workers by Consecutive Failures")
            print("=" * 140)

            if runners_with_issues:
                print(
                    f"\n{'Machine Name':<30} {'Curr':<5} {'Max':<5} {'Fail%':<7} {'AvgQ':<7} {'Total':<7} {'Unique':<8} {'First':<13} {'Last':<13}"
                )
                print("-" * 140)

                for runner_data in runners_with_issues[:15]:
                    # Truncate machine name if too long for display
                    display_name = (
                        runner_data["runner_name"]
                        if len(runner_data["runner_name"]) <= 28
                        else runner_data["runner_name"][:25] + "..."
                    )

                    # Format streaks
                    streak_str = str(runner_data["current_streak"])
                    max_str = str(runner_data["max_streak"])

                    # Format queue time
                    avg_queue_str = (
                        f"{runner_data['avg_queue'] / 60:.1f}m"
                        if runner_data["queue_samples"] > 0
                        else "N/A"
                    )

                    # Get first and last failure info
                    first_failure = runner_data.get("first_failure")
                    first_failure_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )

                    last_failure = runner_data.get("last_failure")
                    last_failure_str = (
                        f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                    )

                    # Color red for workers with failures
                    print(
                        f"\033[91m{display_name:<30}\033[0m {streak_str:<5} {max_str:<5} {runner_data['failure_rate']:>5.1f}% {avg_queue_str:<7} {runner_data['total_jobs']:<7} {runner_data['unique_jobs']:<8} {first_failure_str:<13} {last_failure_str:<13}"
                    )
            else:
                print("\n✅ No runners with active failure streaks (streak >= 2)")

        # Build report data (always needed for GitHub summary)
        # Calculate overall queue time for summary
        overall_avg_queue = 0
        overall_p90_queue = 0
        if runner_stats:
            all_avg_queue_times = [
                stats["avg_queue_time_seconds"]
                for stats in runner_stats.values()
                if stats["queue_time_samples"] > 0
            ]
            all_p90_queue_times = [
                stats["p90_queue_time_seconds"]
                for stats in runner_stats.values()
                if stats["queue_time_samples"] > 0
            ]
            if all_avg_queue_times:
                overall_avg_queue = sum(all_avg_queue_times) / len(all_avg_queue_times)
                overall_p90_queue = sum(all_p90_queue_times) / len(all_p90_queue_times)

        report_data = {
            "summary": {
                "total_jobs": len(sorted_jobs),
                "jobs_with_streaks": sum(
                    1 for j in sorted_jobs if j[1]["current_streak"] > 0
                ),
                "total_runners": len(runner_stats) if runner_stats else 0,
                "analysis_timestamp": datetime.now().isoformat(),
                "avg_queue_time_seconds": overall_avg_queue,
                "p90_queue_time_seconds": overall_p90_queue,
            },
            "pr_test_scheduled_limit": pr_test_scheduled_limit,
            "nightly_scheduled_limit": nightly_scheduled_limit,
            "general_limit": general_limit,
            # Scheduled data
            "pr_test_nvidia_scheduled_data": pr_test_nvidia_scheduled_data,
            "pr_test_amd_scheduled_data": pr_test_amd_scheduled_data,
            "pr_test_xeon_scheduled_data": pr_test_xeon_scheduled_data,
            "pr_test_xpu_scheduled_data": pr_test_xpu_scheduled_data,
            "pr_test_npu_scheduled_data": pr_test_npu_scheduled_data,
            "nightly_nvidia_scheduled_data": nightly_nvidia_scheduled_data,
            "nightly_amd_scheduled_data": nightly_amd_scheduled_data,
            "nightly_intel_scheduled_data": nightly_intel_scheduled_data,
            "nightly_npu_scheduled_data": nightly_npu_scheduled_data,
            # General data
            "pr_test_nvidia_general_data": pr_test_nvidia_general_data,
            "pr_test_amd_general_data": pr_test_amd_general_data,
            "pr_test_xeon_general_data": pr_test_xeon_general_data,
            "pr_test_xpu_general_data": pr_test_xpu_general_data,
            "pr_test_npu_general_data": pr_test_npu_general_data,
            "nightly_nvidia_general_data": nightly_nvidia_general_data,
            "nightly_amd_general_data": nightly_amd_general_data,
            "nightly_intel_general_data": nightly_intel_general_data,
            "nightly_npu_general_data": nightly_npu_general_data,
            "runner_stats": runner_stats if runner_stats else {},
            "runner_instance_data": (
                runner_instance_data if runner_instance_data else {}
            ),
            "runner_streak_data": runner_streak_data if runner_streak_data else {},
            "runner_instance_streak_data": (
                runner_instance_streak_data if runner_instance_streak_data else {}
            ),
        }

        # Save to JSON only if output file is specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed report saved to: {output_file}")

        print("=" * 80)

        return report_data

    def generate_github_summary(self, report_data: Dict):
        """Generate GitHub Actions Step Summary."""
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("Not running in GitHub Actions, skipping summary generation")
                return

            print("Generating GitHub Actions summary...")

            summary_lines = []
            summary_lines.append("# SGLang Consecutive Failures Analysis")
            summary_lines.append("")
            summary_lines.append(
                f"**Analysis Timestamp:** {report_data['summary']['analysis_timestamp']}"
            )
            summary_lines.append("_Note: Recent runs are shown left to right_")
            summary_lines.append("")

            # Summary stats - COLLAPSIBLE
            summary_lines.append("<details>")
            summary_lines.append(
                "<summary>📊 Summary Statistics (click to expand)</summary>"
            )
            summary_lines.append("")
            summary_lines.append("| Metric | Count |")
            summary_lines.append("|--------|-------|")
            summary_lines.append(
                f"| Total (unique) jobs analyzed | {report_data['summary']['total_jobs']} |"
            )
            summary_lines.append(
                f"| Jobs with Active Failure Streaks | {report_data['summary']['jobs_with_streaks']} |"
            )

            # Add main branch job counters
            pr_main_count = report_data["summary"].get("pr_main_count", 0)
            pr_main_with_streaks = report_data["summary"].get("pr_main_with_streaks", 0)
            nightly_main_count = report_data["summary"].get("nightly_main_count", 0)
            nightly_main_with_streaks = report_data["summary"].get(
                "nightly_main_with_streaks", 0
            )

            summary_lines.append(
                f"| PR Test Jobs on Main (scheduled) | {pr_main_count} ({pr_main_with_streaks} with streaks) |"
            )
            summary_lines.append(
                f"| Nightly Test Jobs on Main (scheduled) | {nightly_main_count} ({nightly_main_with_streaks} with streaks) |"
            )

            summary_lines.append(
                f"| Total Runners Analyzed | {report_data['summary']['total_runners']} |"
            )
            summary_lines.append("")
            summary_lines.append("</details>")
            summary_lines.append("")

            # Queue Time Summary - COLLAPSIBLE
            if report_data.get("summary", {}).get("avg_queue_time_seconds") is not None:
                avg_queue = report_data["summary"]["avg_queue_time_seconds"]
                p90_queue = report_data["summary"]["p90_queue_time_seconds"]
                summary_lines.append("<details>")
                summary_lines.append(
                    "<summary>📊 Queue Time Summary (click to expand)</summary>"
                )
                summary_lines.append("")
                summary_lines.append("| Metric | Value |")
                summary_lines.append("|--------|-------|")
                summary_lines.append(
                    f"| Average Queue Time (across all runners) | {avg_queue / 60:.1f} minutes ({avg_queue:.0f}s) |"
                )
                summary_lines.append(
                    f"| P90 Queue Time (across all runners) | {p90_queue / 60:.1f} minutes ({p90_queue:.0f}s) |"
                )
                summary_lines.append("")
                summary_lines.append("</details>")
                summary_lines.append("")

            # Helper function to generate job section for GitHub markdown
            def generate_job_section_md(title: str, data: Dict[str, Dict]):
                sorted_data = sorted(
                    data.items(),
                    key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                    reverse=True,
                )
                broken = [
                    (name, d) for name, d in sorted_data if d["current_streak"] >= 2
                ]
                recently_failed = [
                    (name, d)
                    for name, d in sorted_data
                    if d["current_streak"] < 2 and d["total_failures"] > 0
                ]

                # Always show section header
                summary_lines.append(f"## {title}")
                summary_lines.append("")

                if broken:
                    summary_lines.append(
                        "| Job Name | Current | Max | Runs | First | Last | Recent History |"
                    )
                    summary_lines.append(
                        "|----------|---------|-----|------|-------|------|----------------|"
                    )
                    for job_name, d in broken[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )

                        first_failure = d.get("first_failure_in_streak")
                        first_str = (
                            f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                            if first_failure
                            else "N/A"
                        )

                        last_failure = d.get("last_failure_in_streak")
                        last_str = (
                            f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                            if last_failure
                            else "N/A"
                        )

                        # Recent history (last 5 runs as clickable emoji)
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        # Make entire row red if current streak >= 3
                        if d["current_streak"] >= 3:
                            summary_lines.append(
                                f"| <span style='color:red'>`{display_name}`</span> | <span style='color:red'>{d['current_streak']}</span> | <span style='color:red'>{d['max_streak']}</span> | <span style='color:red'>{d['total_runs']}</span> | "
                                f"<span style='color:red'>{first_str}</span> | <span style='color:red'>{last_str}</span> | <span style='color:red'>{history_links}</span> |"
                            )
                        else:
                            summary_lines.append(
                                f"| `{display_name}` | {d['current_streak']} | {d['max_streak']} | {d['total_runs']} | "
                                f"{first_str} | {last_str} | {history_links} |"
                            )
                    summary_lines.append("")
                else:
                    summary_lines.append(
                        "✅ **No jobs with active failure streaks (streak >= 2)**"
                    )
                    summary_lines.append("")

                # Show recently failed jobs in a collapsible section
                if recently_failed:
                    summary_lines.append("<details>")
                    summary_lines.append(
                        f"<summary>Recently failed jobs (no active streak) - {len(recently_failed)} jobs</summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Failures | Fail Rate | Total Runs | Recent History (last 5) |"
                    )
                    summary_lines.append(
                        "|----------|----------|-----------|------------|-------------------------|"
                    )
                    for job_name, d in recently_failed[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        summary_lines.append(
                            f"| `{display_name}` | {d['total_failures']} | {d['failure_rate']:.1f}% | {d['total_runs']} | {history_links} |"
                        )
                    summary_lines.append("")
                    summary_lines.append("</details>")
                    summary_lines.append("")

            # ========== SCHEDULED RUNS (9 sections) ==========
            summary_lines.append("---")
            summary_lines.append("# 📅 SCHEDULED RUNS (Main Branch)")
            summary_lines.append("")

            # Get limits
            pr_sched_limit = report_data.get("pr_test_scheduled_limit", 12)
            nightly_sched_limit = report_data.get("nightly_scheduled_limit", 6)

            # PR Tests - Scheduled (5 workflows)
            generate_job_section_md(
                f"1. PR Test NVIDIA - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_nvidia_scheduled_data", {}),
            )
            generate_job_section_md(
                f"2. PR Test AMD - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_amd_scheduled_data", {}),
            )
            generate_job_section_md(
                f"3. PR Test Xeon - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_xeon_scheduled_data", {}),
            )
            generate_job_section_md(
                f"4. PR Test XPU - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_xpu_scheduled_data", {}),
            )
            generate_job_section_md(
                f"5. PR Test NPU - Scheduled (latest {pr_sched_limit} runs)",
                report_data.get("pr_test_npu_scheduled_data", {}),
            )

            # Nightly Tests - Scheduled (4 workflows)
            generate_job_section_md(
                f"6. Nightly NVIDIA - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_nvidia_scheduled_data", {}),
            )
            generate_job_section_md(
                f"7. Nightly AMD - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_amd_scheduled_data", {}),
            )
            generate_job_section_md(
                f"8. Nightly Intel - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_intel_scheduled_data", {}),
            )
            generate_job_section_md(
                f"9. Nightly NPU - Scheduled (latest {nightly_sched_limit} runs)",
                report_data.get("nightly_npu_scheduled_data", {}),
            )

            # ========== GENERAL RUNS (9 sections) ==========
            summary_lines.append("---")
            summary_lines.append("# 🌍 GENERAL RUNS (All Branches)")
            summary_lines.append("")

            gen_limit = report_data.get("general_limit", 100)

            # PR Tests - General (5 workflows)
            generate_job_section_md(
                f"10. PR Test NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("pr_test_nvidia_general_data", {}),
            )
            generate_job_section_md(
                f"11. PR Test AMD - General (latest {gen_limit} runs)",
                report_data.get("pr_test_amd_general_data", {}),
            )
            generate_job_section_md(
                f"12. PR Test Xeon - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xeon_general_data", {}),
            )
            generate_job_section_md(
                f"13. PR Test XPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xpu_general_data", {}),
            )
            generate_job_section_md(
                f"14. PR Test NPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_npu_general_data", {}),
            )

            # Nightly Tests - General (4 workflows)
            generate_job_section_md(
                f"15. Nightly NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("nightly_nvidia_general_data", {}),
            )
            generate_job_section_md(
                f"16. Nightly AMD - General (latest {gen_limit} runs)",
                report_data.get("nightly_amd_general_data", {}),
            )
            generate_job_section_md(
                f"17. Nightly Intel - General (latest {gen_limit} runs)",
                report_data.get("nightly_intel_general_data", {}),
            )
            generate_job_section_md(
                f"18. Nightly NPU - General (latest {gen_limit} runs)",
                report_data.get("nightly_npu_general_data", {}),
            )

            # ========== RUNNERS ==========
            summary_lines.append("---")
            summary_lines.append("# 🖥️ RUNNER HEALTH")
            summary_lines.append("")

            # 5. Workers section
            if report_data.get("runner_instance_data") and report_data.get(
                "runner_instance_streak_data"
            ):
                # Combine instance stats with streak data
                combined_data = []
                for instance_key, stats in report_data["runner_instance_data"].items():
                    streak_data = report_data["runner_instance_streak_data"].get(
                        instance_key, {}
                    )
                    combined_data.append(
                        {
                            "runner_name": stats.get("runner_name", "unknown"),
                            "current_streak": streak_data.get("current_streak", 0),
                            "max_streak": streak_data.get("max_streak", 0),
                            "failure_rate": stats["failure_rate"],
                            "total_jobs": stats["total_jobs"],
                            "unique_jobs": len(stats.get("jobs_failed", {})),
                            "avg_queue": stats.get("avg_queue_time_seconds", 0),
                            "first_failure": streak_data.get("first_failure_in_streak"),
                            "last_failure": streak_data.get("last_failure_in_streak"),
                        }
                    )

                sorted_runners = sorted(
                    combined_data,
                    key=lambda x: (
                        x["current_streak"],
                        x["max_streak"],
                        x["failure_rate"],
                    ),
                    reverse=True,
                )

                runners_with_issues = [
                    r for r in sorted_runners if r["current_streak"] >= 2
                ]

                # Always show section header
                summary_lines.append("## 5. Workers")
                summary_lines.append("")

                if runners_with_issues:
                    summary_lines.append(
                        "| Machine Name | Current Streak | Max | Fail Rate | Avg Queue | Total Jobs | Unique Jobs | First Failure | Last Failure |"
                    )
                    summary_lines.append(
                        "|--------------|----------------|-----|-----------|-----------|------------|-------------|---------------|--------------|"
                    )

                    for runner_data in runners_with_issues[:15]:
                        display_name = (
                            runner_data["runner_name"]
                            if len(runner_data["runner_name"]) <= 28
                            else runner_data["runner_name"][:25] + "..."
                        )

                        avg_queue_str = (
                            f"{runner_data['avg_queue'] / 60:.1f}m"
                            if runner_data["avg_queue"] > 0
                            else "N/A"
                        )

                        first_failure = runner_data.get("first_failure")
                        first_str = (
                            f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                            if first_failure
                            else "N/A"
                        )

                        last_failure = runner_data.get("last_failure")
                        last_str = (
                            f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                            if last_failure
                            else "N/A"
                        )

                        # Make entire row red if current streak >= 3
                        if runner_data["current_streak"] >= 3:
                            summary_lines.append(
                                f"| <span style='color:red'>`{display_name}`</span> | <span style='color:red'>{runner_data['current_streak']}</span> | <span style='color:red'>{runner_data['max_streak']}</span> | "
                                f"<span style='color:red'>{runner_data['failure_rate']:.1f}%</span> | <span style='color:red'>{avg_queue_str}</span> | <span style='color:red'>{runner_data['total_jobs']}</span> | <span style='color:red'>{runner_data.get('unique_jobs', 0)}</span> | <span style='color:red'>{first_str}</span> | <span style='color:red'>{last_str}</span> |"
                            )
                        else:
                            summary_lines.append(
                                f"| `{display_name}` | {runner_data['current_streak']} | {runner_data['max_streak']} | "
                                f"{runner_data['failure_rate']:.1f}% | {avg_queue_str} | {runner_data['total_jobs']} | {runner_data.get('unique_jobs', 0)} | {first_str} | {last_str} |"
                            )

                    summary_lines.append("")
                else:
                    summary_lines.append(
                        "✅ **No runners with active failure streaks (streak >= 2)**"
                    )
                    summary_lines.append("")

            # Write summary
            with open(github_step_summary, "a", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="SGLang Consecutive Failures Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of workflow runs to analyze per workflow for general analysis (default: 100)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (optional, only writes if specified)",
    )

    args = parser.parse_args()

    analyzer = SGLangFailuresAnalyzer(args.token)

    try:
        # Fetch runs for each category separately
        print("\n" + "=" * 80)
        print("FETCHING WORKFLOW RUNS")
        print("=" * 80)

        # Fixed limits for scheduled runs
        pr_test_scheduled_limit = 12  # Past 12 scheduled PR Test runs
        nightly_scheduled_limit = 6  # Past 6 scheduled Nightly Test runs

        # === SCHEDULED RUNS (9 workflows) ===
        # PR Tests - Scheduled (5 workflows)
        pr_test_nvidia_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test.yml"],
            filters={"event": "schedule"},
        )
        # These 4 don't have scheduled events, so filter by main branch instead
        pr_test_amd_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-amd.yml"],
            filters={"branch": "main"},
        )
        pr_test_xeon_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-xeon.yml"],
            filters={"branch": "main"},
        )
        pr_test_xpu_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-xpu.yml"],
            filters={"branch": "main"},
        )
        pr_test_npu_scheduled_runs = analyzer.get_recent_runs(
            limit=pr_test_scheduled_limit,
            workflow_filter=["pr-test-npu.yml"],
            filters={"branch": "main"},
        )

        # Nightly Tests - Scheduled (4 workflows)
        nightly_nvidia_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-nvidia.yml"],
            filters={"event": "schedule"},
        )
        nightly_amd_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-amd.yml"],
            filters={"event": "schedule"},
        )
        nightly_intel_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-intel.yml"],
            filters={"event": "schedule"},
        )
        nightly_npu_scheduled_runs = analyzer.get_recent_runs(
            limit=nightly_scheduled_limit,
            workflow_filter=["nightly-test-npu.yml"],
            filters={"event": "schedule"},
        )

        # === GENERAL RUNS (9 workflows) ===
        # PR Tests - General (5 workflows)
        pr_test_nvidia_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test.yml"],
        )
        pr_test_amd_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-amd.yml"],
        )
        pr_test_xeon_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-xeon.yml"],
        )
        pr_test_xpu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-xpu.yml"],
        )
        pr_test_npu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["pr-test-npu.yml"],
        )

        # Nightly Tests - General (4 workflows)
        nightly_nvidia_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-nvidia.yml"],
        )
        nightly_amd_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-amd.yml"],
        )
        nightly_intel_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-intel.yml"],
        )
        nightly_npu_general_runs = analyzer.get_recent_runs(
            limit=args.limit,
            workflow_filter=["nightly-test-npu.yml"],
        )

        # Choosing nvidia pr test and nightly for runner health analysis
        runner_runs = pr_test_nvidia_general_runs + nightly_nvidia_general_runs

        if not runner_runs:
            print("No workflow runs found")
            return

        print("\n" + "=" * 80)
        print("ANALYZING CONSECUTIVE FAILURES")
        print("=" * 80)

        # Analyze SCHEDULED runs
        pr_test_nvidia_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_nvidia_scheduled_runs)
            if pr_test_nvidia_scheduled_runs
            else ({}, {})
        )
        pr_test_amd_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_amd_scheduled_runs)
            if pr_test_amd_scheduled_runs
            else ({}, {})
        )
        pr_test_xeon_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xeon_scheduled_runs)
            if pr_test_xeon_scheduled_runs
            else ({}, {})
        )
        pr_test_xpu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xpu_scheduled_runs)
            if pr_test_xpu_scheduled_runs
            else ({}, {})
        )
        pr_test_npu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_npu_scheduled_runs)
            if pr_test_npu_scheduled_runs
            else ({}, {})
        )

        nightly_nvidia_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_nvidia_scheduled_runs)
            if nightly_nvidia_scheduled_runs
            else ({}, {})
        )
        nightly_amd_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_amd_scheduled_runs)
            if nightly_amd_scheduled_runs
            else ({}, {})
        )
        nightly_intel_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_intel_scheduled_runs)
            if nightly_intel_scheduled_runs
            else ({}, {})
        )
        nightly_npu_scheduled_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_npu_scheduled_runs)
            if nightly_npu_scheduled_runs
            else ({}, {})
        )

        # Analyze GENERAL runs
        pr_test_nvidia_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_nvidia_general_runs)
            if pr_test_nvidia_general_runs
            else ({}, {})
        )
        pr_test_amd_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_amd_general_runs)
            if pr_test_amd_general_runs
            else ({}, {})
        )
        pr_test_xeon_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xeon_general_runs)
            if pr_test_xeon_general_runs
            else ({}, {})
        )
        pr_test_xpu_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_xpu_general_runs)
            if pr_test_xpu_general_runs
            else ({}, {})
        )
        pr_test_npu_general_data, _ = (
            analyzer.analyze_consecutive_failures(pr_test_npu_general_runs)
            if pr_test_npu_general_runs
            else ({}, {})
        )

        nightly_nvidia_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_nvidia_general_runs)
            if nightly_nvidia_general_runs
            else ({}, {})
        )
        nightly_amd_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_amd_general_runs)
            if nightly_amd_general_runs
            else ({}, {})
        )
        nightly_intel_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_intel_general_runs)
            if nightly_intel_general_runs
            else ({}, {})
        )
        nightly_npu_general_data, _ = (
            analyzer.analyze_consecutive_failures(nightly_npu_general_runs)
            if nightly_npu_general_runs
            else ({}, {})
        )

        # Analyze runner health and consecutive failures on all runs
        (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        ) = analyzer.analyze_runner_health(runner_runs)

        # Generate report with all datasets
        report_data = analyzer.generate_failure_report(
            # Scheduled runs (9 workflows)
            pr_test_nvidia_scheduled_data,
            pr_test_amd_scheduled_data,
            pr_test_xeon_scheduled_data,
            pr_test_xpu_scheduled_data,
            pr_test_npu_scheduled_data,
            nightly_nvidia_scheduled_data,
            nightly_amd_scheduled_data,
            nightly_intel_scheduled_data,
            nightly_npu_scheduled_data,
            # General runs (9 workflows)
            pr_test_nvidia_general_data,
            pr_test_amd_general_data,
            pr_test_xeon_general_data,
            pr_test_xpu_general_data,
            pr_test_npu_general_data,
            nightly_nvidia_general_data,
            nightly_amd_general_data,
            nightly_intel_general_data,
            nightly_npu_general_data,
            # Runners
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
            # Config
            args.output,
            pr_test_scheduled_limit,
            nightly_scheduled_limit,
            args.limit,
        )

        # Generate GitHub Actions summary
        analyzer.generate_github_summary(report_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
