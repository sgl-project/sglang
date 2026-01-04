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

    def get_job_logs(self, job_id: int) -> str:
        """Fetch logs for a specific job."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{job_id}/logs"
            response = self.session.get(url, timeout=60, allow_redirects=True)
            if response.status_code == 200:
                return response.text
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error fetching logs for job {job_id}: {e}")
            return ""

    def get_online_runners(self) -> Dict[str, Dict]:
        """
        Fetch all self-hosted runners and their online status from GitHub API.

        Returns:
            Dict mapping runner label sets to their online/total counts.
            E.g., {"8-gpu-h200-runner": {"online": 2, "total": 3, "busy": 1}}
        """
        print("Fetching self-hosted runner status...")
        try:
            # Use separate admin token if available (needs repo admin scope)
            runner_token = os.environ.get("GH_PAT_FOR_RUNNER_ADMIN") or self.token
            runner_headers = {
                "Authorization": f"token {runner_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            all_runners = []
            url = f"{self.base_url}/repos/{self.repo}/actions/runners"
            params = {"per_page": 100}

            while url:
                response = requests.get(
                    url, headers=runner_headers, params=params, timeout=30
                )
                if response.status_code != 200:
                    print(
                        f"  Warning: Runner API returned {response.status_code}: {response.text[:200]}"
                    )
                    return {}
                data = response.json()
                runners = data.get("runners", [])
                all_runners.extend(runners)

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
                params = {}  # Clear params for subsequent requests

            print(f"  Found {len(all_runners)} self-hosted runners")

            # Group runners by their labels (excluding common labels like "self-hosted")
            # A runner can have multiple labels, so count it for each relevant label
            runner_stats_by_label = defaultdict(
                lambda: {"online": 0, "total": 0, "busy": 0}
            )

            # Common labels to exclude (not useful for grouping)
            excluded_labels = {"self-hosted", "Linux", "X64", "ARM64"}

            for runner in all_runners:
                # Get all custom/relevant labels for this runner
                labels = [
                    label.get("name", "")
                    for label in runner.get("labels", [])
                    if label.get("name", "") not in excluded_labels
                ]

                # Count this runner for EACH of its relevant labels
                for runner_label in labels:
                    runner_stats_by_label[runner_label]["total"] += 1
                    if runner.get("status") == "online":
                        runner_stats_by_label[runner_label]["online"] += 1
                    if runner.get("busy", False):
                        runner_stats_by_label[runner_label]["busy"] += 1

            return dict(runner_stats_by_label)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching runners: {e}")
            return {}

    def parse_test_summary(self, logs: str) -> Optional[Dict]:
        """
        Parse the test summary block from job logs.

        Returns:
            Dict with passed/total counts and list of failed tests, or None if no summary found.
        """
        import re

        # Strip ANSI escape codes that GitHub Actions logs may contain
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        logs = ansi_escape.sub("", logs)

        # Look for the test summary pattern
        # Pattern matches: "Test Summary: 7/8 passed"
        summary_match = re.search(r"Test Summary:\s*(\d+)/(\d+)\s*passed", logs)
        if not summary_match:
            return None

        passed = int(summary_match.group(1))
        total = int(summary_match.group(2))

        # Find failed tests section
        # Look for "FAILED:" (the ✗ character may be mangled due to encoding)
        failed_tests = []
        # Match any character(s) before FAILED: (could be ✗, â, or other encoding artifacts)
        failed_section_match = re.search(
            r".?\s*FAILED:\s*\n(.*?)(?:={10,}|$)", logs, re.DOTALL
        )

        if failed_section_match:
            failed_section = failed_section_match.group(1)
            # Find all .py files - just look for non-whitespace ending in .py
            for match in re.finditer(r"(\S+\.py)", failed_section):
                full_path = match.group(1)
                # Extract just the filename from the path
                test_file = full_path.split("/")[-1] if "/" in full_path else full_path
                failed_tests.append(
                    {
                        "test_file": test_file,
                        "full_path": full_path,
                    }
                )

        return {
            "passed": passed,
            "total": total,
            "failed_tests": failed_tests,
        }

    def analyze_test_failures_for_job(self, recent_runs: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze test-level failures for a specific job across its recent runs.

        Args:
            recent_runs: List of recent run info dicts with job_id, job_url, conclusion, etc.
            debug: Enable debug logging

        Returns:
            Dict mapping test_file -> {
                "total_failures": int,
                "current_streak": int,
                "recent_runs": [{"run_number": ..., "job_url": ..., "status": ..., "failed": bool}, ...]
            }
        """
        test_failures: Dict[str, Dict] = defaultdict(
            lambda: {"total_failures": 0, "current_streak": 0, "recent_runs": []}
        )

        # Track whether we successfully parsed any test summaries
        parsed_any_test_summary = False

        # Process runs in chronological order (oldest first) to track streaks
        for run_info in recent_runs:
            job_id = run_info.get("job_id")
            conclusion = run_info.get("conclusion")

            # For failed jobs, fetch logs and parse test failures
            if conclusion == "failure" and job_id:
                logs = self.get_job_logs(job_id)
                test_summary = self.parse_test_summary(logs) if logs else None

                if test_summary and test_summary["failed_tests"]:
                    parsed_any_test_summary = True
                    # Track each failed test
                    failed_test_files = set()
                    for failed_test in test_summary["failed_tests"]:
                        test_file = failed_test["test_file"]
                        failed_test_files.add(test_file)
                        test_failures[test_file]["total_failures"] += 1
                        test_failures[test_file]["current_streak"] += 1
                        test_failures[test_file]["recent_runs"].append(
                            {
                                "run_number": run_info.get("run_number"),
                                "job_url": run_info.get("job_url"),
                                "status": "❌",
                                "failed": True,
                            }
                        )

                    # For tests we've seen before that didn't fail this time,
                    # they get a "pass" (the job failed but this specific test passed)
                    for test_file in test_failures.keys():
                        if test_file not in failed_test_files:
                            # Test passed in this run (job failed for other reasons)
                            test_failures[test_file]["current_streak"] = 0
                            test_failures[test_file]["recent_runs"].append(
                                {
                                    "run_number": run_info.get("run_number"),
                                    "job_url": run_info.get("job_url"),
                                    "status": "✅",
                                    "failed": False,
                                }
                            )
                else:
                    # Job failed but no test summary found - don't reset streaks, mark as unknown
                    for test_file in test_failures.keys():
                        test_failures[test_file]["recent_runs"].append(
                            {
                                "run_number": run_info.get("run_number"),
                                "job_url": run_info.get("job_url"),
                                "status": "⚪",  # Unknown - couldn't parse logs
                                "failed": None,
                            }
                        )
            elif conclusion == "success":
                # Job passed - all tests passed, reset streaks
                for test_file in test_failures.keys():
                    test_failures[test_file]["current_streak"] = 0
                    test_failures[test_file]["recent_runs"].append(
                        {
                            "run_number": run_info.get("run_number"),
                            "job_url": run_info.get("job_url"),
                            "status": "✅",
                            "failed": False,
                        }
                    )
            else:
                # Other conclusion (cancelled, skipped, etc.) - don't reset streaks, mark as unknown
                for test_file in test_failures.keys():
                    test_failures[test_file]["recent_runs"].append(
                        {
                            "run_number": run_info.get("run_number"),
                            "job_url": run_info.get("job_url"),
                            "status": "⚪",
                            "failed": None,
                        }
                    )

            time.sleep(0.1)  # Rate limiting for log fetches

        # If we couldn't parse any test summaries, return special marker
        if not parsed_any_test_summary:
            return {"_no_test_summary": True}

        # Convert to regular dict and sort by streak then total failures
        result = {}
        for test_file, data in test_failures.items():
            result[test_file] = {
                "total_failures": data["total_failures"],
                "current_streak": data["current_streak"],
                "recent_runs": data["recent_runs"][-10:],  # Keep last 10
            }

        return result

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
        job_recent_runs: Dict[str, List[Dict]] = defaultdict(list)  # Track last 10 runs

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
                        "job_id": job.get("id"),  # Needed for fetching logs
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
            # Get last 10 runs (oldest to latest, chronological order)
            recent_runs = job_recent_runs.get(job_name, [])[-10:]

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
                "recent_runs": recent_runs,  # Last 10 runs with status emoji
            }

        return job_streak_data, job_current_streak

    def analyze_test_failures_for_broken_jobs(
        self, job_streak_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Analyze test-level failures for jobs with current_streak >= 2 or failure_rate >= 50%.

        Args:
            job_streak_data: Dict mapping job_name -> job stats including recent_runs

        Returns:
            Dict mapping job_name -> {test_file -> test failure stats}
        """
        # Filter to only broken/high-failure-rate jobs
        jobs_to_analyze = [
            (job_name, data)
            for job_name, data in job_streak_data.items()
            if data["current_streak"] >= 2 or data["failure_rate"] >= 50.0
        ]

        if not jobs_to_analyze:
            print("No broken or high-failure-rate jobs to analyze for test failures")
            return {}

        print(f"\nAnalyzing test-level failures for {len(jobs_to_analyze)} jobs...")

        job_test_failures = {}
        for i, (job_name, data) in enumerate(jobs_to_analyze, 1):
            print(
                f"  [{i}/{len(jobs_to_analyze)}] Analyzing test failures for: {job_name}"
            )
            recent_runs = data.get("recent_runs", [])

            if recent_runs:
                test_failures = self.analyze_test_failures_for_job(recent_runs)
                if test_failures:
                    job_test_failures[job_name] = test_failures

        print(f"Found test-level failures for {len(job_test_failures)} jobs")
        return job_test_failures

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
        online_runners: Optional[Dict[str, Dict]] = None,
        # Test failures (per job -> per test)
        job_test_failures: Optional[Dict[str, Dict[str, Dict]]] = None,
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

        # Calculate PR Test and Nightly Test job counts for scheduled runs (main branch)
        pr_scheduled_combined = {
            **pr_test_nvidia_scheduled_data,
            **pr_test_amd_scheduled_data,
            **pr_test_xeon_scheduled_data,
            **pr_test_xpu_scheduled_data,
            **pr_test_npu_scheduled_data,
        }
        nightly_scheduled_combined = {
            **nightly_nvidia_scheduled_data,
            **nightly_amd_scheduled_data,
            **nightly_intel_scheduled_data,
            **nightly_npu_scheduled_data,
        }

        pr_main_count = len(pr_scheduled_combined)
        pr_main_with_streaks = sum(
            1 for d in pr_scheduled_combined.values() if d["current_streak"] >= 2
        )
        nightly_main_count = len(nightly_scheduled_combined)
        nightly_main_with_streaks = sum(
            1 for d in nightly_scheduled_combined.values() if d["current_streak"] >= 2
        )

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
                "pr_main_count": pr_main_count,
                "pr_main_with_streaks": pr_main_with_streaks,
                "nightly_main_count": nightly_main_count,
                "nightly_main_with_streaks": nightly_main_with_streaks,
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
            "job_test_failures": job_test_failures if job_test_failures else {},
            "online_runners": online_runners if online_runners else {},
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
            summary_lines.append(
                "_Note: Recent runs are shown oldest → latest (left to right)_"
            )
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

            # Runner Statistics - COLLAPSIBLE
            runner_stats = report_data.get("runner_stats", {})
            online_runners = report_data.get("online_runners", {})
            if runner_stats:
                summary_lines.append("<details>")
                summary_lines.append(
                    "<summary>📊 Runner Statistics (by type) (click to expand)</summary>"
                )
                summary_lines.append("")
                summary_lines.append(
                    "_High queue times indicate that runner type may need more workers. Online column shows current runner availability._"
                )
                summary_lines.append("")
                summary_lines.append(
                    "| Runner Type | Online | Avg Queue | P90 Queue | # of Jobs Processed | Jobs Using This Runner |"
                )
                summary_lines.append(
                    "|-------------|--------|-----------|-----------|---------------------|------------------------|"
                )

                # Sort by P90 queue time descending (longest waits first)
                sorted_runners = sorted(
                    runner_stats.items(),
                    key=lambda x: x[1].get("p90_queue_time_seconds", 0),
                    reverse=True,
                )

                for runner_key, stats in sorted_runners:
                    avg_queue = stats.get("avg_queue_time_seconds", 0)
                    p90_queue = stats.get("p90_queue_time_seconds", 0)
                    total_jobs = stats.get("total_jobs", 0)

                    # Get online runner count for this runner type
                    # First try exact match, then fall back to substring match
                    online_count = online_runners.get(runner_key)
                    if not online_count:
                        # Fall back to substring match (but prefer longer matches)
                        best_match = None
                        best_match_len = 0
                        for online_key, online_stats in online_runners.items():
                            if online_key in runner_key or runner_key in online_key:
                                # Prefer longer matching keys (more specific)
                                if len(online_key) > best_match_len:
                                    best_match = online_stats
                                    best_match_len = len(online_key)
                        online_count = best_match
                    if online_count:
                        online_str = f"{online_count['online']}/{online_count['total']}"
                    else:
                        online_str = "N/A"

                    # Get unique job names that run on this runner
                    jobs_total = stats.get("jobs_total", {})
                    unique_jobs = list(jobs_total.keys())
                    # Truncate job names and limit to first 3
                    job_names_short = [
                        (j if len(j) <= 25 else j[:22] + "...") for j in unique_jobs[:3]
                    ]
                    jobs_str = ", ".join(f"`{j}`" for j in job_names_short)
                    if len(unique_jobs) > 3:
                        jobs_str += f" +{len(unique_jobs) - 3} more"

                    # Format queue times
                    avg_str = f"{avg_queue / 60:.1f}m" if avg_queue > 0 else "N/A"
                    p90_str = f"{p90_queue / 60:.1f}m" if p90_queue > 0 else "N/A"

                    # Truncate long runner labels
                    display_name = (
                        runner_key if len(runner_key) <= 35 else runner_key[:32] + "..."
                    )

                    # Highlight if P90 queue time > 10 minutes (potential bottleneck)
                    if p90_queue > 600:
                        summary_lines.append(
                            f"| <span style='color:orange'>`{display_name}`</span> | <span style='color:orange'>{online_str}</span> | <span style='color:orange'>{avg_str}</span> | <span style='color:orange'>{p90_str}</span> | <span style='color:orange'>{total_jobs}</span> | {jobs_str} |"
                        )
                    else:
                        summary_lines.append(
                            f"| `{display_name}` | {online_str} | {avg_str} | {p90_str} | {total_jobs} | {jobs_str} |"
                        )

                summary_lines.append("")
                summary_lines.append("</details>")
                summary_lines.append("")

            # Get test failures data
            job_test_failures = report_data.get("job_test_failures", {})

            # Helper function to generate test failure dropdown for a job
            def generate_test_failure_dropdown(job_name: str) -> List[str]:
                """Generate collapsible test failure details for a job."""
                lines = []
                test_failures = job_test_failures.get(job_name, {})
                if not test_failures:
                    return lines

                # Check if we couldn't parse any test summaries for this job
                if test_failures.get("_no_test_summary"):
                    lines.append("<details>")
                    lines.append(
                        f"<summary>🔬 Test failures for <code>{job_name}</code> (no test summary available)</summary>"
                    )
                    lines.append("")
                    lines.append(
                        "_Could not parse test summary from job logs. The job may not produce a test summary, or the log format may have changed._"
                    )
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                    return lines

                # Filter out the marker key if present
                test_failures = {
                    k: v for k, v in test_failures.items() if not k.startswith("_")
                }
                if not test_failures:
                    return lines

                # Sort by current streak (descending), then total failures
                sorted_tests = sorted(
                    test_failures.items(),
                    key=lambda x: (x[1]["current_streak"], x[1]["total_failures"]),
                    reverse=True,
                )

                lines.append("<details>")
                lines.append(
                    f"<summary>🔬 Test failures for <code>{job_name}</code> ({len(sorted_tests)} tests)</summary>"
                )
                lines.append("")
                lines.append(
                    "| Test File | Failures | Streak | Recent Runs (oldest → latest) |"
                )
                lines.append(
                    "|-----------|----------|--------|-------------------------------|"
                )

                for test_file, test_data in sorted_tests[:10]:  # Limit to top 10 tests
                    total_failures = test_data["total_failures"]
                    current_streak = test_data["current_streak"]
                    recent_runs = test_data.get("recent_runs", [])

                    # Format streak with fire emoji if consecutive
                    streak_str = (
                        f"🔥 {current_streak}"
                        if current_streak >= 2
                        else str(current_streak)
                    )

                    # Build history links
                    if recent_runs:
                        history_links = "… " + " ".join(
                            [f"[{r['status']}]({r['job_url']})" for r in recent_runs]
                        )
                    else:
                        history_links = "N/A"

                    # Highlight tests with high streak
                    if current_streak >= 3:
                        lines.append(
                            f"| <span style='color:red'>`{test_file}`</span> | <span style='color:red'>{total_failures}</span> | <span style='color:red'>{streak_str}</span> | <span style='color:red'>{history_links}</span> |"
                        )
                    else:
                        lines.append(
                            f"| `{test_file}` | {total_failures} | {streak_str} | {history_links} |"
                        )

                lines.append("")
                lines.append("</details>")
                lines.append("")
                return lines

            # Helper function to generate job section for GitHub markdown
            def generate_job_section_md(
                title: str, data: Dict[str, Dict], show_test_failures: bool = True
            ):
                sorted_data = sorted(
                    data.items(),
                    key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                    reverse=True,
                )
                broken = [
                    (name, d) for name, d in sorted_data if d["current_streak"] >= 2
                ]
                high_failure_rate = [
                    (name, d)
                    for name, d in sorted_data
                    if d["current_streak"] < 2
                    and d["failure_rate"] >= 50.0
                    and d["total_failures"] > 0
                ]
                recently_failed = [
                    (name, d)
                    for name, d in sorted_data
                    if d["current_streak"] < 2
                    and d["failure_rate"] < 50.0
                    and d["total_failures"] > 0
                ]

                # Always show section header
                summary_lines.append(f"## {title}")
                summary_lines.append("")

                if broken:
                    # Add sub-header to clarify this shows ongoing failure streaks
                    summary_lines.append(
                        "🔥 **Consecutive failures (≥2) & currently failing**"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Current | Max | Runs | First | Last | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|---------|-----|------|-------|------|-------------------------------|"
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
                            history_links = "… " + " ".join(
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

                    # Add test failure dropdowns after the table (only for scheduled runs)
                    summary_lines.append("")
                    if show_test_failures:
                        for job_name, d in broken[:15]:
                            test_dropdown = generate_test_failure_dropdown(job_name)
                            if test_dropdown:
                                summary_lines.extend(test_dropdown)
                else:
                    summary_lines.append(
                        "✅ **No jobs with active failure streaks (streak >= 2)**"
                    )
                    summary_lines.append("")

                # Show high failure rate jobs (NOT collapsible)
                if high_failure_rate:
                    summary_lines.append(
                        "⚠️ **No current failure streak but high intermittent failure rate (≥50%)**"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Failures | Fail Rate | Total Runs | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|----------|-----------|------------|-------------------------------|"
                    )
                    for job_name, d in high_failure_rate[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = "… " + " ".join(
                                [
                                    f"[{r['status']}]({r['job_url']})"
                                    for r in recent_runs
                                ]
                            )
                        else:
                            history_links = "N/A"

                        # Highlight rows with failure rate >= 50%
                        summary_lines.append(
                            f"| <span style='color:orange'>`{display_name}`</span> | <span style='color:orange'>{d['total_failures']}</span> | <span style='color:orange'>{d['failure_rate']:.1f}%</span> | <span style='color:orange'>{d['total_runs']}</span> | <span style='color:orange'>{history_links}</span> |"
                        )

                    # Add test failure dropdowns after the table (only for scheduled runs)
                    summary_lines.append("")
                    if show_test_failures:
                        for job_name, d in high_failure_rate[:15]:
                            test_dropdown = generate_test_failure_dropdown(job_name)
                            if test_dropdown:
                                summary_lines.extend(test_dropdown)

                # Show recently failed jobs in a collapsible section
                if recently_failed:
                    # Extract just the workflow name without the run count for cleaner display
                    # e.g., "1. PR Test NVIDIA - Scheduled (latest 12 runs)" -> "PR Test NVIDIA - Scheduled"
                    short_title = title.split("(")[0].strip()
                    # Remove the leading number and period if present
                    if short_title and short_title[0].isdigit():
                        short_title = short_title.split(".", 1)[-1].strip()
                    # Get the max total_runs from recently_failed jobs to show the analysis window
                    max_total_runs = max(d["total_runs"] for _, d in recently_failed)
                    summary_lines.append("<details>")
                    summary_lines.append(
                        f"<summary>📋 [{short_title}] No current failure streak, but had failures in the past {max_total_runs} runs - {len(recently_failed)} jobs</summary>"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Failures | Fail Rate | Total Runs | Recent Runs (oldest → latest) |"
                    )
                    summary_lines.append(
                        "|----------|----------|-----------|------------|-------------------------------|"
                    )
                    for job_name, d in recently_failed[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )
                        recent_runs = d.get("recent_runs", [])
                        if recent_runs:
                            history_links = "… " + " ".join(
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

            # ========== RUNNERS (at the top) ==========
            summary_lines.append("---")
            summary_lines.append("# 🖥️ RUNNER HEALTH")
            summary_lines.append("")

            # Workers section
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
                summary_lines.append("## Workers")
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

            # PR Tests - General (5 workflows) - no test failure analysis
            generate_job_section_md(
                f"10. PR Test NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("pr_test_nvidia_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"11. PR Test AMD - General (latest {gen_limit} runs)",
                report_data.get("pr_test_amd_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"12. PR Test Xeon - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xeon_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"13. PR Test XPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_xpu_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"14. PR Test NPU - General (latest {gen_limit} runs)",
                report_data.get("pr_test_npu_general_data", {}),
                show_test_failures=False,
            )

            # Nightly Tests - General (4 workflows) - no test failure analysis
            generate_job_section_md(
                f"15. Nightly NVIDIA - General (latest {gen_limit} runs)",
                report_data.get("nightly_nvidia_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"16. Nightly AMD - General (latest {gen_limit} runs)",
                report_data.get("nightly_amd_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"17. Nightly Intel - General (latest {gen_limit} runs)",
                report_data.get("nightly_intel_general_data", {}),
                show_test_failures=False,
            )
            generate_job_section_md(
                f"18. Nightly NPU - General (latest {gen_limit} runs)",
                report_data.get("nightly_npu_general_data", {}),
                show_test_failures=False,
            )

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

        if not runner_runs and not pr_test_nvidia_scheduled_runs:
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

        # Fetch online runner status
        online_runners = analyzer.get_online_runners()

        # Analyze test-level failures for broken/high-failure-rate jobs
        # Combine all scheduled data for test failure analysis (main branch, most important)
        all_scheduled_data = {
            **pr_test_nvidia_scheduled_data,
            **pr_test_amd_scheduled_data,
            **pr_test_xeon_scheduled_data,
            **pr_test_xpu_scheduled_data,
            **pr_test_npu_scheduled_data,
            **nightly_nvidia_scheduled_data,
            **nightly_amd_scheduled_data,
            **nightly_intel_scheduled_data,
            **nightly_npu_scheduled_data,
        }
        job_test_failures = analyzer.analyze_test_failures_for_broken_jobs(
            all_scheduled_data
        )

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
            online_runners,
            # Test failures
            job_test_failures,
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
