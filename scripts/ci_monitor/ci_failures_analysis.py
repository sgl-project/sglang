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
    python ci_failures_analysis.py --token <GITHUB_TOKEN> --limit 500 --threshold 3
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

    def __init__(self, token: str, alert_threshold: int = 3):
        self.token = token
        self.alert_threshold = alert_threshold
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Failures-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Target workflows to monitor
        self.target_workflows = [
            "PR Test",  # Nvidia GPU tests
            "PR Test (AMD)",  # AMD GPU tests
            "PR Test (Xeon)",  # Intel Xeon CPU tests
        ]

        # Jobs to EXCLUDE from analysis (administrative/setup jobs, not actual tests)
        self.excluded_jobs = [
            "check-changes",
            "pr-test-finish",  # Nvidia workflow teardown
            "pr-test-amd-finish",  # AMD workflow teardown
            "call-gate",
            "pr-gate",
        ]

    def get_recent_runs(self, limit: int = 500) -> List[Dict]:
        """
        Fetch recent workflow runs from GitHub API.
        Keeps fetching until we have 'limit' runs from target workflows.
        """
        print(
            f"Fetching until we have {limit} runs from target workflows (PR Test, PR Test AMD, PR Test Xeon)..."
        )

        filtered_runs = []
        page = 1
        per_page = 100
        max_pages = 100  # Safety limit to prevent infinite loops (10,000 total runs)

        while len(filtered_runs) < limit and page <= max_pages:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": per_page, "page": page}

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    print("No more workflow runs available")
                    break

                # Filter this batch to target workflows
                batch_filtered = [
                    run
                    for run in data["workflow_runs"]
                    if run.get("name") in self.target_workflows
                    and run.get("status") == "completed"
                ]

                filtered_runs.extend(batch_filtered)
                print(
                    f"Fetched {len(filtered_runs)} target workflow runs so far (scanned page {page})..."
                )

                # If GitHub returned fewer than per_page, we've reached the end
                if len(data["workflow_runs"]) < per_page:
                    print("Reached end of available workflow runs")
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching workflow runs: {e}")
                break

        if page > max_pages:
            print(
                f"Warning: Reached max pages limit ({max_pages}). Consider reducing --limit or increasing max_pages."
            )

        print(f"Collected {len(filtered_runs)} completed target workflow runs")
        return filtered_runs[:limit]

    def get_jobs_for_run(self, run_id: int) -> List[Dict]:
        """Get all jobs for a specific workflow run."""
        try:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs", [])
            return jobs
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
        runner_error_signatures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Track consecutive failures per runner instance
        runner_instance_current_streak: Dict[str, int] = defaultdict(int)
        runner_instance_max_streak: Dict[str, int] = defaultdict(int)
        runner_instance_first_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_last_failure: Dict[str, Optional[Dict]] = {}
        runner_instance_recovery: Dict[str, Optional[Dict]] = {}
        runner_instance_error_signatures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

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

                    # Extract error signature for runner
                    error_signature = self._extract_error_signature(job)
                    if error_signature:
                        runner_error_signatures[runner_key][error_signature] += 1

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

                        # Extract error signature for runner instance
                        if error_signature:
                            runner_instance_error_signatures[runner_instance_key][
                                error_signature
                            ] += 1

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
            # Get top 3 error signatures for this runner
            error_sigs = runner_error_signatures.get(runner_key, {})
            top_errors = sorted(error_sigs.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]

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
                "top_error_signatures": top_errors,
            }

        # Build runner instance streak data
        runner_instance_streak_data = {}
        for instance_key in runner_instance_stats.keys():
            # Get top 3 error signatures for this runner instance
            error_sigs = runner_instance_error_signatures.get(instance_key, {})
            top_errors = sorted(error_sigs.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]

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
                "top_error_signatures": top_errors,
            }

        return (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        )

    def _extract_error_signature(self, job: Dict) -> str:
        """
        Extract error signature from a failed job.

        Returns a simplified error type string.
        """
        # Check if job has steps with failures
        steps = job.get("steps", [])
        if not steps:
            return "Unknown Error"

        # Look for failed steps
        failed_steps = [s for s in steps if s.get("conclusion") == "failure"]
        if not failed_steps:
            return "Unknown Error"

        # Try to fetch and parse logs for the first failed step
        first_failed_step = failed_steps[0]
        step_number = first_failed_step.get("number")

        # Attempt to get detailed error from logs
        if step_number is not None:
            try:
                job_id = job.get("id")
                # Fetch logs for this specific step
                log_url = (
                    f"{self.base_url}/repos/{self.repo}/actions/jobs/{job_id}/logs"
                )
                response = self.session.get(log_url, timeout=10)

                if response.status_code == 200:
                    log_text = response.text

                    # Check for specific error patterns in logs (case-insensitive)
                    log_lower = log_text.lower()

                    # CUDA/GPU Memory errors (most common for GPU clusters)
                    if (
                        "cuda out of memory" in log_lower
                        or "cudaerror: out of memory" in log_lower
                    ):
                        return "CUDA OOM"
                    elif "out of memory" in log_lower and (
                        "gpu" in log_lower or "device" in log_lower
                    ):
                        return "GPU OOM"
                    elif "out of memory" in log_lower and "cuda" not in log_lower:
                        return "Out of Memory"

                    # CUDA/GPU device errors
                    if (
                        "cuda error: device-side assert" in log_lower
                        or "device-side assert" in log_lower
                    ):
                        return "CUDA Device Assert"
                    elif (
                        "cuda error: an illegal memory access" in log_lower
                        or "illegal memory access" in log_lower
                    ):
                        return "CUDA Illegal Memory Access"
                    elif "cuda error" in log_lower or "cudaerror" in log_lower:
                        return "CUDA Error"
                    elif "gpu" in log_lower and (
                        "hang" in log_lower or "hung" in log_lower
                    ):
                        return "GPU Hang"
                    elif (
                        "no cuda-capable device" in log_lower
                        or "cuda device count" in log_lower
                        and "0" in log_lower
                    ):
                        return "No GPU Available"

                    # ROCm/AMD GPU errors
                    if (
                        "hipoutofmemoryerror" in log_lower
                        or "hip out of memory" in log_lower
                    ):
                        return "ROCm OOM"
                    elif "hiperror" in log_lower or "rocm error" in log_lower:
                        return "ROCm/HIP Error"

                    # NCCL/collective communication errors (multi-GPU)
                    if "nccl error" in log_lower or "ncclerror" in log_lower:
                        return "NCCL Error"
                    elif "timeout after" in log_lower and "nccl" in log_lower:
                        return "NCCL Timeout"

                    # Process/system errors
                    if "killed" in log_lower and (
                        "oom" in log_lower or "out of memory" in log_lower
                    ):
                        return "Process Killed (OOM)"
                    elif "killed" in log_lower or "sigkill" in log_lower:
                        return "Process Killed"
                    elif "segmentation fault" in log_lower or "sigsegv" in log_lower:
                        return "Segmentation Fault"

                    # Timeout errors
                    if "timeout" in log_lower or "timed out" in log_lower:
                        return "Timeout"

                    # Connection/network errors
                    if (
                        "connection refused" in log_lower
                        or "connection reset" in log_lower
                    ):
                        return "Connection Error"
                    elif "ssh" in log_lower and (
                        "failed" in log_lower or "error" in log_lower
                    ):
                        return "SSH Error"

                    # Import/module errors
                    if "modulenotfounderror" in log_lower or "importerror" in log_lower:
                        return "Import Error"

                    # Assertion errors
                    if "assertionerror" in log_lower:
                        return "Assertion Error"

                    # Pytest-specific errors
                    if (
                        "pytest" in log_lower
                        and "error" in log_lower
                        and "collection" in log_lower
                    ):
                        return "Pytest Collection Error"

            except Exception:
                # If log fetching fails, fall back to step name analysis
                pass

        # Fallback to step name analysis if we couldn't get logs or didn't find specific errors
        step_name = first_failed_step.get("name", "Unknown Step")

        # Simplify common patterns based on step name
        if "timeout" in step_name.lower():
            return "Timeout"
        elif "setup" in step_name.lower() or "install" in step_name.lower():
            return "Setup/Installation Error"
        elif "test" in step_name.lower():
            return f"Test Failure: {step_name[:50]}"
        elif "build" in step_name.lower():
            return "Build Error"
        else:
            return f"Step Failed: {step_name[:50]}"

    def construct_cron_failures_on_main(
        self, runs: List[Dict], overall_job_streak_data: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Analyses consecutive failures for each job triggered by cron on main branch only.
        Compares error signatures with overall data to detect if main-branch failures
        have same or different error patterns than PR-triggered failures.

        Args:
            runs: All workflow runs (will be filtered to cron-triggered only)
            overall_job_streak_data: Overall job streak data (from all runs) for comparison

        Returns:
            Tuple of (main_streak_data, job_current_streaks_main)
            - main_streak_data: Same structure as job_streak_data, plus 'matches_overall_error' field
            - job_current_streaks_main: Dict mapping job name to current streak count on main
        """
        print(
            "\nAnalyzing consecutive failures on main branch (cron-triggered runs only)..."
        )

        # Filter to only cron-triggered runs (scheduled runs)
        # Scheduled/cron runs have event == 'schedule'
        cron_runs = [run for run in runs if run.get("event") == "schedule"]

        print(
            f"Found {len(cron_runs)} cron-triggered runs out of {len(runs)} total runs"
        )

        if not cron_runs:
            print("No cron-triggered runs found")
            return {}, {}

        # Reuse existing analyze_consecutive_failures on filtered runs
        main_streak_data, job_current_streaks_main = self.analyze_consecutive_failures(
            cron_runs
        )

        # Now add comparison with overall data for at-a-glance diagnostics
        for job_name, main_data in main_streak_data.items():
            matches_overall_error = False

            if job_name in overall_job_streak_data:
                main_top_errors = main_data.get("top_error_signatures", [])
                overall_top_errors = overall_job_streak_data[job_name].get(
                    "top_error_signatures", []
                )

                if main_top_errors and overall_top_errors:
                    # Check if the most common error on main matches the most common overall error
                    main_top_error = main_top_errors[0][0]
                    overall_top_error = overall_top_errors[0][0]
                    matches_overall_error = main_top_error == overall_top_error

            # Add comparison flag to the data
            main_data["matches_overall_error"] = matches_overall_error

        return main_streak_data, job_current_streaks_main

    def analyze_consecutive_failures(
        self, runs: List[Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Analyze consecutive failures for each job.

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
        job_error_signatures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

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

                    # Extract error signature from job
                    error_signature = self._extract_error_signature(job)
                    if error_signature:
                        job_error_signatures[job_name][error_signature] += 1

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

            time.sleep(0.05)

        # Build final results
        job_streak_data = {}
        for job_name in job_current_streak.keys():
            # Get top 3 error signatures
            error_sigs = job_error_signatures.get(job_name, {})
            top_errors = sorted(error_sigs.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]

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
                "top_error_signatures": top_errors,
            }

        return job_streak_data, job_current_streak

    def detect_alerts(
        self,
        job_streak_data: Dict[str, Dict],
        job_current_streaks: Dict[str, int],
        runner_stats: Optional[Dict[str, Dict]] = None,
        runner_instance_data: Optional[Dict[str, Dict]] = None,
        runner_streak_data: Optional[Dict[str, Dict]] = None,
        runner_instance_streak_data: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect jobs and runners that need alerts based on thresholds.

        Returns:
            Tuple of (job_alerts, runner_alerts)
        """
        job_alerts = []

        for job_name, data in job_streak_data.items():
            current_streak = data["current_streak"]

            # Alert condition: consecutive failures >= threshold
            if current_streak >= self.alert_threshold:
                job_alerts.append(
                    {
                        "job_name": job_name,
                        "current_streak": current_streak,
                        "max_streak": data["max_streak"],
                        "failure_rate": data["failure_rate"],
                        "first_failure": data["first_failure_in_streak"],
                        "last_failure": data["last_failure_in_streak"],
                        "top_error_signatures": data.get("top_error_signatures", []),
                        "alert_type": "consecutive_failures",
                        "severity": "high" if current_streak >= 5 else "medium",
                    }
                )

        # Detect runner alerts
        runner_alerts = []

        # Alert for runners with consecutive failures
        if runner_streak_data:
            for runner_labels, streak_data in runner_streak_data.items():
                if streak_data["current_streak"] >= self.alert_threshold:
                    runner_alerts.append(
                        {
                            "runner_labels": runner_labels,
                            "current_streak": streak_data["current_streak"],
                            "max_streak": streak_data["max_streak"],
                            "failure_rate": streak_data["failure_rate"],
                            "total_failures": streak_data["total_failures"],
                            "total_jobs": streak_data["total_jobs"],
                            "jobs_failed": streak_data.get("jobs_failed", {}),
                            "first_failure": streak_data["first_failure_in_streak"],
                            "last_failure": streak_data["last_failure_in_streak"],
                            "top_error_signatures": streak_data.get(
                                "top_error_signatures", []
                            ),
                            "alert_type": "runner_consecutive_failures",
                            "severity": (
                                "high"
                                if streak_data["current_streak"] >= 5
                                else "medium"
                            ),
                        }
                    )

        # Alert for runner instances with consecutive failures
        if runner_instance_streak_data:
            for instance_key, streak_data in runner_instance_streak_data.items():
                if streak_data["current_streak"] >= self.alert_threshold:
                    # Get queue time info from runner_instance_data
                    instance_data = runner_instance_data.get(instance_key, {})
                    avg_queue = instance_data.get("avg_queue_time_seconds", 0)

                    runner_alerts.append(
                        {
                            "runner_instance": instance_key,
                            "runner_name": streak_data.get("runner_name", "unknown"),
                            "current_streak": streak_data["current_streak"],
                            "max_streak": streak_data["max_streak"],
                            "failure_rate": streak_data["failure_rate"],
                            "total_failures": streak_data["total_failures"],
                            "total_jobs": streak_data["total_jobs"],
                            "jobs_failed": streak_data.get("jobs_failed", {}),
                            "first_failure": streak_data["first_failure_in_streak"],
                            "last_failure": streak_data["last_failure_in_streak"],
                            "top_error_signatures": streak_data.get(
                                "top_error_signatures", []
                            ),
                            "avg_queue_time_seconds": avg_queue,
                            "alert_type": "runner_instance_consecutive_failures",
                            "severity": (
                                "high"
                                if streak_data["current_streak"] >= 5
                                else "medium"
                            ),
                        }
                    )

        if runner_stats:
            # Alert if runner has high failure rate (>30%) and multiple jobs failing
            for runner_labels, stats in runner_stats.items():
                if (
                    stats["failure_rate"] > 50
                    and stats["unique_jobs_with_failures"] >= 3
                ):
                    runner_alerts.append(
                        {
                            "runner_labels": runner_labels,
                            "failure_rate": stats["failure_rate"],
                            "total_jobs": stats["total_jobs"],
                            "failed_jobs": stats["failed_jobs"],
                            "unique_jobs_with_failures": stats[
                                "unique_jobs_with_failures"
                            ],
                            "alert_type": "runner_health",
                            "severity": (
                                "high" if stats["failure_rate"] > 50 else "medium"
                            ),
                        }
                    )

        # Check for specific runner instances with concerning patterns
        if runner_instance_data:
            for instance_key, stats in runner_instance_data.items():
                # Alert if a specific runner instance has >50% failure rate with >=3 jobs
                if stats["failure_rate"] > 50 and stats["total_jobs"] >= 3:
                    runner_alerts.append(
                        {
                            "runner_instance": instance_key,
                            "runner_name": stats.get("runner_name", "unknown"),
                            "failure_rate": stats["failure_rate"],
                            "total_jobs": stats["total_jobs"],
                            "failed_jobs": stats["failed_jobs"],
                            "jobs_failed": stats["jobs_failed"],
                            "alert_type": "runner_instance_health",
                            "severity": "high",
                        }
                    )

        return job_alerts, runner_alerts

    # print statements here mainly for local testing
    def generate_failure_report(
        self,
        job_streak_data: Dict[str, Dict],
        job_alerts: List[Dict],
        runner_stats: Optional[Dict[str, Dict]] = None,
        runner_instance_data: Optional[Dict[str, Dict]] = None,
        runner_alerts: Optional[List[Dict]] = None,
        runner_streak_data: Optional[Dict[str, Dict]] = None,
        runner_instance_streak_data: Optional[Dict[str, Dict]] = None,
        main_streak_data: Optional[Dict[str, Dict]] = None,
        output_file: Optional[str] = None,
    ):
        """Generate detailed failure analysis report."""
        print("\n" + "=" * 80)
        print("SGLang Consecutive Failures Analysis Report")
        print("=" * 80)

        # Sort jobs by current streak (descending)
        sorted_jobs = sorted(
            job_streak_data.items(),
            key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
            reverse=True,
        )

        # Summary Statistics
        print("\n## Summary Statistics")
        print(
            f"Total (unique) jobs analyzed across PR Test workflows: {len(sorted_jobs)}"
        )
        print(
            f"Jobs with Active Failure Streaks: {sum(1 for j in sorted_jobs if j[1]['current_streak'] > 0)}"
        )
        print(f"Job Alerts Triggered: {len(job_alerts)}")

        # Add counter for main branch cron jobs
        if main_streak_data:
            main_jobs_count = len(main_streak_data)
            main_jobs_with_streaks = sum(
                1 for j in main_streak_data.values() if j["current_streak"] > 0
            )
            print(
                f"Jobs on Main Branch (cron-triggered): {main_jobs_count} ({main_jobs_with_streaks} with active streaks)"
            )
        else:
            print(f"Jobs on Main Branch (cron-triggered): 0 (no cron runs found)")

        if runner_stats:
            print(f"Total Runners Analyzed: {len(runner_stats)}")
            print(
                f"Runner Alerts Triggered: {len(runner_alerts) if runner_alerts else 0}"
            )

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

        # ALERTS: Critical Consecutive Job Failures (streak >= 2)
        if job_alerts:
            # Filter alerts with streak >= 2
            filtered_job_alerts = [a for a in job_alerts if a["current_streak"] >= 2]

            if filtered_job_alerts:
                print("\n" + "=" * 150)
                print(
                    "## ALERTS: Critical Consecutive Job Failures (PR + Scheduled, streak >= 2)"
                )
                print("=" * 150)
                print(
                    f"\n{'Job Name':<40} {'Streak':<8} {'Max':<6} {'First Failure':<16} {'Last Failure':<16} {'Top Errors':<60}"
                )
                print("-" * 150)

                for alert in sorted(
                    filtered_job_alerts, key=lambda x: x["current_streak"], reverse=True
                ):
                    job_name = alert["job_name"]
                    display_name = (
                        job_name if len(job_name) <= 38 else job_name[:35] + "..."
                    )

                    first_failure = alert.get("first_failure")
                    first_failure_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )

                    last_failure = alert.get("last_failure")
                    last_failure_str = (
                        f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                    )

                    # Format top errors - don't truncate
                    top_errors = alert.get("top_error_signatures", [])
                    if top_errors:
                        error_display = ", ".join(
                            [f"{err[0]} ({err[1]})" for err in top_errors]
                        )
                    else:
                        error_display = "N/A"

                    print(
                        f"{display_name:<40} {alert['current_streak']:<8} {alert['max_streak']:<6} {first_failure_str:<16} {last_failure_str:<16} {error_display:<60}"
                    )
            else:
                print("\n" + "=" * 100)
                print(
                    "## ALERTS: Critical Consecutive Job Failures (PR + Scheduled, streak >= 2)"
                )
                print("=" * 100)
                print(
                    "\nNothing to display (no jobs with consecutive failure streak >= 2)"
                )

        # ALERTS: Runners with Issues (streak >= 2)
        if runner_alerts:
            # Only show consecutive failure alerts with streak >= 2, and only machine instances
            instance_alerts = [
                a
                for a in runner_alerts
                if a["alert_type"] == "runner_instance_consecutive_failures"
                and a.get("current_streak", 0) >= 2
            ]

            if instance_alerts:
                print("\n" + "=" * 170)
                print("## ALERTS: Runners with Issues (streak >= 2)")
                print("=" * 170)
                print("\n### Runner Consecutive Failures")
                print(
                    f"\n{'Runner':<30} {'Str':<5} {'Max':<5} {'Fail%':<7} {'AvgQ':<7} {'First':<13} {'Last':<13} {'Top Errors':<45} {'Jobs Failed':<40}"
                )
                print("-" * 170)

                for alert in sorted(
                    instance_alerts,
                    key=lambda x: x.get("current_streak", 0),
                    reverse=True,
                ):
                    # Use the actual machine name instead of labels or instance key
                    runner_name = alert.get("runner_name", "unknown")
                    display_name = (
                        runner_name
                        if len(runner_name) <= 28
                        else runner_name[:25] + "..."
                    )

                    # Get all failed jobs - don't truncate
                    jobs_failed = alert.get("jobs_failed", {})
                    top_jobs = sorted(
                        jobs_failed.items(), key=lambda x: x[1], reverse=True
                    )
                    jobs_display = (
                        ", ".join([f"{job} ({count})" for job, count in top_jobs])
                        if top_jobs
                        else "N/A"
                    )

                    # Format queue time
                    avg_queue = alert.get("avg_queue_time_seconds", 0)
                    avg_queue_str = f"{avg_queue / 60:.1f}m" if avg_queue > 0 else "N/A"

                    first_failure = alert.get("first_failure")
                    first_failure_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )

                    last_failure = alert.get("last_failure")
                    last_failure_str = (
                        f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                    )

                    # Format top errors - don't truncate
                    top_errors = alert.get("top_error_signatures", [])
                    if top_errors:
                        error_display = ", ".join(
                            [f"{err[0]} ({err[1]})" for err in top_errors]
                        )
                    else:
                        error_display = "N/A"

                    print(
                        f"{display_name:<30} {alert['current_streak']:<5} {alert['max_streak']:<5} {alert['failure_rate']:>5.1f}% {avg_queue_str:<7} {first_failure_str:<13} {last_failure_str:<13} {error_display:<45} {jobs_display:<40}"
                    )
            else:
                print("\n" + "=" * 100)
                print("## ALERTS: Runners with Issues (streak >= 2)")
                print("=" * 100)
                print(
                    "\nNothing to display (no runners with consecutive failure streak >= 2)"
                )

        # Main Branch Health Section: Jobs failing on cron-triggered main branch runs
        if main_streak_data:
            # Sort by current streak (descending)
            sorted_main_jobs = sorted(
                main_streak_data.items(),
                key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                reverse=True,
            )

            # Show only jobs with streak >= 2
            broken_main_jobs = [
                (name, data)
                for name, data in sorted_main_jobs
                if data["current_streak"] >= 2
            ]

            if broken_main_jobs:
                print("\n" + "=" * 140)
                print(
                    f"## MAIN BRANCH HEALTH: Failing Jobs on Scheduled Main Branch Runs ({len(broken_main_jobs)} jobs)"
                )
                print("=" * 140)
                print(
                    f"\n{'Job Name':<40} {'Streak':<8} {'Max':<6} {'First':<13} {'Last':<13} {'Top Errors':<50}"
                )
                print("-" * 140)
                for job_name, data in broken_main_jobs[:15]:
                    display_name = (
                        job_name if len(job_name) <= 38 else job_name[:35] + "..."
                    )

                    # Get first and last failure info
                    first_failure = data.get("first_failure_in_streak")
                    first_failure_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )

                    last_failure = data.get("last_failure_in_streak")
                    last_failure_str = (
                        f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                    )

                    # Format top errors - don't truncate
                    top_errors = data.get("top_error_signatures", [])
                    if top_errors:
                        error_display = ", ".join(
                            [f"{err[0]} ({err[1]})" for err in top_errors]
                        )
                    else:
                        error_display = "N/A"

                    print(
                        f"{display_name:<40} {data['current_streak']:<8} {data['max_streak']:<6} {first_failure_str:<13} {last_failure_str:<13} {error_display:<50}"
                    )
            else:
                print("\n" + "=" * 100)
                print("## MAIN BRANCH HEALTH: Scheduled Main Branch Runs")
                print("=" * 100)
                print(
                    "\n  No consecutive failing jobs (streak >= 2) on main branch scheduled runs"
                )

        # Section 1: Currently Broken Jobs (streak >= 2)
        broken_jobs = [
            (name, data) for name, data in sorted_jobs if data["current_streak"] >= 2
        ]

        if broken_jobs:
            print("\n" + "=" * 140)
            print(
                "## Section 1: Top 15 Consecutively Failing Jobs (PR + Scheduled, streak >= 2)"
            )
            print("=" * 140)
            print(
                f"\n{'Job Name':<40} {'Streak':<8} {'Max':<6} {'First':<13} {'Last':<13} {'Top Errors':<50}"
            )
            print("-" * 140)
            for job_name, data in broken_jobs[:20]:
                display_name = (
                    job_name if len(job_name) <= 38 else job_name[:35] + "..."
                )

                # Get first and last failure info
                first_failure = data.get("first_failure_in_streak")
                first_failure_str = (
                    f"Run #{first_failure['run_number']}" if first_failure else "N/A"
                )

                last_failure = data.get("last_failure_in_streak")
                last_failure_str = (
                    f"Run #{last_failure['run_number']}" if last_failure else "N/A"
                )

                # Format top errors - don't truncate
                top_errors = data.get("top_error_signatures", [])
                if top_errors:
                    error_display = ", ".join(
                        [f"{err[0]} ({err[1]})" for err in top_errors]
                    )
                else:
                    error_display = "N/A"

                print(
                    f"{display_name:<40} {data['current_streak']:<8} {data['max_streak']:<6} {first_failure_str:<13} {last_failure_str:<13} {error_display:<50}"
                )

        # Section 2: Runner Health Analysis - Use machine names from runner instances (streak >= 2)
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
                        "top_error_signatures": streak_data.get(
                            "top_error_signatures", []
                        ),
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

            if runners_with_issues:
                print("\n" + "=" * 160)
                print(
                    "## Section 2: Top 15 Workers by Consecutive Failures (streak >= 2)"
                )
                print("=" * 160)
                print(
                    f"\n{'Machine Name':<30} {'Str':<5} {'Max':<5} {'Fail%':<7} {'AvgQ':<7} {'First':<13} {'Last':<13} {'Top Errors':<45} {'Total Jobs':<11} {'Unique Jobs':<12}"
                )
                print("-" * 160)

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

                    # Format top errors - don't truncate
                    top_errors = runner_data.get("top_error_signatures", [])
                    if top_errors:
                        error_display = ", ".join(
                            [f"{err[0]} ({err[1]})" for err in top_errors]
                        )
                    else:
                        error_display = "N/A"

                    print(
                        f"{display_name:<30} {streak_str:<5} {max_str:<5} {runner_data['failure_rate']:>5.1f}% {avg_queue_str:<7} {first_failure_str:<13} {last_failure_str:<13} {error_display:<45} {runner_data['total_jobs']:<11} {runner_data['unique_jobs']:<12}"
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

        # Calculate main branch stats
        main_jobs_count = len(main_streak_data) if main_streak_data else 0
        main_jobs_with_streaks = (
            sum(1 for j in main_streak_data.values() if j["current_streak"] > 0)
            if main_streak_data
            else 0
        )

        report_data = {
            "summary": {
                "total_jobs": len(sorted_jobs),
                "jobs_with_streaks": sum(
                    1 for j in sorted_jobs if j[1]["current_streak"] > 0
                ),
                "job_alerts_triggered": len(job_alerts),
                "runner_alerts_triggered": len(runner_alerts) if runner_alerts else 0,
                "total_runners": len(runner_stats) if runner_stats else 0,
                "alert_threshold": self.alert_threshold,
                "analysis_timestamp": datetime.now().isoformat(),
                "avg_queue_time_seconds": overall_avg_queue,
                "p90_queue_time_seconds": overall_p90_queue,
                "main_jobs_count": main_jobs_count,
                "main_jobs_with_streaks": main_jobs_with_streaks,
            },
            "job_streak_data": {
                job_name: {
                    **data,
                    # Convert datetime objects to strings for JSON serialization
                    "first_failure_in_streak": data["first_failure_in_streak"],
                    "recovery_info": data["recovery_info"],
                }
                for job_name, data in sorted_jobs
            },
            "job_alerts": job_alerts,
            "runner_stats": runner_stats if runner_stats else {},
            "runner_instance_data": (
                runner_instance_data if runner_instance_data else {}
            ),
            "runner_streak_data": runner_streak_data if runner_streak_data else {},
            "runner_instance_streak_data": (
                runner_instance_streak_data if runner_instance_streak_data else {}
            ),
            "runner_alerts": runner_alerts if runner_alerts else [],
            "main_streak_data": main_streak_data if main_streak_data else {},
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
                f"**Alert Threshold:** {report_data['summary']['alert_threshold']} consecutive failures"
            )
            summary_lines.append("")

            # Summary stats
            summary_lines.append("## Summary Statistics")
            summary_lines.append("")
            summary_lines.append("| Metric | Count |")
            summary_lines.append("|--------|-------|")
            summary_lines.append(
                f"| Total (unique) jobs analyzed across PR Test workflows | {report_data['summary']['total_jobs']} |"
            )
            summary_lines.append(
                f"| Jobs with Active Failure Streaks | {report_data['summary']['jobs_with_streaks']} |"
            )
            summary_lines.append(
                f"| Job Alerts Triggered | {report_data['summary']['job_alerts_triggered']} |"
            )

            # Add main branch job counter
            main_jobs_count = report_data["summary"].get("main_jobs_count", 0)
            main_jobs_with_streaks = report_data["summary"].get(
                "main_jobs_with_streaks", 0
            )
            if main_jobs_count > 0:
                summary_lines.append(
                    f"| Jobs on Main Branch (cron-triggered) | {main_jobs_count} ({main_jobs_with_streaks} with active streaks) |"
                )
            else:
                summary_lines.append(
                    f"| Jobs on Main Branch (cron-triggered) | 0 (no cron runs found) |"
                )

            summary_lines.append(
                f"| Total Runners Analyzed | {report_data['summary']['total_runners']} |"
            )
            summary_lines.append(
                f"| Runner Alerts Triggered | {report_data['summary']['runner_alerts_triggered']} |"
            )
            summary_lines.append("")

            # Queue Time Summary
            if report_data.get("summary", {}).get("avg_queue_time_seconds") is not None:
                summary_lines.append("## Queue Time Summary")
                summary_lines.append("")
                summary_lines.append("| Metric | Value |")
                summary_lines.append("|--------|-------|")
                avg_queue = report_data["summary"]["avg_queue_time_seconds"]
                p90_queue = report_data["summary"]["p90_queue_time_seconds"]
                summary_lines.append(
                    f"| Average Queue Time (across all runners) | {avg_queue / 60:.1f} minutes ({avg_queue:.0f}s) |"
                )
                summary_lines.append(
                    f"| P90 Queue Time (across all runners) | {p90_queue / 60:.1f} minutes ({p90_queue:.0f}s) |"
                )
                summary_lines.append("")

            # Job Alerts section (streak >= 2)
            if report_data.get("job_alerts"):
                # Filter alerts with streak >= 2
                filtered_job_alerts = [
                    a for a in report_data["job_alerts"] if a["current_streak"] >= 2
                ]

                if filtered_job_alerts:
                    summary_lines.append(
                        "## Alerts: Critical Consecutive Job Failures (PR + Scheduled, streak >= 2)"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Streak | Max | First Failure | Last Failure | Top Errors |"
                    )
                    summary_lines.append(
                        "|----------|--------|-----|---------------|--------------|------------|"
                    )

                    for alert in sorted(
                        filtered_job_alerts,
                        key=lambda x: x["current_streak"],
                        reverse=True,
                    ):
                        job_name = alert["job_name"]
                        if len(job_name) > 35:
                            job_name = job_name[:32] + "..."

                        first_failure = alert.get("first_failure")
                        if first_failure:
                            first_failure_str = f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                        else:
                            first_failure_str = "N/A"

                        last_failure = alert.get("last_failure")
                        if last_failure:
                            last_failure_str = f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                        else:
                            last_failure_str = "N/A"

                        # Format top errors as bullet list
                        top_errors = alert.get("top_error_signatures", [])
                        if top_errors:
                            error_str = "<br>".join(
                                [f"• {err[0]} ({err[1]})" for err in top_errors]
                            )
                        else:
                            error_str = "N/A"

                        summary_lines.append(
                            f"| `{job_name}` | {alert['current_streak']} | {alert['max_streak']} | "
                            f"{first_failure_str} | {last_failure_str} | {error_str} |"
                        )

                    summary_lines.append("")
                else:
                    summary_lines.append(
                        "## Alerts: Critical Consecutive Job Failures (PR + Scheduled, streak >= 2)"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "Nothing to display (no jobs with consecutive failure streak >= 2)"
                    )
                    summary_lines.append("")

            # Runner Alerts section (streak >= 2)
            if report_data.get("runner_alerts"):
                # Only show consecutive failure alerts with streak >= 2, and only machine instances
                instance_alerts = [
                    a
                    for a in report_data["runner_alerts"]
                    if a["alert_type"] == "runner_instance_consecutive_failures"
                    and a.get("current_streak", 0) >= 2
                ]

                if instance_alerts:
                    summary_lines.append("## Alerts: Workers with Issues (streak >= 2)")
                    summary_lines.append("")
                    summary_lines.append(
                        "| Runner | Streak | Max | Fail Rate | Avg Queue | First Failure | Last Failure | Top Errors | Jobs Failed |"
                    )
                    summary_lines.append(
                        "|--------|--------|-----|-----------|-----------|---------------|--------------|------------|-------------|"
                    )

                    for alert in sorted(
                        instance_alerts,
                        key=lambda x: x.get("current_streak", 0),
                        reverse=True,
                    ):
                        # Use the actual machine name instead of labels or instance key
                        runner_name = alert.get("runner_name", "unknown")
                        if len(runner_name) > 28:
                            runner_name = runner_name[:25] + "..."

                        # Get all failed jobs as bullet list
                        jobs_failed = alert.get("jobs_failed", {})
                        top_jobs = sorted(
                            jobs_failed.items(), key=lambda x: x[1], reverse=True
                        )
                        jobs_str = (
                            "<br>".join(
                                [f"• {job} ({count})" for job, count in top_jobs]
                            )
                            if top_jobs
                            else "N/A"
                        )

                        # Format queue time
                        avg_queue = alert.get("avg_queue_time_seconds", 0)
                        avg_queue_str = (
                            f"{avg_queue / 60:.1f}m" if avg_queue > 0 else "N/A"
                        )

                        first_failure = alert.get("first_failure")
                        if first_failure:
                            first_failure_str = f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                        else:
                            first_failure_str = "N/A"

                        last_failure = alert.get("last_failure")
                        if last_failure:
                            last_failure_str = f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                        else:
                            last_failure_str = "N/A"

                        # Format top errors as bullet list
                        top_errors = alert.get("top_error_signatures", [])
                        if top_errors:
                            error_str = "<br>".join(
                                [f"• {err[0]} ({err[1]})" for err in top_errors]
                            )
                        else:
                            error_str = "N/A"

                        summary_lines.append(
                            f"| `{runner_name}` | {alert['current_streak']} | {alert['max_streak']} | "
                            f"{alert['failure_rate']:.1f}% | {avg_queue_str} | {first_failure_str} | {last_failure_str} | "
                            f"{error_str} | {jobs_str} |"
                        )

                    summary_lines.append("")
                    summary_lines.append("")
                else:
                    summary_lines.append("## Alerts: Runners with Issues (streak >= 2)")
                    summary_lines.append("")
                    summary_lines.append(
                        "Nothing to display (no runners with consecutive failure streak >= 2)"
                    )
                    summary_lines.append("")
                    summary_lines.append("")

            # Main Branch Health Section: Jobs failing on cron-triggered main branch runs
            if report_data.get("main_streak_data"):
                # Sort by current streak (descending)
                sorted_main_jobs = sorted(
                    report_data["main_streak_data"].items(),
                    key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                    reverse=True,
                )

                # Show only jobs with streak >= 2
                broken_main_jobs = [
                    (name, data)
                    for name, data in sorted_main_jobs
                    if data["current_streak"] >= 2
                ]

                if broken_main_jobs:
                    summary_lines.append(
                        f"## Main Branch Health: Consecutive Failing Jobs on Scheduled Main Branch Runs (streak >= 2)"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Job Name | Streak | Max | First Failure | Last Failure | Top Errors |"
                    )
                    summary_lines.append(
                        "|----------|--------|-----|---------------|--------------|------------|"
                    )
                    for job_name, data in broken_main_jobs[:15]:
                        display_name = (
                            job_name if len(job_name) <= 35 else job_name[:32] + "..."
                        )

                        # Get first and last failure info
                        first_failure = data.get("first_failure_in_streak")
                        if first_failure:
                            first_failure_str = f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                        else:
                            first_failure_str = "N/A"

                        last_failure = data.get("last_failure_in_streak")
                        if last_failure:
                            last_failure_str = f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                        else:
                            last_failure_str = "N/A"

                        # Format top errors as bullet list
                        top_errors = data.get("top_error_signatures", [])
                        if top_errors:
                            error_str = "<br>".join(
                                [f"• {err[0]} ({err[1]})" for err in top_errors]
                            )
                        else:
                            error_str = "N/A"

                        summary_lines.append(
                            f"| `{display_name}` | {data['current_streak']} | {data['max_streak']} | "
                            f"{first_failure_str} | {last_failure_str} | {error_str} |"
                        )

                    summary_lines.append("")
                else:
                    summary_lines.append(
                        "## Main Branch Health: Scheduled Main Branch Runs"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "No consecutive failing jobs (streak >= 2) on main branch scheduled runs"
                    )
                    summary_lines.append("")

            # Section 1: Currently Broken Jobs - Only show if there are broken jobs
            sorted_jobs = sorted(
                report_data["job_streak_data"].items(),
                key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                reverse=True,
            )

            # Only show jobs with streak >= 2
            broken_jobs = [
                (name, data)
                for name, data in sorted_jobs
                if data["current_streak"] >= 2
            ]

            if broken_jobs:
                summary_lines.append(
                    "## Section 1: Top 15 Consecutively Failing Jobs (PR + Scheduled, streak >= 2)"
                )
                summary_lines.append("")
                summary_lines.append(
                    "| Job Name | Streak | Max | First Failure | Last Failure | Top Errors |"
                )
                summary_lines.append(
                    "|----------|--------|-----|---------------|--------------|------------|"
                )
                for job_name, data in broken_jobs[:20]:
                    display_name = (
                        job_name if len(job_name) <= 35 else job_name[:32] + "..."
                    )

                    # Get first and last failure info
                    first_failure = data.get("first_failure_in_streak")
                    if first_failure:
                        first_failure_str = f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                    else:
                        first_failure_str = "N/A"

                    last_failure = data.get("last_failure_in_streak")
                    if last_failure:
                        last_failure_str = f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                    else:
                        last_failure_str = "N/A"

                    # Format top errors as bullet list
                    top_errors = data.get("top_error_signatures", [])
                    if top_errors:
                        error_str = "<br>".join(
                            [f"• {err[0]} ({err[1]})" for err in top_errors]
                        )
                    else:
                        error_str = "N/A"

                    summary_lines.append(
                        f"| `{display_name}` | {data['current_streak']} | {data['max_streak']} | "
                        f"{first_failure_str} | {last_failure_str} | {error_str} |"
                    )

                summary_lines.append("")

            # Section 2: Runner Health Analysis - Use machine names from runner instances
            if report_data.get("runner_instance_data") and report_data.get(
                "runner_instance_streak_data"
            ):
                # Combine instance stats with streak data and sort by consecutive failures first
                combined_data = []
                for instance_key, stats in report_data["runner_instance_data"].items():
                    streak_data = report_data["runner_instance_streak_data"].get(
                        instance_key, {}
                    )
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
                            "top_error_signatures": streak_data.get(
                                "top_error_signatures", []
                            ),
                        }
                    )

                # Sort by current streak (descending), then max streak, then failure rate
                sorted_runners = sorted(
                    combined_data,
                    key=lambda x: (
                        x["current_streak"],
                        x["max_streak"],
                        x["failure_rate"],
                    ),
                    reverse=True,
                )

                # Only show runners with streak >= 2
                runners_with_issues = [
                    r for r in sorted_runners if r["current_streak"] >= 2
                ]

                if runners_with_issues:
                    summary_lines.append(
                        "## Section 2: Top 15 Consecutively Failing Workers (streak >= 2)"
                    )
                    summary_lines.append("")
                    summary_lines.append(
                        "| Machine Name | Streak | Max | Fail Rate | Avg Queue | First Failure | Last Failure | Top Errors | Total Jobs | Unique Jobs |"
                    )
                    summary_lines.append(
                        "|--------------|--------|-----|-----------|-----------|---------------|--------------|------------|------------|-------------|"
                    )

                    for runner_data in runners_with_issues[:15]:
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
                        if first_failure:
                            first_failure_str = f"[Run #{first_failure['run_number']}]({first_failure.get('job_url', first_failure['url'])})"
                        else:
                            first_failure_str = "N/A"

                        last_failure = runner_data.get("last_failure")
                        if last_failure:
                            last_failure_str = f"[Run #{last_failure['run_number']}]({last_failure.get('job_url', last_failure['url'])})"
                        else:
                            last_failure_str = "N/A"

                        # Format top errors as bullet list
                        top_errors = runner_data.get("top_error_signatures", [])
                        if top_errors:
                            error_str = "<br>".join(
                                [f"• {err[0]} ({err[1]})" for err in top_errors]
                            )
                        else:
                            error_str = "N/A"

                        summary_lines.append(
                            f"| `{display_name}` | {streak_str} | {max_str} | {runner_data['failure_rate']:.1f}% | "
                            f"{avg_queue_str} | {first_failure_str} | {last_failure_str} | {error_str} | "
                            f"{runner_data['total_jobs']} | {runner_data['unique_jobs']} |"
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
        default=1000,
        help="Number of workflow runs to analyze across all monitored workflows (default: 1000)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Alert threshold for consecutive failures (default: 3)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (optional, only writes if specified)",
    )

    args = parser.parse_args()

    analyzer = SGLangFailuresAnalyzer(args.token, alert_threshold=args.threshold)

    try:
        # Fetch recent runs
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No workflow runs found")
            return

        # Analyze consecutive failures
        job_streak_data, job_current_streaks = analyzer.analyze_consecutive_failures(
            runs
        )

        if not job_streak_data:
            print("No job data found")
            return

        # Skip aggregation to show individual job shards
        print(f"\nTotal jobs (including shards): {len(job_streak_data)}")

        # Analyze consecutive failures on main branch (cron-triggered only)
        main_streak_data, main_current_streaks = (
            analyzer.construct_cron_failures_on_main(runs, job_streak_data)
        )

        # Analyze runner health and consecutive failures
        (
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        ) = analyzer.analyze_runner_health(runs)

        # Detect alerts
        job_alerts, runner_alerts = analyzer.detect_alerts(
            job_streak_data,
            job_current_streaks,
            runner_stats,
            runner_instance_data,
            runner_streak_data,
            runner_instance_streak_data,
        )

        # Generate report
        report_data = analyzer.generate_failure_report(
            job_streak_data,
            job_alerts,
            runner_stats,
            runner_instance_data,
            runner_alerts,
            runner_streak_data,
            runner_instance_streak_data,
            main_streak_data,
            args.output,
        )

        # Generate GitHub Actions summary
        analyzer.generate_github_summary(report_data)

        # Exit with error code if alerts triggered
        total_alerts = len(job_alerts) + len(runner_alerts)
        if total_alerts > 0:
            print(
                f"\n!!!!! {len(job_alerts)} job alert(s) and {len(runner_alerts)} runner alert(s) triggered!"
            )
            sys.exit(0)  # Don't fail the workflow, just report
        else:
            print("\n No alerts triggered")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
