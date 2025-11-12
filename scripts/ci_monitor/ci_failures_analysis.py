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
        self.target_workflows = ["PR Test"]

        # Jobs to EXCLUDE from analysis (administrative/setup jobs, not actual tests)
        self.excluded_jobs = [
            "check-changes",
            "pr-test-finish",
        ]

    def get_recent_runs(self, limit: int = 500) -> List[Dict]:
        """Fetch recent workflow runs from GitHub API."""
        print(f"Fetching {limit} recent workflow runs...")

        all_runs = []
        page = 1
        per_page = 100

        while len(all_runs) < limit:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": min(per_page, limit - len(all_runs)), "page": page}

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                all_runs.extend(data["workflow_runs"])
                print(f"Fetched {len(all_runs)} runs so far...")

                if len(data["workflow_runs"]) < per_page:
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching workflow runs: {e}")
                break

        # Filter to target workflows only
        filtered_runs = [
            run
            for run in all_runs
            if run.get("name") in self.target_workflows
            and run.get("status") == "completed"
        ]

        print(f"Filtered to {len(filtered_runs)} completed target workflow runs")
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
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Analyze runner health by tracking failures per runner.

        Returns:
            Tuple of (runner_stats, runner_job_failures)
            - runner_stats: Overall stats per runner (failure rate, total jobs, etc.)
            - runner_job_failures: Per-runner breakdown of which jobs failed
        """
        print("\nAnalyzing runner health...")

        # Sort runs by created_at (oldest first)
        sorted_runs = sorted(runs, key=lambda x: x.get("created_at", ""))

        # Track runner statistics
        runner_total_jobs: Dict[str, int] = defaultdict(int)
        runner_failed_jobs: Dict[str, int] = defaultdict(int)
        runner_job_failures: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        runner_job_totals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Track individual runner instances (runner_name + runner_id)
        runner_instance_stats: Dict[str, Dict] = defaultdict(
            lambda: {"total_jobs": 0, "failed_jobs": 0, "jobs_failed": defaultdict(int)}
        )

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed} for runner analysis: #{run.get('run_number')}"
                )

            # Get jobs for this run
            jobs = self.get_jobs_for_run(run.get("id"))

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

                conclusion = job.get("conclusion")

                if conclusion == "failure":
                    # Failure detected
                    runner_failed_jobs[runner_key] += 1
                    runner_job_failures[runner_key][job_name] += 1

                    if runner_id:
                        runner_instance_stats[runner_instance_key]["failed_jobs"] += 1
                        runner_instance_stats[runner_instance_key]["jobs_failed"][
                            job_name
                        ] += 1

            time.sleep(0.05)

        # Build final runner stats
        runner_stats = {}
        for runner_key in runner_total_jobs.keys():
            total = runner_total_jobs[runner_key]
            failed = runner_failed_jobs[runner_key]
            failure_rate = (failed / total * 100) if total > 0 else 0

            runner_stats[runner_key] = {
                "total_jobs": total,
                "failed_jobs": failed,
                "failure_rate": failure_rate,
                "unique_jobs_with_failures": len(runner_job_failures[runner_key]),
                "jobs_failed": dict(runner_job_failures[runner_key]),
                "jobs_total": dict(runner_job_totals[runner_key]),
            }

        # Convert runner instance stats to regular dicts
        runner_instance_data = {}
        for instance_key, stats in runner_instance_stats.items():
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
            }

        return runner_stats, runner_instance_data

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
        job_streaks: Dict[str, List[Dict]] = defaultdict(list)
        job_current_streak: Dict[str, int] = defaultdict(int)
        job_max_streak: Dict[str, int] = defaultdict(int)
        job_total_failures: Dict[str, int] = defaultdict(int)
        job_total_runs: Dict[str, int] = defaultdict(int)
        job_first_failure_in_streak: Dict[str, Optional[Dict]] = {}
        job_recovery_info: Dict[str, Optional[Dict]] = {}

        total_runs_processed = len(sorted_runs)
        for i, run in enumerate(sorted_runs, 1):
            if i % 50 == 0 or i == total_runs_processed:
                print(
                    f"Processing run {i}/{total_runs_processed}: #{run.get('run_number')}"
                )

            run_info = {
                "run_number": run.get("run_number"),
                "run_id": run.get("id"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": run.get("head_commit", {})
                .get("author", {})
                .get("name", "Unknown"),
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

            time.sleep(0.05)

        # Build final results
        job_streak_data = {}
        for job_name in job_current_streak.keys():
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
                "recovery_info": job_recovery_info.get(job_name),
            }

        return job_streak_data, job_current_streak

    def aggregate_matrix_jobs(
        self, job_streak_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Aggregate matrix jobs (e.g., 'job-name (0)', 'job-name (1)') into a single entry.

        Returns:
            Dictionary with aggregated job data
        """
        import re

        # Identify base job names (strip matrix suffix like " (0)", " (1)")
        base_jobs: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

        for job_name, data in job_streak_data.items():
            # Match pattern like "job-name (0)" or "job-name (1)"
            match = re.match(r"^(.+?)\s*\((\d+)\)$", job_name)
            if match:
                base_name = match.group(1)
                base_jobs[base_name].append((job_name, data))
            else:
                # Not a matrix job, keep as-is
                base_jobs[job_name].append((job_name, data))

        # Aggregate stats for matrix jobs
        aggregated_data = {}

        for base_name, job_list in base_jobs.items():
            if len(job_list) == 1:
                # Single job, no aggregation needed
                job_name, data = job_list[0]
                aggregated_data[job_name] = data
            else:
                # Multiple matrix jobs - aggregate them
                total_runs = sum(data["total_runs"] for _, data in job_list)
                total_failures = sum(data["total_failures"] for _, data in job_list)

                # Current streak: take the max across all matrix jobs
                # (if any partition is broken, the whole job is considered broken)
                current_streak = max(data["current_streak"] for _, data in job_list)
                max_streak = max(data["max_streak"] for _, data in job_list)

                # Get the first failure from the job with the longest current streak
                first_failure_in_streak = None
                for _, data in job_list:
                    if (
                        data["current_streak"] == current_streak
                        and data["first_failure_in_streak"]
                    ):
                        first_failure_in_streak = data["first_failure_in_streak"]
                        break

                # Recovery info from most recent recovery
                recovery_info = None
                for _, data in job_list:
                    if data["recovery_info"]:
                        recovery_info = data["recovery_info"]
                        break

                aggregated_data[base_name] = {
                    "current_streak": current_streak,
                    "max_streak": max_streak,
                    "total_failures": total_failures,
                    "total_runs": total_runs,
                    "failure_rate": (
                        (total_failures / total_runs * 100) if total_runs > 0 else 0
                    ),
                    "first_failure_in_streak": first_failure_in_streak,
                    "recovery_info": recovery_info,
                    "is_aggregated": True,
                    "partition_count": len(job_list),
                    "partitions": [job_name for job_name, _ in job_list],
                }

        return aggregated_data

    def detect_alerts(
        self,
        job_streak_data: Dict[str, Dict],
        job_current_streaks: Dict[str, int],
        runner_stats: Optional[Dict[str, Dict]] = None,
        runner_instance_data: Optional[Dict[str, Dict]] = None,
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
                        "alert_type": "consecutive_failures",
                        "severity": "high" if current_streak >= 5 else "medium",
                    }
                )

        # Detect runner alerts
        runner_alerts = []
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

        print(
            f"\nTotal (unique) jobs analyzed across PR Test workflows: {len(sorted_jobs)}"
        )
        print(
            f"Jobs with active failure streaks: {sum(1 for j in sorted_jobs if j[1]['current_streak'] > 0)}"
        )
        print(
            f"Job alerts triggered (>={self.alert_threshold} consecutive failures): {len(job_alerts)}"
        )

        if runner_stats:
            print(f"Total runners analyzed: {len(runner_stats)}")
            print(
                f"Runner alerts triggered: {len(runner_alerts) if runner_alerts else 0}"
            )

        # Section 1: Currently Broken Jobs (Consecutive Failures) - URGENT
        print("\n" + "=" * 100)
        print("SECTION 1: Currently Broken Jobs (Active Consecutive Failures)")
        print("=" * 100)

        broken_jobs = [
            (name, data) for name, data in sorted_jobs if data["current_streak"] > 0
        ]

        if broken_jobs:
            print(
                f"\n{'Rank':<4} {'Job Name':<50} {'Current Streak':<16} {'Max Streak':<12}"
            )
            print("-" * 100)
            for i, (job_name, data) in enumerate(broken_jobs[:20], 1):
                print(
                    f"{i:<4} {job_name:<50} {data['current_streak']:<16} {data['max_streak']:<12}"
                )
        else:
            print("\n✓ No jobs are currently in a failure streak!")

        # Print job alerts
        if job_alerts:
            print("\n" + "!" * 40)
            print("ALERTS: Jobs with Consecutive Failures Exceeding Threshold")
            print("!" * 40)

            for alert in sorted(
                job_alerts, key=lambda x: x["current_streak"], reverse=True
            ):
                print(f"\n  {alert['job_name']}")
                print(
                    f"   Current Streak: {alert['current_streak']} consecutive failures"
                )
                print(f"   Max Streak: {alert['max_streak']}")
                print(f"   Severity: {alert['severity'].upper()}")

                if alert["first_failure"]:
                    first = alert["first_failure"]
                    print(
                        f"   First Failure in Streak: Run #{first['run_number']} ({first['created_at']})"
                    )
                    print(f"   Link: {first['url']}")

        # Section 3: Runner Health Analysis
        if runner_stats:
            print("\n" + "=" * 100)
            print("SECTION 2: Runner Health Analysis")
            print("=" * 100)

            # Sort runners by failure rate
            sorted_runners = sorted(
                runner_stats.items(),
                key=lambda x: (x[1]["failure_rate"], x[1]["failed_jobs"]),
                reverse=True,
            )

            print(f"\nTop 15 Runners by Failure Rate:")
            print("-" * 100)
            print(
                f"{'Rank':<4} {'Runner Labels':<45} {'Fail Rate':<12} {'Failed':<10} {'Total':<10} {'Unique Jobs':<12}"
            )
            print("-" * 100)

            for i, (runner_labels, stats) in enumerate(sorted_runners[:15], 1):
                # Truncate labels if too long for display
                display_labels = (
                    runner_labels
                    if len(runner_labels) <= 43
                    else runner_labels[:40] + "..."
                )
                print(
                    f"{i:<4} {display_labels:<45} {stats['failure_rate']:>10.1f}% "
                    f"{stats['failed_jobs']:<10} {stats['total_jobs']:<10} {stats['unique_jobs_with_failures']:<12}"
                )

        # Print runner alerts
        if runner_alerts:
            print("\n" + "!" * 40)
            print("ALERTS: Runners with High Failure Rates")
            print("!" * 40)

            for alert in sorted(
                runner_alerts, key=lambda x: x.get("failure_rate", 0), reverse=True
            ):
                if alert["alert_type"] == "runner_health":
                    print(f"\n  Runner Labels: {alert['runner_labels']}")
                    print(f"   Failure Rate: {alert['failure_rate']:.1f}%")
                    print(
                        f"   Failed Jobs: {alert['failed_jobs']} / {alert['total_jobs']}"
                    )
                    print(
                        f"   Unique Jobs with Failures: {alert['unique_jobs_with_failures']}"
                    )
                    print(f"   Severity: {alert['severity'].upper()}")
                elif alert["alert_type"] == "runner_instance_health":
                    print(f"\n  Runner Instance: {alert['runner_instance']}")
                    print(f"   Runner Name: {alert['runner_name']}")
                    print(f"   Failure Rate: {alert['failure_rate']:.1f}%")
                    print(
                        f"   Failed Jobs: {alert['failed_jobs']} / {alert['total_jobs']}"
                    )
                    print(f"   Jobs Failed: {list(alert['jobs_failed'].keys())}")
                    print(f"   Severity: {alert['severity'].upper()}")

        # Build report data (always needed for GitHub summary)
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
            "runner_alerts": runner_alerts if runner_alerts else [],
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
            summary_lines.append(
                f"| Total Runners Analyzed | {report_data['summary']['total_runners']} |"
            )
            summary_lines.append(
                f"| Runner Alerts Triggered | {report_data['summary']['runner_alerts_triggered']} |"
            )
            summary_lines.append("")

            # Job Alerts section
            if report_data.get("job_alerts"):
                summary_lines.append("## ALERTS: Critical Consecutive Job Failures")
                summary_lines.append("")
                summary_lines.append(
                    "| Job Name | Current Streak | Max Streak | First Failure | Link |"
                )
                summary_lines.append(
                    "|----------|----------------|------------|---------------|------|"
                )

                for alert in sorted(
                    report_data["job_alerts"],
                    key=lambda x: x["current_streak"],
                    reverse=True,
                ):
                    job_name = alert["job_name"]
                    if len(job_name) > 40:
                        job_name = job_name[:37] + "..."

                    first_failure = alert.get("first_failure")
                    first_failure_str = (
                        f"Run #{first_failure['run_number']}"
                        if first_failure
                        else "N/A"
                    )
                    first_failure_link = first_failure["url"] if first_failure else ""

                    summary_lines.append(
                        f"| `{job_name}` | {alert['current_streak']} | {alert['max_streak']} | "
                        f"{first_failure_str} | [View]({first_failure_link}) |"
                    )

                summary_lines.append("")

            # Runner Alerts section
            if report_data.get("runner_alerts"):
                summary_lines.append("## ALERTS: Runners with High Failure Rates")
                summary_lines.append("")
                summary_lines.append(
                    "| Runner Labels | Failure Rate | Failed Jobs | Total Jobs | Unique Jobs Failed | Severity |"
                )
                summary_lines.append(
                    "|---------------|--------------|-------------|------------|-------------------|----------|"
                )

                for alert in sorted(
                    report_data["runner_alerts"],
                    key=lambda x: x.get("failure_rate", 0),
                    reverse=True,
                ):
                    if alert["alert_type"] == "runner_health":
                        runner_labels = alert["runner_labels"]
                        if len(runner_labels) > 35:
                            runner_labels = runner_labels[:32] + "..."

                        summary_lines.append(
                            f"| `{runner_labels}` | {alert['failure_rate']:.1f}% | {alert['failed_jobs']} | "
                            f"{alert['total_jobs']} | {alert['unique_jobs_with_failures']} | {alert['severity'].upper()} |"
                        )
                    elif alert["alert_type"] == "runner_instance_health":
                        instance = alert["runner_instance"]
                        runner_name = alert["runner_name"]
                        if len(instance) > 35:
                            instance = instance[:32] + "..."

                        summary_lines.append(
                            f"| `{instance}` | {alert['failure_rate']:.1f}% | {alert['failed_jobs']} | "
                            f"{alert['total_jobs']} | {len(alert['jobs_failed'])} | {alert['severity'].upper()} |"
                        )
                        summary_lines.append(f"| (Runner: {runner_name}) | | | | | |")

                summary_lines.append("")

            # Section 1: Currently Broken Jobs
            summary_lines.append(
                "## Section 1: Currently Broken Jobs (Active Failures)"
            )
            summary_lines.append("")

            sorted_jobs = sorted(
                report_data["job_streak_data"].items(),
                key=lambda x: (x[1]["current_streak"], x[1]["failure_rate"]),
                reverse=True,
            )

            broken_jobs = [
                (name, data) for name, data in sorted_jobs if data["current_streak"] > 0
            ]

            if broken_jobs:
                summary_lines.append(
                    "| Rank | Job Name | Current Streak | Max Streak |"
                )
                summary_lines.append(
                    "|------|----------|----------------|------------|"
                )
                for i, (job_name, data) in enumerate(broken_jobs[:20], 1):
                    display_name = (
                        job_name if len(job_name) <= 40 else job_name[:37] + "..."
                    )
                    summary_lines.append(
                        f"| {i} | `{display_name}` | {data['current_streak']} | {data['max_streak']} |"
                    )
            else:
                summary_lines.append("No jobs are currently in a failure streak!")

            summary_lines.append("")

            # Section 2: Runner Health Analysis
            if report_data.get("runner_stats"):
                summary_lines.append("## Section 2: Runner Health Analysis")
                summary_lines.append("")

                # Sort runners by failure rate
                sorted_runners = sorted(
                    report_data["runner_stats"].items(),
                    key=lambda x: (x[1]["failure_rate"], x[1]["failed_jobs"]),
                    reverse=True,
                )

                summary_lines.append("### Top 15 Runners by Failure Rate")
                summary_lines.append("")
                summary_lines.append(
                    "| Rank | Runner Labels | Failure Rate | Failed Jobs | Total Jobs | Unique Jobs Failed |"
                )
                summary_lines.append(
                    "|------|---------------|--------------|-------------|------------|--------------------|"
                )

                for i, (runner_labels, stats) in enumerate(sorted_runners[:15], 1):
                    display_labels = (
                        runner_labels
                        if len(runner_labels) <= 35
                        else runner_labels[:32] + "..."
                    )
                    summary_lines.append(
                        f"| {i} | `{display_labels}` | {stats['failure_rate']:.1f}% | "
                        f"{stats['failed_jobs']} | {stats['total_jobs']} | {stats['unique_jobs_with_failures']} |"
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
        default=500,
        help="Number of workflow runs to analyze (default: 500)",
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

        # Aggregate matrix jobs (e.g., "job (0)", "job (1)" -> "job")
        print("\nAggregating matrix jobs...")
        job_streak_data = analyzer.aggregate_matrix_jobs(job_streak_data)
        print(f"After aggregation: {len(job_streak_data)} unique jobs")

        # Analyze runner health
        runner_stats, runner_instance_data = analyzer.analyze_runner_health(runs)

        # Detect alerts
        job_alerts, runner_alerts = analyzer.detect_alerts(
            job_streak_data, job_current_streaks, runner_stats, runner_instance_data
        )

        # Generate report
        report_data = analyzer.generate_failure_report(
            job_streak_data,
            job_alerts,
            runner_stats,
            runner_instance_data,
            runner_alerts,
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
