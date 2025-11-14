#!/usr/bin/env python3
"""
Nightly Test Monitor

Monitors nightly test runs for performance and accuracy regressions.
Analyzes metrics from GitHub summaries and tracks trends over time.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests


class NightlyTestMonitor:
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Nightly-Monitor/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Nightly test jobs to monitor
        self.nightly_jobs = [
            "nightly-test-eval-text-models",
            "nightly-test-perf-text-models",
            "nightly-test-eval-vlms",
            "nightly-test-perf-vlms",
            "nightly-test-1-gpu",
            "nightly-test-4-gpu",
            "nightly-test-8-gpu-h200",
            "nightly-test-8-gpu-h20",
            "nightly-test-4-gpu-b200",
            "nightly-test-8-gpu-b200",
        ]

        # Performance metric patterns for parsing logs
        self.perf_patterns = {
            "output_throughput": re.compile(
                r"Output token throughput \(tok/s\):\s*([\d.]+)"
            ),
            "input_throughput": re.compile(
                r"Input token throughput \(tok/s\):\s*([\d.]+)"
            ),
            "latency": re.compile(r"Median E2E Latency \(ms\):\s*([\d.]+)"),
            "ttft": re.compile(r"Median TTFT \(ms\):\s*([\d.]+)"),
            "accept_length": re.compile(r"Accept length:\s*([\d.]+)"),
            "accuracy": re.compile(r"Accuracy:\s*([\d.]+)"),
            "gsm8k_score": re.compile(r"GSM8K Score:\s*([\d.]+)"),
        }

        # Historical data repository
        self.data_repo = "sglang-bot/sglang-ci-data"
        self.data_branch = "main"

    def get_nightly_runs(self, days: int = 7) -> List[Dict]:
        """Get nightly test workflow runs from the last N days"""
        print(f"Fetching nightly test runs from the last {days} days...")

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        all_runs = []
        page = 1
        per_page = 100

        while True:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {
                "workflow_id": "nightly-test.yml",
                "per_page": per_page,
                "page": page,
                "created": f">={since_date}",
            }

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                runs = data["workflow_runs"]
                all_runs.extend(runs)
                print(f"Fetched {len(all_runs)} nightly runs so far...")

                if len(runs) < per_page:
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching nightly test data: {e}")
                break

        print(f"Total nightly runs fetched: {len(all_runs)}")
        return all_runs

    def get_job_details(self, run_id: int) -> List[Dict]:
        """Get job details for a specific run"""
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except:
            return []

    def get_job_logs(self, job_id: int) -> Optional[str]:
        """Get logs for a specific job"""
        url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{job_id}/logs"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Could not fetch logs for job {job_id}: {e}")
            return None

    def parse_metrics_from_logs(
        self, logs: str, job_name: str
    ) -> Dict[str, List[float]]:
        """
        Parse performance metrics from job logs.

        Args:
            logs: Raw log text from the job
            job_name: Name of the job (to determine which metrics to look for)

        Returns:
            Dictionary mapping metric names to lists of values found
        """
        metrics = defaultdict(list)

        if not logs:
            return metrics

        # Parse each line for matching patterns
        for line in logs.split("\n"):
            for metric_name, pattern in self.perf_patterns.items():
                match = pattern.search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        metrics[metric_name].append(value)
                    except (ValueError, IndexError):
                        continue

        return dict(metrics)

    def get_historical_data_paths(self) -> List[str]:
        """
        Get list of available nightly monitor data files from the data repository.

        Returns:
            List of file paths in the repository
        """
        url = f"{self.base_url}/repos/{self.data_repo}/contents/nightly_monitor"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            contents = response.json()

            # Filter for JSON files
            json_files = [
                item["path"]
                for item in contents
                if item["type"] == "file" and item["name"].endswith(".json")
            ]
            return sorted(json_files, reverse=True)  # Most recent first
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch historical data paths: {e}")
            return []

    def fetch_historical_data(self, file_path: str) -> Optional[Dict]:
        """
        Fetch a specific historical data file from the repository.

        Args:
            file_path: Path to the file in the repository

        Returns:
            Dictionary with historical data, or None if fetch failed
        """
        url = f"{self.base_url}/repos/{self.data_repo}/contents/{file_path}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            # Decode base64 content
            content = base64.b64decode(data["content"]).decode("utf-8")
            return json.loads(content)
        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
        ) as e:
            print(f"Warning: Could not fetch historical data from {file_path}: {e}")
            return None

    def get_recent_historical_metrics(
        self, job_name: str, metric_name: str, days: int = 7
    ) -> List[Dict]:
        """
        Get recent historical metrics for a specific job and metric.

        Args:
            job_name: Name of the job
            metric_name: Name of the metric (e.g., 'output_throughput')
            days: Number of days to look back

        Returns:
            List of metric data points with timestamps
        """
        print(
            f"  Fetching historical {metric_name} data for {job_name} (last {days} days)..."
        )

        historical_paths = self.get_historical_data_paths()
        if not historical_paths:
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        historical_metrics = []

        # Fetch recent files (limit to avoid too many API calls)
        for file_path in historical_paths[: min(days * 2, 14)]:  # Max 14 files
            historical_data = self.fetch_historical_data(file_path)
            if not historical_data:
                continue

            # Check if this file has data for our job
            job_stats = historical_data.get("job_stats", {}).get(job_name, {})
            if not job_stats:
                continue

            # Extract metrics
            perf_metrics = job_stats.get("performance_metrics", {}).get(metric_name, [])
            for metric_entry in perf_metrics:
                try:
                    timestamp = datetime.fromisoformat(
                        metric_entry["timestamp"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if timestamp >= cutoff_date:
                        historical_metrics.append(metric_entry)
                except (ValueError, KeyError):
                    continue

        return sorted(historical_metrics, key=lambda x: x["timestamp"])

    def compare_with_historical(
        self, current_metrics: Dict[str, List[Dict]], days: int = 7
    ) -> Dict[str, Dict]:
        """
        Compare current metrics with historical data to detect changes.

        Args:
            current_metrics: Dictionary of metric_name -> list of metric data points
            days: Number of days to look back for comparison

        Returns:
            Dictionary with comparison results including percentage changes
        """
        comparisons = {}

        for metric_name, current_data in current_metrics.items():
            if not current_data:
                continue

            # Calculate current average
            current_values = [d["value"] for d in current_data]
            current_avg = sum(current_values) / len(current_values)

            # Get the job name from the first data point
            # (assumes all data points are from the same job)
            job_name = current_data[0].get("job_name", "unknown")

            # Fetch historical data
            historical_data = self.get_recent_historical_metrics(
                job_name, metric_name, days
            )

            if not historical_data:
                comparisons[metric_name] = {
                    "current_avg": current_avg,
                    "historical_avg": None,
                    "percent_change": None,
                    "status": "no_history",
                }
                continue

            # Calculate historical average
            historical_values = [d["value"] for d in historical_data]
            historical_avg = sum(historical_values) / len(historical_values)

            # Calculate percentage change
            if historical_avg > 0:
                percent_change = ((current_avg - historical_avg) / historical_avg) * 100
            else:
                percent_change = 0

            # Determine status based on change
            if abs(percent_change) < 5:
                status = "stable"
            elif abs(percent_change) < 10:
                status = "minor_change"
            else:
                status = "significant_change"

            comparisons[metric_name] = {
                "current_avg": current_avg,
                "historical_avg": historical_avg,
                "percent_change": percent_change,
                "status": status,
                "current_count": len(current_values),
                "historical_count": len(historical_values),
            }

        return comparisons

    def analyze_nightly_tests(self, runs: List[Dict]) -> Dict:
        """Analyze nightly test runs for failures and performance"""
        print("Analyzing nightly test data...")

        stats = {
            "total_runs": len(runs),
            "successful_runs": 0,
            "failed_runs": 0,
            "cancelled_runs": 0,
            "job_stats": defaultdict(
                lambda: {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "recent_failures": [],
                    "avg_duration_minutes": 0,
                    "durations": [],
                    "performance_metrics": defaultdict(list),  # New: track perf metrics
                }
            ),
            "daily_stats": defaultdict(
                lambda: {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                }
            ),
        }

        for i, run in enumerate(runs, 1):
            if i % 10 == 0:
                print(f"Processed {i}/{len(runs)} runs...")

            run_status = run.get("conclusion", "unknown")
            run_id = run.get("id")
            run_number = run.get("run_number")
            created_at = run.get("created_at")
            run_url = f"https://github.com/{self.repo}/actions/runs/{run_id}"

            # Track daily stats
            date_str = created_at.split("T")[0] if created_at else "unknown"
            stats["daily_stats"][date_str]["total"] += 1

            if run_status == "success":
                stats["successful_runs"] += 1
                stats["daily_stats"][date_str]["success"] += 1
            elif run_status == "failure":
                stats["failed_runs"] += 1
                stats["daily_stats"][date_str]["failure"] += 1
            elif run_status == "cancelled":
                stats["cancelled_runs"] += 1

            # Analyze individual jobs
            jobs = self.get_job_details(run_id)
            for job in jobs:
                job_name = job.get("name", "Unknown")
                job_conclusion = job.get("conclusion", "unknown")
                job_id = job.get("id")
                started_at = job.get("started_at")
                completed_at = job.get("completed_at")

                # Only track our nightly test jobs
                if job_name not in self.nightly_jobs:
                    continue

                job_stat = stats["job_stats"][job_name]
                job_stat["total"] += 1

                if job_conclusion == "success":
                    job_stat["success"] += 1

                    # For successful performance jobs, fetch metrics
                    if "perf" in job_name.lower() or "eval" in job_name.lower():
                        logs = self.get_job_logs(job_id)
                        if logs:
                            metrics = self.parse_metrics_from_logs(logs, job_name)
                            # Store metrics with timestamp and job name
                            for metric_name, values in metrics.items():
                                if values:  # Only store if we found values
                                    job_stat["performance_metrics"][metric_name].extend(
                                        [
                                            {
                                                "value": v,
                                                "timestamp": created_at,
                                                "run_id": run_id,
                                                "job_name": job_name,
                                            }
                                            for v in values
                                        ]
                                    )

                elif job_conclusion == "failure":
                    job_stat["failure"] += 1

                    # Store recent failures (up to 5)
                    if len(job_stat["recent_failures"]) < 5:
                        job_stat["recent_failures"].append(
                            {
                                "run_url": run_url,
                                "run_number": run_number,
                                "created_at": created_at,
                                "job_url": job.get("html_url"),
                            }
                        )

                # Track duration
                if started_at and completed_at:
                    try:
                        start = datetime.fromisoformat(
                            started_at.replace("Z", "+00:00")
                        )
                        end = datetime.fromisoformat(
                            completed_at.replace("Z", "+00:00")
                        )
                        duration_minutes = (end - start).total_seconds() / 60
                        job_stat["durations"].append(duration_minutes)
                    except:
                        pass

            time.sleep(0.1)

        # Calculate average durations
        for job_name, job_stat in stats["job_stats"].items():
            if job_stat["durations"]:
                job_stat["avg_duration_minutes"] = sum(job_stat["durations"]) / len(
                    job_stat["durations"]
                )
                del job_stat["durations"]  # Remove raw data to reduce size

        return stats

    def generate_github_summary(self, stats: Dict, regressions: List[Dict]):
        """Generate GitHub Actions step summary"""
        github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if not github_summary:
            return

        with open(github_summary, "a") as f:
            f.write("# Nightly Test Monitor Report\n\n")
            f.write(
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Summary stats
            f.write("## Summary\n\n")
            f.write(f"- Total Runs: {stats['total_runs']}\n")
            f.write(
                f"- Successful: {stats['successful_runs']} "
                f"({stats['successful_runs']/max(1, stats['total_runs'])*100:.1f}%)\n"
            )
            f.write(
                f"- Failed: {stats['failed_runs']} "
                f"({stats['failed_runs']/max(1, stats['total_runs'])*100:.1f}%)\n\n"
            )

            # Regressions
            if regressions:
                f.write("## Regressions Detected\n\n")
                for reg in regressions:
                    if reg["type"] == "performance_regression":
                        f.write(
                            f"- **{reg['job_name']}**: {reg['metric_name']} "
                            f"({reg['percent_change']:+.1f}%)\n"
                        )

            # Performance metrics table
            f.write("\n## Performance Metrics\n\n")
            f.write("| Job | Metric | Current | Change |\n")
            f.write("|-----|--------|---------|--------|\n")

            for job_name, job_stat in stats["job_stats"].items():
                if job_stat.get("performance_metrics"):
                    perf_metrics = job_stat["performance_metrics"]
                    comparisons = self.compare_with_historical(perf_metrics, days=7)

                    for metric_name, metric_data in perf_metrics.items():
                        if metric_data:
                            values = [m["value"] for m in metric_data]
                            avg_value = sum(values) / len(values)
                            comparison = comparisons.get(metric_name, {})
                            percent_change = comparison.get("percent_change")

                            if percent_change is not None:
                                f.write(
                                    f"| {job_name} | {metric_name} | {avg_value:.2f} | "
                                    f"{percent_change:+.1f}% |\n"
                                )

    def generate_report(self, stats: Dict, output_file: str = None):
        """Generate a human-readable report"""
        print("\n" + "=" * 80)
        print("NIGHTLY TEST MONITOR REPORT")
        print("=" * 80)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Runs Analyzed: {stats['total_runs']}")
        print(
            f"Successful: {stats['successful_runs']} "
            f"({stats['successful_runs']/max(1, stats['total_runs'])*100:.1f}%)"
        )
        print(
            f"Failed: {stats['failed_runs']} "
            f"({stats['failed_runs']/max(1, stats['total_runs'])*100:.1f}%)"
        )
        print(f"Cancelled: {stats['cancelled_runs']}")
        print("=" * 80)

        # Daily trend
        print("\nDAILY TRENDS:")
        print("-" * 80)
        daily_stats = sorted(stats["daily_stats"].items(), reverse=True)[:7]
        for date, day_stats in daily_stats:
            success_rate = (day_stats["success"] / max(1, day_stats["total"])) * 100
            print(
                f"{date}: {day_stats['total']} runs, {day_stats['success']} success "
                f"({success_rate:.1f}%), {day_stats['failure']} failed"
            )

        # Job statistics
        print("\nJOB STATISTICS:")
        print("-" * 80)
        print(
            f"{'Job Name':<40} {'Total':<8} {'Success':<8} {'Failed':<8} "
            f"{'Rate':<8} {'Avg Duration'}"
        )
        print("-" * 80)

        job_stats_sorted = sorted(
            stats["job_stats"].items(), key=lambda x: x[1]["failure"], reverse=True
        )

        for job_name, job_stat in job_stats_sorted:
            total = job_stat["total"]
            success = job_stat["success"]
            failure = job_stat["failure"]
            success_rate = (success / max(1, total)) * 100
            avg_duration = job_stat["avg_duration_minutes"]

            print(
                f"{job_name:<40} {total:<8} {success:<8} {failure:<8} "
                f"{success_rate:>6.1f}% {avg_duration:>7.1f}m"
            )

            # Show performance metrics with day-to-day comparison if available
            if job_stat.get("performance_metrics"):
                perf_metrics = job_stat["performance_metrics"]
                print(f"  Performance metrics:")

                # Compare with historical data
                comparisons = self.compare_with_historical(perf_metrics, days=7)

                for metric_name, metric_data in perf_metrics.items():
                    if metric_data:
                        values = [m["value"] for m in metric_data]
                        avg_value = sum(values) / len(values)

                        # Get comparison data
                        comparison = comparisons.get(metric_name, {})
                        percent_change = comparison.get("percent_change")

                        if percent_change is not None:
                            change_indicator = "📈" if percent_change > 0 else "📉"
                            if abs(percent_change) < 1:
                                change_indicator = "➡️"

                            print(
                                f"    - {metric_name}: {avg_value:.2f} "
                                f"(n={len(values)}) {change_indicator} "
                                f"{percent_change:+.1f}% vs 7d avg"
                            )
                        else:
                            print(
                                f"    - {metric_name}: {avg_value:.2f} (n={len(values)}) [no history]"
                            )

            # Show recent failures
            if job_stat["recent_failures"]:
                print(f"  Recent failures:")
                for failure in job_stat["recent_failures"][:3]:
                    print(f"    - Run #{failure['run_number']}: {failure['run_url']}")

        print("=" * 80)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"\nDetailed stats saved to: {output_file}")

    def detect_regressions(self, stats: Dict) -> List[Dict]:
        """Detect potential regressions in nightly tests"""
        regressions = []

        for job_name, job_stat in stats["job_stats"].items():
            total = job_stat["total"]
            failure = job_stat["failure"]

            if total > 0:
                failure_rate = (failure / total) * 100

                # Flag jobs with high failure rates
                if failure_rate > 30:
                    regressions.append(
                        {
                            "job_name": job_name,
                            "type": "high_failure_rate",
                            "failure_rate": failure_rate,
                            "total_runs": total,
                            "failures": failure,
                        }
                    )

                # Flag jobs with recent consecutive failures
                recent_failures = len(job_stat["recent_failures"])
                if recent_failures >= 3:
                    regressions.append(
                        {
                            "job_name": job_name,
                            "type": "consecutive_failures",
                            "recent_failure_count": recent_failures,
                        }
                    )

            # Check for performance regressions >10%
            if job_stat.get("performance_metrics"):
                perf_metrics = job_stat["performance_metrics"]
                comparisons = self.compare_with_historical(perf_metrics, days=7)

                for metric_name, comparison in comparisons.items():
                    percent_change = comparison.get("percent_change")
                    if percent_change is None:
                        continue

                    # Flag performance regressions >10%
                    # For throughput metrics, negative change is bad
                    # For latency/ttft metrics, positive change is bad
                    is_regression = False
                    if "throughput" in metric_name.lower():
                        if percent_change < -10:  # >10% decrease in throughput
                            is_regression = True
                    elif (
                        "latency" in metric_name.lower()
                        or "ttft" in metric_name.lower()
                    ):
                        if percent_change > 10:  # >10% increase in latency
                            is_regression = True

                    if is_regression:
                        regressions.append(
                            {
                                "job_name": job_name,
                                "type": "performance_regression",
                                "metric_name": metric_name,
                                "percent_change": percent_change,
                                "current_avg": comparison["current_avg"],
                                "historical_avg": comparison["historical_avg"],
                            }
                        )

        if regressions:
            print("\n" + "=" * 80)
            print("REGRESSIONS DETECTED:")
            print("=" * 80)
            for regression in regressions:
                print(f"\nJob: {regression['job_name']}")
                if regression["type"] == "high_failure_rate":
                    print(
                        f"  High failure rate: {regression['failure_rate']:.1f}% "
                        f"({regression['failures']}/{regression['total_runs']})"
                    )
                elif regression["type"] == "consecutive_failures":
                    print(
                        f"  {regression['recent_failure_count']} recent consecutive failures"
                    )
                elif regression["type"] == "performance_regression":
                    print(f"  Performance regression: {regression['metric_name']}")
                    print(
                        f"    Change: {regression['percent_change']:+.1f}% "
                        f"(current: {regression['current_avg']:.2f}, "
                        f"7d avg: {regression['historical_avg']:.2f})"
                    )
            print("=" * 80)

        return regressions


def main():
    parser = argparse.ArgumentParser(
        description="Monitor nightly test runs for regressions"
    )
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to analyze (default: 7)"
    )
    parser.add_argument("--output", help="Output file for detailed stats (JSON)")

    args = parser.parse_args()

    monitor = NightlyTestMonitor(args.token)

    # Get nightly runs
    runs = monitor.get_nightly_runs(days=args.days)

    if not runs:
        print("No nightly test runs found in the specified time period.")
        sys.exit(1)

    # Analyze runs
    stats = monitor.analyze_nightly_tests(runs)

    # Generate report
    monitor.generate_report(stats, args.output)

    # Detect regressions
    regressions = monitor.detect_regressions(stats)

    # Generate GitHub Actions summary
    monitor.generate_github_summary(stats, regressions)

    # Exit with error code if regressions detected
    if regressions:
        sys.exit(1)
    else:
        print("\n✓ No significant regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
