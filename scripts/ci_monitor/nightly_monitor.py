#!/usr/bin/env python3
"""
Nightly Test Monitor

Monitors nightly test runs for performance and accuracy regressions.
Analyzes metrics from GitHub summaries and tracks trends over time.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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

    def parse_metrics_from_summary(self, run_id: int, job_id: int) -> List[Dict]:
        """
        Parse metrics from GitHub step summary.
        This would ideally download the summary artifact and parse JSON metrics.
        For now, we'll track basic job success/failure and timing.
        """
        # TODO: Implement actual metric parsing from step summary artifacts
        # This would use the MetricReport JSON format we set up
        return []

    def analyze_nightly_tests(self, runs: List[Dict]) -> Dict:
        """Analyze nightly test runs for failures and performance"""
        print("Analyzing nightly test data...")

        stats = {
            "total_runs": len(runs),
            "successful_runs": 0,
            "failed_runs": 0,
            "cancelled_runs": 0,
            "job_stats": defaultdict(lambda: {
                "total": 0,
                "success": 0,
                "failure": 0,
                "recent_failures": [],
                "avg_duration_minutes": 0,
                "durations": [],
            }),
            "daily_stats": defaultdict(lambda: {
                "total": 0,
                "success": 0,
                "failure": 0,
            }),
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
                started_at = job.get("started_at")
                completed_at = job.get("completed_at")

                # Only track our nightly test jobs
                if job_name not in self.nightly_jobs:
                    continue

                job_stat = stats["job_stats"][job_name]
                job_stat["total"] += 1

                if job_conclusion == "success":
                    job_stat["success"] += 1
                elif job_conclusion == "failure":
                    job_stat["failure"] += 1

                    # Store recent failures (up to 5)
                    if len(job_stat["recent_failures"]) < 5:
                        job_stat["recent_failures"].append({
                            "run_url": run_url,
                            "run_number": run_number,
                            "created_at": created_at,
                            "job_url": job.get("html_url"),
                        })

                # Track duration
                if started_at and completed_at:
                    try:
                        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                        duration_minutes = (end - start).total_seconds() / 60
                        job_stat["durations"].append(duration_minutes)
                    except:
                        pass

            time.sleep(0.1)

        # Calculate average durations
        for job_name, job_stat in stats["job_stats"].items():
            if job_stat["durations"]:
                job_stat["avg_duration_minutes"] = sum(job_stat["durations"]) / len(job_stat["durations"])
                del job_stat["durations"]  # Remove raw data to reduce size

        return stats

    def generate_report(self, stats: Dict, output_file: str = None):
        """Generate a human-readable report"""
        print("\n" + "=" * 80)
        print("NIGHTLY TEST MONITOR REPORT")
        print("=" * 80)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Runs Analyzed: {stats['total_runs']}")
        print(f"Successful: {stats['successful_runs']} ({stats['successful_runs']/max(1, stats['total_runs'])*100:.1f}%)")
        print(f"Failed: {stats['failed_runs']} ({stats['failed_runs']/max(1, stats['total_runs'])*100:.1f}%)")
        print(f"Cancelled: {stats['cancelled_runs']}")
        print("=" * 80)

        # Daily trend
        print("\nDAILY TRENDS:")
        print("-" * 80)
        daily_stats = sorted(stats["daily_stats"].items(), reverse=True)[:7]
        for date, day_stats in daily_stats:
            success_rate = (day_stats["success"] / max(1, day_stats["total"])) * 100
            print(f"{date}: {day_stats['total']} runs, {day_stats['success']} success ({success_rate:.1f}%), {day_stats['failure']} failed")

        # Job statistics
        print("\nJOB STATISTICS:")
        print("-" * 80)
        print(f"{'Job Name':<40} {'Total':<8} {'Success':<8} {'Failed':<8} {'Rate':<8} {'Avg Duration'}")
        print("-" * 80)

        job_stats_sorted = sorted(
            stats["job_stats"].items(),
            key=lambda x: x[1]["failure"],
            reverse=True
        )

        for job_name, job_stat in job_stats_sorted:
            total = job_stat["total"]
            success = job_stat["success"]
            failure = job_stat["failure"]
            success_rate = (success / max(1, total)) * 100
            avg_duration = job_stat["avg_duration_minutes"]

            print(f"{job_name:<40} {total:<8} {success:<8} {failure:<8} {success_rate:>6.1f}% {avg_duration:>7.1f}m")

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
                    regressions.append({
                        "job_name": job_name,
                        "type": "high_failure_rate",
                        "failure_rate": failure_rate,
                        "total_runs": total,
                        "failures": failure,
                    })

                # Flag jobs with recent consecutive failures
                recent_failures = len(job_stat["recent_failures"])
                if recent_failures >= 3:
                    regressions.append({
                        "job_name": job_name,
                        "type": "consecutive_failures",
                        "recent_failure_count": recent_failures,
                    })

        if regressions:
            print("\n" + "⚠" * 40)
            print("POTENTIAL REGRESSIONS DETECTED:")
            print("⚠" * 40)
            for regression in regressions:
                print(f"\nJob: {regression['job_name']}")
                if regression["type"] == "high_failure_rate":
                    print(f"  High failure rate: {regression['failure_rate']:.1f}% ({regression['failures']}/{regression['total_runs']})")
                elif regression["type"] == "consecutive_failures":
                    print(f"  {regression['recent_failure_count']} recent consecutive failures")
            print("⚠" * 40)

        return regressions


def main():
    parser = argparse.ArgumentParser(
        description="Monitor nightly test runs for regressions"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="GitHub personal access token"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)"
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed stats (JSON)"
    )

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

    # Exit with error code if regressions detected
    if regressions:
        sys.exit(1)
    else:
        print("\n✓ No significant regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
