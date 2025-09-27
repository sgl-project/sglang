#!/usr/bin/env python3
"""
SGLang CI Analyzer
Simple tool to analyze CI failures for SGLang project
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List

import requests


class SGLangCIAnalyzer:
    """SGLang CI Analyzer"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-CI-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_recent_runs(self, limit: int = 100) -> List[Dict]:
        """Get recent CI run data"""
        print(f"Fetching {limit} recent CI runs...")

        all_runs = []
        page = 1
        per_page = 100

        while len(all_runs) < limit:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": min(per_page, limit - len(all_runs)), "page": page}

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                all_runs.extend(data["workflow_runs"])
                print(f"Fetched {len(all_runs)} runs so far...")

                if len(data["workflow_runs"]) < per_page:
                    break

                page += 1
                time.sleep(0.1)  # Avoid API rate limits

            except requests.exceptions.RequestException as e:
                print(f"Error fetching CI data: {e}")
                break

        return all_runs[:limit]

    def analyze_ci_failures(self, runs: List[Dict]) -> Dict:
        """Analyze CI failure patterns"""
        print("Analyzing CI failure data...")

        # SGLang specific job categories
        job_categories = {
            "sgl-kernel": [
                "sgl-kernel-build-wheels",
                "sgl-kernel-unit-test",
                "sgl-kernel-mla-test",
            ],
            "unit-test": [
                "unit-test-frontend",
                "unit-test-backend-1-gpu",
                "unit-test-backend-2-gpu",
                "unit-test-backend-4-gpu",
                "unit-test-backend-8-gpu",
            ],
            "performance": [
                "performance-test-1-gpu-part-1",
                "performance-test-1-gpu-part-2",
                "performance-test-2-gpu",
            ],
            "accuracy": ["accuracy-test-1-gpu", "accuracy-test-2-gpu"],
            "deepep": ["unit-test-deepep-4-gpu", "unit-test-deepep-8-gpu"],
            "b200": ["unit-test-backend-4-gpu-b200"],
        }

        stats = {
            "total_runs": len(runs),
            "failed_runs": 0,
            "successful_runs": 0,
            "cancelled_runs": 0,
            "skipped_runs": 0,
            "category_failures": defaultdict(int),
            "job_failures": defaultdict(int),
            "failure_patterns": defaultdict(int),
            "job_failure_links": defaultdict(
                list
            ),  # Store recent failure links for each job
            "job_last_success": {},  # Store last successful run for each job
        }

        total_runs = len(runs)
        for i, run in enumerate(runs, 1):
            # Show progress every 10% or every 50 runs, whichever is smaller
            if i % max(1, min(50, total_runs // 10)) == 0 or i == total_runs:
                progress = (i / total_runs) * 100
                print(f"Progress: {i}/{total_runs} ({progress:.1f}%)")

            run_status = run.get("conclusion", "unknown")
            workflow_name = run.get("name", "Unknown")
            run_id = run.get("id")
            run_number = run.get("run_number")
            created_at = run.get("created_at")

            # Count run status
            if run_status == "failure":
                stats["failed_runs"] += 1
            elif run_status == "success":
                stats["successful_runs"] += 1
            elif run_status == "cancelled":
                stats["cancelled_runs"] += 1
            elif run_status == "skipped":
                stats["skipped_runs"] += 1

            # Get detailed job information for all runs
            jobs = self._get_job_details(run_id)
            run_url = f"https://github.com/{self.repo}/actions/runs/{run_id}"
            pr_info = self._get_pr_info(run)

            for job in jobs:
                job_name = job.get("name", "Unknown")
                job_conclusion = job.get("conclusion", "unknown")

                # Filter out non-specific CI jobs
                if job_name not in [
                    "check-changes",
                    "pr-test-finish",
                    "pr-test-h20-finish",
                    "lint",
                ]:
                    # Record successful jobs (update last success)
                    if job_conclusion == "success":
                        stats["job_last_success"][job_name] = {
                            "url": run_url,
                            "run_number": run_number,
                            "created_at": created_at,
                            "pr_info": pr_info,
                        }

                    # Record failed jobs
                    elif job_conclusion == "failure" and run_status == "failure":
                        stats["job_failures"][job_name] += 1

                        # Store failure link (keep only last 3 for each job)
                        if len(stats["job_failure_links"][job_name]) < 3:
                            stats["job_failure_links"][job_name].append(
                                {
                                    "url": run_url,
                                    "run_number": run_number,
                                    "created_at": created_at,
                                    "pr_info": pr_info,
                                }
                            )

                        # Categorize failed jobs
                        for category, jobs_list in job_categories.items():
                            if any(
                                job_pattern in job_name for job_pattern in jobs_list
                            ):
                                stats["category_failures"][category] += 1
                                break

                        # Analyze failure patterns
                        self._analyze_failure_pattern(job, stats)

            time.sleep(0.1)  # Avoid API rate limits

        return stats

    def _get_job_details(self, run_id: int) -> List[Dict]:
        """Get job details for a specific run"""
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except:
            return []

    def _get_pr_info(self, run: Dict) -> Dict:
        """Get PR information from a run"""
        pr_info = {
            "pr_number": None,
            "author": run.get("head_commit", {})
            .get("author", {})
            .get("name", "Unknown"),
            "head_sha": run.get("head_sha", ""),
            "head_branch": run.get("head_branch", ""),
        }

        # Try to extract PR number from pull_requests
        pull_requests = run.get("pull_requests", [])
        if pull_requests:
            pr_info["pr_number"] = pull_requests[0].get("number")

        return pr_info

    def _analyze_failure_pattern(self, job: Dict, stats: Dict):
        """Analyze failure patterns"""
        job_name = job.get("name", "")
        steps = job.get("steps", [])

        for step in steps:
            if step.get("conclusion") == "failure":
                step_name = step.get("name", "")

                # SGLang specific failure pattern recognition
                if "timeout" in step_name.lower():
                    stats["failure_patterns"]["Timeout"] += 1
                elif "test" in step_name.lower() and "unit" in job_name.lower():
                    stats["failure_patterns"]["Unit Test Failure"] += 1
                elif "performance" in job_name.lower():
                    stats["failure_patterns"]["Performance Test Failure"] += 1
                elif "accuracy" in job_name.lower():
                    stats["failure_patterns"]["Accuracy Test Failure"] += 1
                elif "build" in step_name.lower():
                    stats["failure_patterns"]["Build Failure"] += 1
                elif "install" in step_name.lower():
                    stats["failure_patterns"]["Dependency Installation Failure"] += 1
                elif "gpu" in job_name.lower():
                    stats["failure_patterns"]["GPU Related Failure"] += 1
                else:
                    stats["failure_patterns"]["Other"] += 1

    def generate_report(self, stats: Dict):
        """Generate CI analysis report"""
        print("\n" + "=" * 60)
        print("SGLang CI Analysis Report")
        print("=" * 60)

        # Overall statistics
        total = stats["total_runs"]
        failed = stats["failed_runs"]
        success = stats["successful_runs"]
        cancelled = stats["cancelled_runs"]
        skipped = stats["skipped_runs"]
        success_rate = (success / total * 100) if total > 0 else 0

        print(f"\nOverall Statistics:")
        print(f"  Total runs: {total}")
        print(f"  Successful: {success}")
        print(f"  Failed: {failed}")
        print(f"  Cancelled: {cancelled}")
        print(f"  Skipped: {skipped}")
        print(f"  Success rate: {success_rate:.1f}%")

        # Category failure statistics
        if stats["category_failures"]:
            print(f"\nCategory Failure Statistics:")
            for category, count in sorted(
                stats["category_failures"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {category}: {count} failures")

        # Most frequently failed jobs with links
        if stats["job_failures"]:
            print(f"\nMost Frequently Failed Jobs (Top 50):")
            for i, (job, count) in enumerate(
                sorted(stats["job_failures"].items(), key=lambda x: x[1], reverse=True)[
                    :50
                ],
                1,
            ):
                print(f"  {i:2d}. {job}: {count} times")

                # Show last successful run
                if job in stats["job_last_success"]:
                    last_success = stats["job_last_success"][job]
                    success_date = datetime.fromisoformat(
                        last_success["created_at"].replace("Z", "+00:00")
                    )
                    pr_info = last_success["pr_info"]

                    pr_text = ""
                    if pr_info["pr_number"]:
                        pr_text = (
                            f" (PR #{pr_info['pr_number']} by {pr_info['author']})"
                        )
                    else:
                        pr_text = f" by {pr_info['author']}"

                    print(
                        f"      Last Success: Run #{last_success['run_number']} ({success_date.strftime('%Y-%m-%d %H:%M')}){pr_text}: {last_success['url']}"
                    )

                # Show recent failure links
                if (
                    job in stats["job_failure_links"]
                    and stats["job_failure_links"][job]
                ):
                    print("      Recent Failures:")
                    for link_info in stats["job_failure_links"][job]:
                        created_at = datetime.fromisoformat(
                            link_info["created_at"].replace("Z", "+00:00")
                        )

                        # Format PR info for failures
                        pr_info = link_info.get("pr_info", {})
                        pr_text = ""
                        if pr_info.get("pr_number"):
                            pr_text = f" (PR #{pr_info['pr_number']} by {pr_info.get('author', 'Unknown')})"
                        else:
                            pr_text = f" by {pr_info.get('author', 'Unknown')}"

                        print(
                            f"        - Run #{link_info['run_number']} ({created_at.strftime('%Y-%m-%d %H:%M')}){pr_text}: {link_info['url']}"
                        )

        # Failure pattern analysis
        if stats["failure_patterns"]:
            print(f"\nFailure Pattern Analysis:")
            for pattern, count in sorted(
                stats["failure_patterns"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {pattern}: {count} times")

        print("\n" + "=" * 60)

    def save_detailed_report(self, stats: Dict, output_file: str = "ci_analysis.json"):
        """Save detailed report to file"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SGLang CI Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of runs to analyze (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="ci_analysis.json",
        help="Output file (default: ci_analysis.json)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = SGLangCIAnalyzer(args.token)

    try:
        # Get CI run data
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No CI run data found")
            return

        # Analyze failures
        stats = analyzer.analyze_ci_failures(runs)

        # Generate report
        analyzer.generate_report(stats)

        # Save detailed report
        analyzer.save_detailed_report(stats, args.output)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
