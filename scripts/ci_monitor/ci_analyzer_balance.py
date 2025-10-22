#!/usr/bin/env python3
"""
SGLang CI Test Balance Analyzer
Analyze test time gaps between elapsed and estimated times to help balance CI
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests


class SGLangTestBalanceAnalyzer:
    """SGLang CI Test Balance Analyzer"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Test-Balance-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Pattern to match test time information from logs
        # Example: filename='/public_sglang_ci/runner-l3b-gpu-0/_work/sglang/sglang/test/srt/models/test_encoder_embedding_models.py', elapsed=350, estimated_time=100
        self.test_time_pattern = re.compile(
            r"filename='([^']+)',\s*elapsed=(\d+),\s*estimated_time=(\d+)"
        )

    def get_recent_runs(self, limit: int = 1000) -> List[Dict]:
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

    def get_job_logs(self, run_id: int, job_name: str) -> Optional[str]:
        """Get logs for specific job"""
        try:
            # First get job list
            jobs_url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            response = self.session.get(jobs_url)
            response.raise_for_status()
            jobs_data = response.json()

            # Find matching job
            target_job = None
            for job in jobs_data.get("jobs", []):
                if job_name in job.get("name", ""):
                    target_job = job
                    break

            if not target_job:
                return None

            # Get logs
            logs_url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{target_job['id']}/logs"
            response = self.session.get(logs_url)
            response.raise_for_status()

            return response.text

        except Exception as e:
            if "404" not in str(e):
                print(f"Failed to get job {job_name} logs: {e}")
            return None

    def parse_test_times(self, log_content: str) -> List[Dict]:
        """Parse test time information from logs"""
        if not log_content:
            return []

        test_times = []
        matches = self.test_time_pattern.findall(log_content)

        for match in matches:
            filename, elapsed_str, estimated_str = match
            try:
                elapsed = int(elapsed_str)
                estimated = int(estimated_str)
                gap = elapsed - estimated

                test_times.append({
                    "filename": filename,
                    "elapsed": elapsed,
                    "estimated": estimated,
                    "gap": gap
                })
            except ValueError:
                continue

        return test_times

    def collect_test_balance_data(self, runs: List[Dict]) -> Dict[str, Dict]:
        """Collect test balance data from all runs"""
        print("Starting test balance data collection...")

        # Track gap statistics for each test
        test_gaps = defaultdict(lambda: {
            "max_gap": 0,
            "max_elapsed": 0,
            "max_estimated": 0,
            "max_gap_run_info": {},
            "min_gap": float('inf'),
            "min_elapsed": 0,
            "min_estimated": 0,
            "min_gap_run_info": {},
            "total_runs": 0,
            "all_gaps": []
        })

        # Job names to analyze (unit test jobs)
        target_jobs = [
            "unit-test-backend-1-gpu",
            "unit-test-backend-2-gpu", 
            "unit-test-backend-4-gpu",
            "unit-test-backend-8-gpu",
            "unit-test-frontend"
        ]

        total_runs = len(runs)
        for i, run in enumerate(runs, 1):
            if i % 50 == 0 or i == total_runs:
                print(f"Processing run {i}/{total_runs}: #{run.get('run_number')}")

            run_info = {
                "run_number": run.get("run_number"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": run.get("head_commit", {})
                .get("author", {})
                .get("name", "Unknown"),
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

            # Extract PR number
            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            # Get logs for each target job
            for job_name in target_jobs:
                logs = self.get_job_logs(run.get("id"), job_name)
                if not logs:
                    continue

                # Parse test times from logs
                test_times = self.parse_test_times(logs)
                
                for test_data in test_times:
                    filename = test_data["filename"]
                    elapsed = test_data["elapsed"]
                    estimated = test_data["estimated"]
                    gap = test_data["gap"]

                    # Update statistics for this test
                    test_stats = test_gaps[filename]
                    test_stats["total_runs"] += 1
                    test_stats["all_gaps"].append(gap)

                    # Track maximum gap
                    if gap > test_stats["max_gap"]:
                        test_stats["max_gap"] = gap
                        test_stats["max_elapsed"] = elapsed
                        test_stats["max_estimated"] = estimated
                        test_stats["max_gap_run_info"] = {
                            **run_info,
                            "job_name": job_name
                        }

                    # Track minimum gap
                    if gap < test_stats["min_gap"]:
                        test_stats["min_gap"] = gap
                        test_stats["min_elapsed"] = elapsed
                        test_stats["min_estimated"] = estimated
                        test_stats["min_gap_run_info"] = {
                            **run_info,
                            "job_name": job_name
                        }

            time.sleep(0.1)  # Avoid API rate limits

        return dict(test_gaps)

    def generate_balance_report(self, test_data: Dict[str, Dict], output_file: str = "test_balance_report.json"):
        """Generate test balance report"""
        print("\n" + "=" * 80)
        print("SGLang Test Balance Analysis Report")
        print("=" * 80)

        # Convert to list and sort by max gap (descending)
        sorted_tests = sorted(
            test_data.items(), 
            key=lambda x: x[1]["max_gap"], 
            reverse=True
        )

        print(f"\nTotal tests analyzed: {len(sorted_tests)}")
        print(f"Tests with significant gaps (>100s): {len([t for t in sorted_tests if t[1]['max_gap'] > 100])}")
        print(f"Tests with large gaps (>300s): {len([t for t in sorted_tests if t[1]['max_gap'] > 300])}")

        # Generate detailed report
        report_data = {
            "summary": {
                "total_tests": len(sorted_tests),
                "tests_with_gaps_over_100s": len([t for t in sorted_tests if t[1]['max_gap'] > 100]),
                "tests_with_gaps_over_300s": len([t for t in sorted_tests if t[1]['max_gap'] > 300]),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "test_balance_table": []
        }

        print(f"\nTop 100 Tests with Largest Time Gaps:")
        print("-" * 120)
        print(f"{'Rank':<4} {'Test File':<50} {'Max Gap':<8} {'Max Elapsed':<12} {'Max Estimated':<15} {'Min Gap':<8} {'Min Elapsed':<12} {'Min Estimated':<15}")
        print("-" * 120)

        for i, (filename, stats) in enumerate(sorted_tests[:100], 1):
            # Extract just the test filename from the full path
            test_name = filename.split("/")[-1] if "/" in filename else filename
            
            print(f"{i:<4} {test_name:<50} {stats['max_gap']:<8} {stats['max_elapsed']:<12} {stats['max_estimated']:<15} {stats['min_gap']:<8} {stats['min_elapsed']:<12} {stats['min_estimated']:<15}")
            
            # Add to report data
            report_data["test_balance_table"].append({
                "rank": i,
                "filename": filename,
                "test_name": test_name,
                "max_gap": stats["max_gap"],
                "max_elapsed": stats["max_elapsed"],
                "max_estimated": stats["max_estimated"],
                "max_gap_run_info": stats["max_gap_run_info"],
                "min_gap": stats["min_gap"],
                "min_elapsed": stats["min_elapsed"],
                "min_estimated": stats["min_estimated"],
                "min_gap_run_info": stats["min_gap_run_info"],
                "total_runs": stats["total_runs"]
            })

        # Save detailed report
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {output_file}")

        return report_data

    def generate_github_summary(self, report_data: Dict):
        """Generate GitHub Actions summary"""
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("Not running in GitHub Actions, skipping summary generation")
                return

            print("Generating GitHub Actions summary for Test Balance Analysis...")

            summary_lines = []
            summary_lines.append("# SGLang Test Balance Analysis Report")
            summary_lines.append("")
            summary_lines.append(f"**Analysis Timestamp:** {report_data['summary']['analysis_timestamp']}")
            summary_lines.append("")

            summary_lines.append("## Summary Statistics")
            summary_lines.append("")
            summary_lines.append("| Metric | Count |")
            summary_lines.append("|--------|-------|")
            summary_lines.append(f"| Total Tests Analyzed | {report_data['summary']['total_tests']} |")
            summary_lines.append(f"| Tests with Gaps > 100s | {report_data['summary']['tests_with_gaps_over_100s']} |")
            summary_lines.append(f"| Tests with Gaps > 300s | {report_data['summary']['tests_with_gaps_over_300s']} |")
            summary_lines.append("")

            summary_lines.append("## Top 20 Tests with Largest Time Gaps")
            summary_lines.append("")
            summary_lines.append("| Rank | Test File | Max Gap (s) | Max Elapsed (s) | Max Estimated (s) | Min Gap (s) | Min Elapsed (s) | Min Estimated (s) | Total Runs |")
            summary_lines.append("|------|-----------|-------------|----------------|------------------|-------------|----------------|------------------|------------|")

            for test in report_data["test_balance_table"][:20]:
                test_name = test["test_name"]
                if len(test_name) > 30:
                    test_name = test_name[:27] + "..."
                
                summary_lines.append(
                    f"| {test['rank']} | `{test_name}` | {test['max_gap']} | {test['max_elapsed']} | {test['max_estimated']} | {test['min_gap']} | {test['min_elapsed']} | {test['min_estimated']} | {test['total_runs']} |"
                )

            summary_lines.append("")
            summary_lines.append("## Recommendations")
            summary_lines.append("")
            summary_lines.append("Based on the analysis above, consider adjusting estimated times for tests with large gaps:")
            summary_lines.append("")
            
            top_5_tests = report_data["test_balance_table"][:5]
            for test in top_5_tests:
                test_name = test["test_name"]
                if len(test_name) > 40:
                    test_name = test_name[:37] + "..."
                suggested_estimated = test["max_elapsed"] + 50
                summary_lines.append(f"- **{test_name}**: Current max elapsed: {test['max_elapsed']}s, suggested estimated: {suggested_estimated}s")
            
            summary_lines.append("")
            summary_lines.append("Set estimated times to be slightly higher than the maximum observed elapsed time to avoid CI timeouts.")

            with open(github_step_summary, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")

    def save_csv_report(self, report_data: Dict, output_file: str = "test_balance_report.csv"):
        """Save CSV report for easy analysis"""
        import csv
        
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Rank", "Test File", "Test Name", "Max Gap (s)", "Max Elapsed (s)", 
                "Max Estimated (s)", "Max Gap Run URL", "Min Gap (s)", "Min Elapsed (s)", 
                "Min Estimated (s)", "Min Gap Run URL", "Total Runs"
            ])
            
            # Write data
            for test in report_data["test_balance_table"]:
                max_run_url = test["max_gap_run_info"].get("url", "") if test["max_gap_run_info"] else ""
                min_run_url = test["min_gap_run_info"].get("url", "") if test["min_gap_run_info"] else ""
                
                writer.writerow([
                    test["rank"],
                    test["filename"],
                    test["test_name"],
                    test["max_gap"],
                    test["max_elapsed"],
                    test["max_estimated"],
                    max_run_url,
                    test["min_gap"],
                    test["min_elapsed"],
                    test["min_estimated"],
                    min_run_url,
                    test["total_runs"]
                ])
        
        print(f"CSV report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SGLang Test Balance Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of runs to analyze (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default="test_balance_report.json",
        help="Output file (default: test_balance_report.json)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = SGLangTestBalanceAnalyzer(args.token)

    try:
        # Get CI run data
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No CI run data found")
            return

        # Collect test balance data
        test_data = analyzer.collect_test_balance_data(runs)

        if not test_data:
            print("No test balance data found")
            return

        # Generate report
        report_data = analyzer.generate_balance_report(test_data, args.output)

        # Generate CSV report
        csv_output = args.output.replace(".json", ".csv")
        analyzer.save_csv_report(report_data, csv_output)

        # Generate GitHub summary
        analyzer.generate_github_summary(report_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
