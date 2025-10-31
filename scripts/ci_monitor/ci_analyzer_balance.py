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

        self.test_time_pattern = re.compile(
            r"filename='([^']+)',\s*elapsed=(\d+),\s*estimated_time=(\d+)"
        )

    def get_recent_runs(self, limit: int = 1000) -> List[Dict]:
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
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching CI data: {e}")
                break

        return all_runs[:limit]

    def get_job_logs(self, run_id: int, job_name: str) -> Optional[str]:
        try:
            jobs_url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            response = self.session.get(jobs_url)
            response.raise_for_status()
            jobs_data = response.json()

            target_job = None
            for job in jobs_data.get("jobs", []):
                if job.get("name", "") == job_name:
                    target_job = job
                    break

            if not target_job:
                return None

            logs_url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{target_job['id']}/logs"
            response = self.session.get(logs_url)
            response.raise_for_status()

            return response.text

        except Exception as e:
            if "404" not in str(e):
                print(f"Failed to get job {job_name} logs: {e}")
            return None

    def get_all_jobs_for_run(self, run_id: int) -> List[Dict]:
        try:
            jobs_url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            response = self.session.get(jobs_url)
            response.raise_for_status()
            jobs_data = response.json()
            return jobs_data.get("jobs", [])
        except Exception as e:
            print(f"Failed to get jobs for run {run_id}: {e}")
            return []

    def get_job_logs_by_id(self, job_id: int) -> Optional[str]:
        try:
            logs_url = f"{self.base_url}/repos/{self.repo}/actions/jobs/{job_id}/logs"
            response = self.session.get(logs_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            if "404" not in str(e):
                print(f"Failed to get job {job_id} logs: {e}")
            return None

    def parse_test_times(self, log_content: str) -> List[Dict]:
        if not log_content:
            return []

        test_times = []
        matches = self.test_time_pattern.findall(log_content)
        filtered_count = 0

        for match in matches:
            filename, elapsed_str, estimated_str = match
            try:
                elapsed = int(elapsed_str)
                estimated = int(estimated_str)
                gap = elapsed - estimated

                if self._is_abnormal_test_data(
                    elapsed, estimated, log_content, filename
                ):
                    filtered_count += 1
                    continue

                test_times.append(
                    {
                        "filename": filename,
                        "elapsed": elapsed,
                        "estimated": estimated,
                        "gap": gap,
                    }
                )
            except ValueError:
                continue

        return test_times

    def _is_abnormal_test_data(
        self, elapsed: int, estimated: int, log_content: str, filename: str
    ) -> bool:

        # To avoid collect retry data
        if elapsed % estimated == 0:
            return True

        return False

    def collect_test_balance_data(self, runs: List[Dict]) -> Dict[str, Dict]:
        print("Starting test balance data collection...")

        test_gaps = defaultdict(
            lambda: {
                "max_gap": 0,
                "max_elapsed": 0,
                "max_estimated": 0,
                "max_gap_run_info": {},
                "total_runs": 0,
                "all_gaps": [],
            }
        )

        total_tests_parsed = 0
        abnormal_tests_filtered = 0

        target_job_prefixes = [
            "unit-test-frontend",
            "unit-test-backend-1-gpu",
            "unit-test-backend-2-gpu",
            "unit-test-backend-4-gpu",
            "unit-test-backend-8-gpu-h200",
            "unit-test-backend-8-gpu-h20",
            "unit-test-backend-4-gpu-b200",
            "unit-test-backend-4-gpu-gb200",
            "unit-test-deepep-4-gpu",
            "unit-test-deepep-8-gpu",
            "unit-test-backend-8-gpu-deepseek-v32",
            "performance-test-1-gpu-part-1",
            "performance-test-1-gpu-part-2",
            "performance-test-1-gpu-part-3",
            "performance-test-2-gpu",
            "accuracy-test-1-gpu",
            "accuracy-test-2-gpu",
        ]

        total_runs = len(runs)
        for i, run in enumerate(runs, 1):
            if i % 10 == 0 or i == total_runs:
                print(f"Processing run {i}/{total_runs}: #{run.get('run_number')}")

            workflow_name = run.get("name", "")
            if "AMD" in workflow_name or "amd" in workflow_name.lower():
                continue

            run_info = {
                "run_number": run.get("run_number"),
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

            all_jobs = self.get_all_jobs_for_run(run.get("id"))

            for job in all_jobs:
                job_name = job.get("name", "")
                job_id = job.get("id")

                matches_prefix = False
                for prefix in target_job_prefixes:
                    if job_name.startswith(prefix):
                        matches_prefix = True
                        break

                if not matches_prefix:
                    continue

                logs = self.get_job_logs_by_id(job_id)
                if not logs:
                    continue

                test_times = self.parse_test_times(logs)
                total_tests_parsed += len(test_times)

                for test_data in test_times:
                    filename = test_data["filename"]
                    elapsed = test_data["elapsed"]
                    estimated = test_data["estimated"]
                    gap = test_data["gap"]

                    test_stats = test_gaps[filename]
                    test_stats["total_runs"] += 1
                    test_stats["all_gaps"].append(gap)

                    if gap > test_stats["max_gap"]:
                        test_stats["max_gap"] = gap
                        test_stats["max_elapsed"] = elapsed
                        test_stats["max_estimated"] = estimated
                        test_stats["max_gap_run_info"] = {
                            **run_info,
                            "job_name": job_name,
                            "job_url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}/job/{job_id}",
                        }

            time.sleep(0.1)

        return dict(test_gaps)

    def generate_balance_report(
        self, test_data: Dict[str, Dict], output_file: str = "test_balance_report.json"
    ):
        print("\n" + "=" * 80)
        print("SGLang Test Balance Analysis Report (PR Test GPU Jobs)")
        print("=" * 80)

        sorted_tests = sorted(
            test_data.items(), key=lambda x: x[1]["max_gap"], reverse=True
        )

        print(f"\nTotal tests analyzed: {len(sorted_tests)}")
        print(
            f"Tests with significant gaps (>100s): {len([t for t in sorted_tests if t[1]['max_gap'] > 100])}"
        )
        print(
            f"Tests with large gaps (>300s): {len([t for t in sorted_tests if t[1]['max_gap'] > 300])}"
        )
        print(
            f"Note: Abnormal test data (due to failures/retries) has been filtered out"
        )

        report_data = {
            "summary": {
                "total_tests": len(sorted_tests),
                "tests_with_gaps_over_100s": len(
                    [t for t in sorted_tests if t[1]["max_gap"] > 100]
                ),
                "tests_with_gaps_over_300s": len(
                    [t for t in sorted_tests if t[1]["max_gap"] > 300]
                ),
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "test_balance_table": [],
        }

        print(f"\nTop 50 PR Test GPU Jobs with Largest Time Gaps:")
        print("-" * 100)
        print(
            f"{'Rank':<4} {'Test File':<40} {'Max Gap':<8} {'Max Elapsed':<12} {'Max Estimated':<15} {'Job Name':<25}"
        )
        print("-" * 100)

        for i, (filename, stats) in enumerate(sorted_tests[:50], 1):
            test_name = filename.split("/")[-1] if "/" in filename else filename
            job_name = (
                stats["max_gap_run_info"].get("job_name", "Unknown")
                if stats["max_gap_run_info"]
                else "Unknown"
            )

            print(
                f"{i:<4} {test_name:<40} {stats['max_gap']:<8} {stats['max_elapsed']:<12} {stats['max_estimated']:<15} {job_name:<25}"
            )

            report_data["test_balance_table"].append(
                {
                    "rank": i,
                    "filename": filename,
                    "test_name": test_name,
                    "max_gap": stats["max_gap"],
                    "max_elapsed": stats["max_elapsed"],
                    "max_estimated": stats["max_estimated"],
                    "max_gap_run_info": stats["max_gap_run_info"],
                    "total_runs": stats["total_runs"],
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {output_file}")

        return report_data

    def generate_github_summary(self, report_data: Dict):
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("Not running in GitHub Actions, skipping summary generation")
                return

            print("Generating GitHub Actions summary for Test Balance Analysis...")

            summary_lines = []
            summary_lines.append(
                "# SGLang Test Balance Analysis Report (PR Test GPU Jobs)"
            )
            summary_lines.append("")
            summary_lines.append(
                f"**Analysis Timestamp:** {report_data['summary']['analysis_timestamp']}"
            )
            summary_lines.append("")

            summary_lines.append("## Summary Statistics")
            summary_lines.append("")
            summary_lines.append("| Metric | Count |")
            summary_lines.append("|--------|-------|")
            summary_lines.append(
                f"| Total Tests Analyzed | {report_data['summary']['total_tests']} |"
            )
            summary_lines.append(
                f"| Tests with Gaps > 100s | {report_data['summary']['tests_with_gaps_over_100s']} |"
            )
            summary_lines.append(
                f"| Tests with Gaps > 300s | {report_data['summary']['tests_with_gaps_over_300s']} |"
            )
            summary_lines.append("")

            summary_lines.append("## Top 30 PR Test GPU Jobs with Largest Time Gaps")
            summary_lines.append("")
            summary_lines.append(
                "| Rank | Test File | Max Gap (s) | Max Elapsed (s) | Max Estimated (s) | Job Name | Job Link | Total Runs |"
            )
            summary_lines.append(
                "|------|-----------|-------------|----------------|------------------|---------|----------|------------|"
            )

            for test in report_data["test_balance_table"][:30]:
                test_name = test["test_name"]
                if len(test_name) > 30:
                    test_name = test_name[:27] + "..."

                job_name = (
                    test["max_gap_run_info"].get("job_name", "Unknown")
                    if test["max_gap_run_info"]
                    else "Unknown"
                )
                job_url = (
                    test["max_gap_run_info"].get("job_url", "")
                    if test["max_gap_run_info"]
                    else ""
                )
                job_link = f"[{job_name}]({job_url})" if job_url else job_name

                summary_lines.append(
                    f"| {test['rank']} | `{test_name}` | {test['max_gap']} | {test['max_elapsed']} | {test['max_estimated']} | {job_name} | [{job_name}]({job_url}) | {test['total_runs']} |"
                )

            summary_lines.append("")
            summary_lines.append("## Recommendations")
            summary_lines.append("")
            summary_lines.append(
                "Based on the analysis above, consider adjusting estimated times for tests with large gaps:"
            )
            summary_lines.append("")

            top_5_tests = report_data["test_balance_table"][:5]
            for test in top_5_tests:
                test_name = test["test_name"]
                if len(test_name) > 40:
                    test_name = test_name[:37] + "..."
                suggested_estimated = test["max_elapsed"] + 50
                summary_lines.append(
                    f"- **{test_name}**: Current max elapsed: {test['max_elapsed']}s, suggested estimated: {suggested_estimated}s"
                )

            summary_lines.append("")
            summary_lines.append(
                "Set estimated times to be slightly higher than the maximum observed elapsed time to avoid CI timeouts."
            )

            with open(github_step_summary, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")

    def save_csv_report(
        self, report_data: Dict, output_file: str = "test_balance_report.csv"
    ):
        import csv

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "Rank",
                    "Test File",
                    "Test Name",
                    "Max Gap (s)",
                    "Max Elapsed (s)",
                    "Max Estimated (s)",
                    "Job Name",
                    "Max Gap Job URL",
                    "Total Runs",
                ]
            )

            for test in report_data["test_balance_table"]:
                max_job_url = (
                    test["max_gap_run_info"].get("job_url", "")
                    if test["max_gap_run_info"]
                    else ""
                )
                job_name = (
                    test["max_gap_run_info"].get("job_name", "Unknown")
                    if test["max_gap_run_info"]
                    else "Unknown"
                )

                writer.writerow(
                    [
                        test["rank"],
                        test["filename"],
                        test["test_name"],
                        test["max_gap"],
                        test["max_elapsed"],
                        test["max_estimated"],
                        job_name,
                        max_job_url,
                        test["total_runs"],
                    ]
                )

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

    analyzer = SGLangTestBalanceAnalyzer(args.token)

    try:
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No CI run data found")
            return

        test_data = analyzer.collect_test_balance_data(runs)

        if not test_data:
            print("No test balance data found")
            return

        report_data = analyzer.generate_balance_report(test_data, args.output)

        csv_output = args.output.replace(".json", ".csv")
        analyzer.save_csv_report(report_data, csv_output)

        analyzer.generate_github_summary(report_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
