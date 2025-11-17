#!/usr/bin/env python3

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

    def get_recent_runs(self, limit: int = 100, branch: str = None) -> List[Dict]:
        branch_info = f" from branch '{branch}'" if branch else ""
        print(f"Fetching {limit} recent CI runs{branch_info}...")

        all_runs = []
        page = 1
        per_page = 100

        while len(all_runs) < limit:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": min(per_page, limit - len(all_runs)), "page": page}
            if branch:
                params["branch"] = branch

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

    def analyze_ci_failures(self, runs: List[Dict]) -> Dict:
        print(
            "Analyzing CI failure data (pr-test.yml, quantization-test.yml, nightly-test.yml jobs only)..."
        )

        job_categories = {
            "build": [
                "build-test",
                "sgl-kernel-build-wheels",
            ],
            "unit-test": [
                "stage-a-test-1",
                "unit-test-backend-1-gpu",
                "unit-test-backend-2-gpu",
                "unit-test-backend-4-gpu",
                "unit-test-backend-8-gpu",
            ],
            "performance": [
                "performance-test-1-gpu-part-1",
                "performance-test-1-gpu-part-2",
                "performance-test-1-gpu-part-3",
                "performance-test-2-gpu",
            ],
            "accuracy": [
                "accuracy-test-1-gpu",
                "accuracy-test-2-gpu",
            ],
            "mla-test": [
                "sgl-kernel-mla-test",
            ],
            "deepep": [
                "unit-test-deepep-4-gpu",
                "unit-test-deepep-8-gpu",
            ],
            "per-commit": [
                "per-commit-8-gpu-h20",
            ],
            "nightly": [
                "nightly-test-perf-text-models",
                "nightly-test-eval-text-models",
                "nightly-test-1-gpu",
                "nightly-test-4-gpu",
                "nightly-test-8-gpu-h200",
                "nightly-test-8-gpu-h20",
                "nightly-test-4-gpu-b200",
            ],
            "integration": [
                "run-all-notebooks",
                "quantization-test",
                "test-disaggregation",
            ],
            "b200": [
                "unit-test-backend-4-gpu-b200",
            ],
            "gb200": [
                "unit-test-backend-4-gpu-gb200",
            ],
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
            if i % max(1, min(50, total_runs // 10)) == 0 or i == total_runs:
                progress = (i / total_runs) * 100
                print(f"Progress: {i}/{total_runs} ({progress:.1f}%)")

            run_status = run.get("conclusion", "unknown")
            workflow_name = run.get("name", "Unknown")
            run_id = run.get("id")
            run_number = run.get("run_number")
            created_at = run.get("created_at")

            if run_status == "failure":
                stats["failed_runs"] += 1
            elif run_status == "success":
                stats["successful_runs"] += 1
            elif run_status == "cancelled":
                stats["cancelled_runs"] += 1
            elif run_status == "skipped":
                stats["skipped_runs"] += 1

            jobs = self._get_job_details(run_id)
            run_url = f"https://github.com/{self.repo}/actions/runs/{run_id}"
            pr_info = self._get_pr_info(run)

            for job in jobs:
                job_name = job.get("name", "Unknown")
                job_conclusion = job.get("conclusion", "unknown")

                target_jobs = [
                    "check-changes",
                    "sgl-kernel-build-wheels",
                    "sgl-kernel-unit-test",
                    "sgl-kernel-mla-test",
                    "sgl-kernel-benchmark-test",
                    "stage-a-test-1",
                    "unit-test-backend-1-gpu",
                    "unit-test-backend-2-gpu",
                    "unit-test-backend-4-gpu",
                    "unit-test-backend-8-gpu-h200",
                    "unit-test-backend-8-gpu-h20",
                    "performance-test-1-gpu-part-1",
                    "performance-test-1-gpu-part-2",
                    "performance-test-1-gpu-part-3",
                    "performance-test-2-gpu",
                    "accuracy-test-1-gpu",
                    "accuracy-test-2-gpu",
                    "unit-test-deepep-4-gpu",
                    "unit-test-deepep-8-gpu",
                    "unit-test-backend-8-gpu-deepseek-v32",
                    "unit-test-backend-4-gpu-b200",
                    "unit-test-backend-4-gpu-gb200",
                    "quantization-test",
                    "nightly-test-eval-text-models",
                    "nightly-test-perf-text-models",
                    "nightly-test-eval-vlms",
                    "nightly-test-perf-vlms",
                    "nightly-test-1-gpu",
                    "nightly-test-4-gpu",
                    "nightly-test-8-gpu-h200",
                    "nightly-test-8-gpu-h20",
                    "nightly-test-4-gpu-b200",
                ]

                if job_name in target_jobs:
                    if job_conclusion == "success":
                        stats["job_last_success"][job_name] = {
                            "url": run_url,
                            "run_number": run_number,
                            "created_at": created_at,
                            "pr_info": pr_info,
                        }

                    elif job_conclusion == "failure":
                        stats["job_failures"][job_name] += 1

                        if len(stats["job_failure_links"][job_name]) < 3:
                            stats["job_failure_links"][job_name].append(
                                {
                                    "url": run_url,
                                    "run_number": run_number,
                                    "created_at": created_at,
                                    "pr_info": pr_info,
                                }
                            )

                        for category, jobs_list in job_categories.items():
                            if any(
                                job_pattern in job_name for job_pattern in jobs_list
                            ):
                                stats["category_failures"][category] += 1
                                break

                        self._analyze_failure_pattern(job, stats)

            time.sleep(0.1)

        return stats

    def _get_job_details(self, run_id: int) -> List[Dict]:
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except:
            return []

    def _get_pr_info(self, run: Dict) -> Dict:
        pr_info = {
            "pr_number": None,
            "author": run.get("head_commit", {})
            .get("author", {})
            .get("name", "Unknown"),
            "head_sha": run.get("head_sha", ""),
            "head_branch": run.get("head_branch", ""),
        }

        pull_requests = run.get("pull_requests", [])
        if pull_requests:
            pr_info["pr_number"] = pull_requests[0].get("number")

        return pr_info

    def _analyze_failure_pattern(self, job: Dict, stats: Dict):
        job_name = job.get("name", "")
        steps = job.get("steps", [])

        for step in steps:
            if step.get("conclusion") == "failure":
                step_name = step.get("name", "")

                if "timeout" in step_name.lower():
                    stats["failure_patterns"]["Timeout"] += 1
                elif "build" in step_name.lower() or "build" in job_name.lower():
                    stats["failure_patterns"]["Build Failure"] += 1
                elif "install" in step_name.lower() or "dependency" in job_name.lower():
                    stats["failure_patterns"]["Dependency Installation Failure"] += 1
                elif "unit" in job_name.lower() or "unit-test" in job_name.lower():
                    stats["failure_patterns"]["Unit Test Failure"] += 1
                elif "performance" in job_name.lower() or "perf" in job_name.lower():
                    stats["failure_patterns"]["Performance Test Failure"] += 1
                elif "accuracy" in job_name.lower():
                    stats["failure_patterns"]["Accuracy Test Failure"] += 1
                elif "mla" in job_name.lower():
                    stats["failure_patterns"]["MLA Test Failure"] += 1
                elif "deepep" in job_name.lower():
                    stats["failure_patterns"]["DeepEP Test Failure"] += 1
                elif "nightly" in job_name.lower():
                    stats["failure_patterns"]["Nightly Test Failure"] += 1
                elif "notebook" in job_name.lower():
                    stats["failure_patterns"]["Notebook Test Failure"] += 1
                elif "disaggregation" in job_name.lower():
                    stats["failure_patterns"]["Disaggregation Test Failure"] += 1
                elif "h20" in job_name.lower() or "h200" in job_name.lower():
                    stats["failure_patterns"]["H20/H200 GPU Failure"] += 1
                elif "b200" in job_name.lower():
                    stats["failure_patterns"]["B200 GPU Failure"] += 1
                elif "gpu" in job_name.lower():
                    stats["failure_patterns"]["GPU Related Failure"] += 1
                else:
                    stats["failure_patterns"]["Other"] += 1

    def generate_report(self, stats: Dict):
        print("\n" + "=" * 60)
        print("SGLang CI Analysis Report (Target Workflows Only)")
        print("=" * 60)

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

        if stats["category_failures"]:
            print(f"\nCategory Failure Statistics:")
            for category, count in sorted(
                stats["category_failures"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {category}: {count} failures")

        if stats["job_failures"]:
            print(f"\nMost Frequently Failed Jobs (Top 50):")
            for i, (job, count) in enumerate(
                sorted(stats["job_failures"].items(), key=lambda x: x[1], reverse=True)[
                    :50
                ],
                1,
            ):
                print(f"  {i:2d}. {job}: {count} times")

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

                if (
                    job in stats["job_failure_links"]
                    and stats["job_failure_links"][job]
                ):
                    print("      Recent Failures:")
                    for link_info in stats["job_failure_links"][job]:
                        created_at = datetime.fromisoformat(
                            link_info["created_at"].replace("Z", "+00:00")
                        )

                        pr_info = link_info.get("pr_info", {})
                        pr_text = ""
                        if pr_info.get("pr_number"):
                            pr_text = f" (PR #{pr_info['pr_number']} by {pr_info.get('author', 'Unknown')})"
                        else:
                            pr_text = f" by {pr_info.get('author', 'Unknown')}"

                        print(
                            f"        - Run #{link_info['run_number']} ({created_at.strftime('%Y-%m-%d %H:%M')}){pr_text}: {link_info['url']}"
                        )

        if stats["failure_patterns"]:
            print(f"\nFailure Pattern Analysis:")
            for pattern, count in sorted(
                stats["failure_patterns"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {pattern}: {count} times")

        print("\n" + "=" * 60)

    def save_detailed_report(self, stats: Dict, output_file: str = "ci_analysis.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {output_file}")

    def generate_github_summary(self, stats: Dict):
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("Not running in GitHub Actions, skipping summary generation")
                return

            print("Generating GitHub Actions summary for CI Analysis...")

            summary_lines = []
            summary_lines.append("# SGLang CI Analysis Report (Target Workflows Only)")
            summary_lines.append("")

            total = stats["total_runs"]
            failed = stats["failed_runs"]
            success = stats["successful_runs"]
            cancelled = stats["cancelled_runs"]
            skipped = stats["skipped_runs"]
            success_rate = (success / total * 100) if total > 0 else 0

            summary_lines.append("## Overall Statistics")
            summary_lines.append("")
            summary_lines.append("| Metric | Count | Percentage |")
            summary_lines.append("|--------|-------|------------|")
            summary_lines.append(f"| Total Runs | {total} | 100% |")
            summary_lines.append(
                f"| Successful | {success} | {success/total*100:.1f}% |"
            )
            summary_lines.append(f"| Failed | {failed} | {failed/total*100:.1f}% |")
            summary_lines.append(
                f"| Cancelled | {cancelled} | {cancelled/total*100:.1f}% |"
            )
            summary_lines.append(f"| Skipped | {skipped} | {skipped/total*100:.1f}% |")
            summary_lines.append(f"| **Success Rate** | **{success_rate:.1f}%** | - |")
            summary_lines.append("")

            if stats["category_failures"]:
                summary_lines.append("## Category Failure Statistics")
                summary_lines.append("")
                summary_lines.append("| Category | Failures |")
                summary_lines.append("|----------|----------|")
                for category, count in sorted(
                    stats["category_failures"].items(), key=lambda x: x[1], reverse=True
                ):
                    summary_lines.append(f"| {category} | {count} |")
                summary_lines.append("")

            if stats["job_failures"]:
                summary_lines.append("## Most Frequently Failed Jobs (Top 20)")
                summary_lines.append("")

                top_failures = sorted(
                    stats["job_failures"].items(), key=lambda x: x[1], reverse=True
                )[:20]

                for i, (job, count) in enumerate(top_failures, 1):
                    summary_lines.append(f"### {i}. `{job}` ({count} failures)")
                    summary_lines.append("")

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

                        summary_lines.append(
                            f"**Last Success:** [Run #{last_success['run_number']}]({last_success['url']}) ({success_date.strftime('%Y-%m-%d %H:%M')}){pr_text}"
                        )
                        summary_lines.append("")

                    if (
                        job in stats["job_failure_links"]
                        and stats["job_failure_links"][job]
                    ):
                        summary_lines.append("**Recent Failures:**")
                        for link_info in stats["job_failure_links"][job]:
                            created_at = datetime.fromisoformat(
                                link_info["created_at"].replace("Z", "+00:00")
                            )

                            pr_info = link_info.get("pr_info", {})
                            pr_text = ""
                            if pr_info.get("pr_number"):
                                pr_text = f" (PR #{pr_info['pr_number']} by {pr_info.get('author', 'Unknown')})"
                            else:
                                pr_text = f" by {pr_info.get('author', 'Unknown')}"

                            summary_lines.append(
                                f"- [Run #{link_info['run_number']}]({link_info['url']}) ({created_at.strftime('%Y-%m-%d %H:%M')}){pr_text}"
                            )
                        summary_lines.append("")

            if stats["failure_patterns"]:
                summary_lines.append("## Failure Pattern Analysis")
                summary_lines.append("")
                summary_lines.append("| Pattern | Count |")
                summary_lines.append("|---------|-------|")
                for pattern, count in sorted(
                    stats["failure_patterns"].items(), key=lambda x: x[1], reverse=True
                ):
                    summary_lines.append(f"| {pattern} | {count} |")
                summary_lines.append("")

            with open(github_step_summary, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
                f.write("\n\n---\n\n")

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")


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
    parser.add_argument(
        "--branch",
        default="main",
        help="Filter runs by branch (default: 'main'). Set to empty string '' to analyze all branches.",
    )

    args = parser.parse_args()

    analyzer = SGLangCIAnalyzer(args.token)

    try:
        branch = args.branch if args.branch else None
        runs = analyzer.get_recent_runs(args.limit, branch)

        if not runs:
            print("No CI run data found")
            return

        stats = analyzer.analyze_ci_failures(runs)

        analyzer.generate_report(stats)

        analyzer.save_detailed_report(stats, args.output)

        analyzer.generate_github_summary(stats)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
