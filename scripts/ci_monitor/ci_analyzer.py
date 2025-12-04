#!/usr/bin/env python3

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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

        # Nightly workflow files to monitor
        self.nightly_workflows = [
            "nightly-test-nvidia.yml",
            "nightly-test-amd.yml",
            "nightly-test-intel.yml",
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
                # NVIDIA job names (nightly-test-nvidia.yml)
                "nightly-test-general-1-gpu-runner",
                "nightly-test-general-4-gpu-h100",
                "nightly-test-general-8-gpu-h200",
                "nightly-test-general-8-gpu-h20",
                "nightly-test-text-accuracy-2-gpu-runner",
                "nightly-test-text-perf-2-gpu-runner",
                "nightly-test-vlm-accuracy-2-gpu-runner",
                "nightly-test-vlm-perf-2-gpu-runner",
                "nightly-test-perf-4-gpu-b200",
                "nightly-test-perf-8-gpu-b200",
                # AMD job names (nightly-test-amd.yml)
                "nightly-test",  # AMD uses this generic name with matrix
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
            "performance_metrics": defaultdict(
                lambda: defaultdict(list)
            ),  # Track performance metrics for nightly jobs
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
                    # NVIDIA job names (nightly-test-nvidia.yml)
                    "nightly-test-general-1-gpu-runner",
                    "nightly-test-general-4-gpu-h100",
                    "nightly-test-general-8-gpu-h200",
                    "nightly-test-general-8-gpu-h20",
                    "nightly-test-text-accuracy-2-gpu-runner",
                    "nightly-test-text-perf-2-gpu-runner",
                    "nightly-test-vlm-accuracy-2-gpu-runner",
                    "nightly-test-vlm-perf-2-gpu-runner",
                    "nightly-test-perf-4-gpu-b200",
                    "nightly-test-perf-8-gpu-b200",
                    # AMD job names (nightly-test-amd.yml)
                    "nightly-test",
                ]

                if job_name in target_jobs:
                    if job_conclusion == "success":
                        stats["job_last_success"][job_name] = {
                            "url": run_url,
                            "run_number": run_number,
                            "created_at": created_at,
                            "pr_info": pr_info,
                        }

                        # Parse performance metrics from successful nightly jobs
                        if job_name in job_categories["nightly"] and (
                            "perf" in job_name.lower()
                            or "accuracy" in job_name.lower()
                            or "eval" in job_name.lower()
                        ):
                            job_id = job.get("id")
                            logs = self.get_job_logs(job_id)
                            if logs:
                                metrics = self.parse_metrics_from_logs(logs, job_name)
                                for metric_name, values in metrics.items():
                                    if values:
                                        for value in values:
                                            stats["performance_metrics"][job_name][
                                                metric_name
                                            ].append(
                                                {
                                                    "value": value,
                                                    "timestamp": created_at,
                                                    "run_id": run_id,
                                                    "run_url": run_url,
                                                }
                                            )

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

            # Performance metrics section for nightly jobs
            if stats.get("performance_metrics"):
                summary_lines.append("## Nightly Test Performance Metrics")
                summary_lines.append("")
                summary_lines.append("| Job | Metric | Latest Value | Count | Trend |")
                summary_lines.append("|-----|--------|--------------|-------|-------|")

                for job_name in sorted(stats["performance_metrics"].keys()):
                    job_metrics = stats["performance_metrics"][job_name]
                    for metric_name in sorted(job_metrics.keys()):
                        metric_data = job_metrics[metric_name]
                        if metric_data:
                            # Calculate average of recent values
                            values = [m["value"] for m in metric_data]
                            avg_value = sum(values) / len(values)
                            count = len(values)

                            # Simple trend: compare first half vs second half
                            trend_indicator = "➡️"
                            if len(values) >= 4:
                                first_half = values[: len(values) // 2]
                                second_half = values[len(values) // 2 :]
                                first_avg = sum(first_half) / len(first_half)
                                second_avg = sum(second_half) / len(second_half)

                                if first_avg > 0:
                                    change_pct = (
                                        (second_avg - first_avg) / first_avg
                                    ) * 100

                                    # For throughput metrics, up is good
                                    # For latency/ttft metrics, down is good
                                    if "throughput" in metric_name.lower():
                                        if change_pct > 10:
                                            trend_indicator = f"📈 +{change_pct:.1f}%"
                                        elif change_pct < -10:
                                            trend_indicator = f"⚠️ 📉 {change_pct:.1f}%"
                                        else:
                                            trend_indicator = f"➡️ {change_pct:+.1f}%"
                                    elif (
                                        "latency" in metric_name.lower()
                                        or "ttft" in metric_name.lower()
                                    ):
                                        if change_pct < -10:
                                            trend_indicator = f"📈 {change_pct:.1f}%"
                                        elif change_pct > 10:
                                            trend_indicator = f"⚠️ 📉 +{change_pct:.1f}%"
                                        else:
                                            trend_indicator = f"➡️ {change_pct:+.1f}%"
                                    else:
                                        trend_indicator = f"➡️ {change_pct:+.1f}%"

                            summary_lines.append(
                                f"| {job_name} | {metric_name} | {avg_value:.2f} | {count} | {trend_indicator} |"
                            )

                summary_lines.append("")

            with open(github_step_summary, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
                f.write("\n\n---\n\n")

            print("GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"Failed to generate GitHub Actions summary: {e}")

    def get_nightly_runs(self, days: int = 2) -> List[Dict]:
        """Get nightly test workflow runs from the last N days"""
        print(f"Fetching nightly test runs from the last {days} days...")

        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        all_runs = []

        for workflow_file in self.nightly_workflows:
            print(f"  Fetching from {workflow_file}...")
            page = 1
            per_page = 10  # Nightly runs once per day, so 10 runs covers ~10 days max
            workflow_runs = []
            max_runs_per_workflow = days * 5  # Allow up to 5 runs per day per workflow

            while len(workflow_runs) < max_runs_per_workflow:
                url = f"{self.base_url}/repos/{self.repo}/actions/runs"
                params = {
                    "workflow_id": workflow_file,
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
                    workflow_runs.extend(runs)

                    if len(runs) < per_page:
                        break

                    page += 1
                    time.sleep(0.1)

                except requests.exceptions.RequestException as e:
                    print(f"    Warning: Error fetching from {workflow_file}: {e}")
                    break

            print(f"    Fetched {len(workflow_runs)} runs from {workflow_file}")
            all_runs.extend(workflow_runs)

        print(f"Total nightly runs fetched: {len(all_runs)}")
        return all_runs

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
        """Parse performance metrics from job logs"""
        metrics = defaultdict(list)

        if not logs:
            return metrics

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

    def analyze_nightly_with_metrics(self, runs: List[Dict]) -> Dict:
        """Analyze nightly test runs including performance metrics"""
        print("Analyzing nightly test data with performance metrics...")

        # Get nightly job names from the existing job categories
        nightly_jobs = [
            # NVIDIA job names (nightly-test-nvidia.yml)
            "nightly-test-general-1-gpu-runner",
            "nightly-test-general-4-gpu-h100",
            "nightly-test-general-8-gpu-h200",
            "nightly-test-general-8-gpu-h20",
            "nightly-test-text-accuracy-2-gpu-runner",
            "nightly-test-text-perf-2-gpu-runner",
            "nightly-test-vlm-accuracy-2-gpu-runner",
            "nightly-test-vlm-perf-2-gpu-runner",
            "nightly-test-perf-4-gpu-b200",
            "nightly-test-perf-8-gpu-b200",
            # AMD job names (nightly-test-amd.yml)
            "nightly-test",
            # Intel job names (nightly-test-intel.yml)
            "placeholder",
        ]

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
                    "performance_metrics": defaultdict(list),
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
            jobs = self._get_job_details(run_id)
            for job in jobs:
                job_name = job.get("name", "Unknown")
                job_conclusion = job.get("conclusion", "unknown")
                job_id = job.get("id")
                started_at = job.get("started_at")
                completed_at = job.get("completed_at")

                # Only track nightly test jobs
                if job_name not in nightly_jobs:
                    continue

                job_stat = stats["job_stats"][job_name]
                job_stat["total"] += 1

                if job_conclusion == "success":
                    job_stat["success"] += 1

                    # For successful performance/accuracy jobs, fetch metrics
                    if (
                        "perf" in job_name.lower()
                        or "accuracy" in job_name.lower()
                        or "eval" in job_name.lower()
                    ):
                        logs = self.get_job_logs(job_id)
                        if logs:
                            metrics = self.parse_metrics_from_logs(logs, job_name)
                            for metric_name, values in metrics.items():
                                if values:
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
                del job_stat["durations"]

        return stats

    def generate_nightly_report(self, stats: Dict, output_file: str = None):
        """Generate a report for nightly test analysis"""
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
            f"{'Job Name':<50} {'Total':<8} {'Success':<8} {'Failed':<8} "
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
                f"{job_name:<50} {total:<8} {success:<8} {failure:<8} "
                f"{success_rate:>6.1f}% {avg_duration:>7.1f}m"
            )

            # Show performance metrics if available
            if job_stat.get("performance_metrics"):
                perf_metrics = job_stat["performance_metrics"]
                print(f"  Performance metrics:")

                for metric_name, metric_data in perf_metrics.items():
                    if metric_data:
                        values = [m["value"] for m in metric_data]
                        avg_value = sum(values) / len(values)
                        print(f"    - {metric_name}: {avg_value:.2f} (n={len(values)})")

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

    def generate_nightly_github_summary(self, stats: Dict):
        """Generate GitHub Actions summary for nightly test analysis"""
        try:
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print(
                    "Not running in GitHub Actions, skipping nightly summary generation"
                )
                return

            print("Generating GitHub Actions summary for Nightly Analysis...")

            summary_lines = []
            summary_lines.append("# Nightly Test Monitor Report")
            summary_lines.append("")
            summary_lines.append(
                f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            summary_lines.append("")

            # Overall statistics
            total = stats["total_runs"]
            success = stats["successful_runs"]
            failed = stats["failed_runs"]
            cancelled = stats["cancelled_runs"]

            summary_lines.append("## Overall Statistics")
            summary_lines.append("")
            summary_lines.append("| Metric | Count | Percentage |")
            summary_lines.append("|--------|-------|------------|")
            summary_lines.append(f"| Total Runs | {total} | 100% |")
            summary_lines.append(
                f"| Successful | {success} | {success/max(1,total)*100:.1f}% |"
            )
            summary_lines.append(
                f"| Failed | {failed} | {failed/max(1,total)*100:.1f}% |"
            )
            summary_lines.append(
                f"| Cancelled | {cancelled} | {cancelled/max(1,total)*100:.1f}% |"
            )
            summary_lines.append("")

            # Daily trends
            summary_lines.append("## Daily Trends")
            summary_lines.append("")
            summary_lines.append(
                "| Date | Total Runs | Success | Failed | Success Rate |"
            )
            summary_lines.append(
                "|------|------------|---------|--------|--------------|"
            )

            daily_stats = sorted(stats["daily_stats"].items(), reverse=True)[:7]
            for date, day_stats in daily_stats:
                success_rate = (day_stats["success"] / max(1, day_stats["total"])) * 100
                summary_lines.append(
                    f"| {date} | {day_stats['total']} | {day_stats['success']} | "
                    f"{day_stats['failure']} | {success_rate:.1f}% |"
                )
            summary_lines.append("")

            # Job statistics with performance metrics
            if stats["job_stats"]:
                summary_lines.append("## Job Statistics")
                summary_lines.append("")

                job_stats_sorted = sorted(
                    stats["job_stats"].items(),
                    key=lambda x: x[1]["failure"],
                    reverse=True,
                )

                for job_name, job_stat in job_stats_sorted:
                    total_job = job_stat["total"]
                    success_job = job_stat["success"]
                    failure_job = job_stat["failure"]
                    success_rate_job = (success_job / max(1, total_job)) * 100
                    avg_duration = job_stat["avg_duration_minutes"]

                    summary_lines.append(f"### {job_name}")
                    summary_lines.append("")
                    summary_lines.append(
                        f"**Stats:** {total_job} runs | {success_job} success ({success_rate_job:.1f}%) | "
                        f"{failure_job} failed | Avg duration: {avg_duration:.1f}m"
                    )
                    summary_lines.append("")

                    # Performance metrics
                    if job_stat.get("performance_metrics"):
                        summary_lines.append("**Performance Metrics:**")
                        summary_lines.append("")
                        summary_lines.append("| Metric | Avg Value | Samples |")
                        summary_lines.append("|--------|-----------|---------|")

                        for metric_name, metric_data in job_stat[
                            "performance_metrics"
                        ].items():
                            if metric_data:
                                values = [m["value"] for m in metric_data]
                                avg_value = sum(values) / len(values)
                                summary_lines.append(
                                    f"| {metric_name} | {avg_value:.2f} | {len(values)} |"
                                )
                        summary_lines.append("")

                    # Recent failures
                    if job_stat["recent_failures"]:
                        summary_lines.append("**Recent Failures:**")
                        for failure in job_stat["recent_failures"][:3]:
                            summary_lines.append(
                                f"- [Run #{failure['run_number']}]({failure['run_url']})"
                            )
                        summary_lines.append("")

            with open(github_step_summary, "a", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))
                f.write("\n\n---\n\n")

            print("GitHub Actions nightly summary generated successfully")

        except Exception as e:
            print(f"Failed to generate nightly GitHub Actions summary: {e}")

    def detect_nightly_regressions(self, stats: Dict) -> List[Dict]:
        """Detect regressions in nightly tests"""
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
            print("=" * 80)

        return regressions


def main():
    parser = argparse.ArgumentParser(description="SGLang CI Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--mode",
        choices=["ci", "nightly"],
        default="ci",
        help="Analysis mode: 'ci' for general CI analysis, 'nightly' for nightly test monitoring (default: ci)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of runs to analyze (for ci mode, default: 100)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of days to analyze (for nightly mode, default: 2)",
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed stats (JSON)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Filter runs by branch (default: None - all branches). Specify branch name to filter.",
    )

    args = parser.parse_args()

    analyzer = SGLangCIAnalyzer(args.token)

    try:
        if args.mode == "nightly":
            # Nightly test monitoring mode
            runs = analyzer.get_nightly_runs(days=args.days)

            if not runs:
                print("No nightly test runs found in the specified time period.")
                sys.exit(1)

            stats = analyzer.analyze_nightly_with_metrics(runs)
            analyzer.generate_nightly_report(stats, args.output)
            analyzer.generate_nightly_github_summary(stats)
            regressions = analyzer.detect_nightly_regressions(stats)

            # Report regressions but don't stop the monitor
            if regressions:
                print("\n⚠️  Regressions detected - see report above")
            else:
                print("\n✓ No significant regressions detected")
            sys.exit(0)

        else:
            # Regular CI analysis mode
            branch = args.branch if args.branch else None
            runs = analyzer.get_recent_runs(args.limit, branch)

            if not runs:
                print("No CI run data found")
                return

            stats = analyzer.analyze_ci_failures(runs)
            analyzer.generate_report(stats)

            output_file = args.output or "ci_analysis.json"
            analyzer.save_detailed_report(stats, output_file)
            analyzer.generate_github_summary(stats)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
