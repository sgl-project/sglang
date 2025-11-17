#!/usr/bin/env python3
"""
SGLang CI Performance Analyzer - Simplified Version
Collect performance data based on actual log format
"""

import argparse
import base64
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib import rcParams


class SGLangPerfAnalyzer:
    """SGLang CI Performance Analyzer"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-Perf-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Performance test job names
        self.performance_jobs = [
            "performance-test-1-gpu-part-1",
            "performance-test-1-gpu-part-2",
            "performance-test-2-gpu",
        ]

        # Strictly match tests and metrics shown in the images
        self.target_tests_and_metrics = {
            "performance-test-1-gpu-part-1": {
                "test_bs1_default": ["output_throughput_token_s"],
                "test_online_latency_default": ["median_e2e_latency_ms"],
                "test_offline_throughput_default": ["output_throughput_token_s"],
                "test_offline_throughput_non_stream_small_batch_size": [
                    "output_throughput_token_s"
                ],
                "test_online_latency_eagle": ["median_e2e_latency_ms", "accept_length"],
                "test_lora_online_latency": ["median_e2e_latency_ms", "median_ttft_ms"],
                "test_lora_online_latency_with_concurrent_adapter_updates": [
                    "median_e2e_latency_ms",
                    "median_ttft_ms",
                ],
            },
            "performance-test-1-gpu-part-2": {
                "test_offline_throughput_without_radix_cache": [
                    "output_throughput_token_s"
                ],
                "test_offline_throughput_with_triton_attention_backend": [
                    "output_throughput_token_s"
                ],
                "test_offline_throughput_default_fp8": ["output_throughput_token_s"],
                "test_vlm_offline_throughput": ["output_throughput_token_s"],
                "test_vlm_online_latency": ["median_e2e_latency_ms"],
            },
            "performance-test-2-gpu": {
                "test_moe_tp2_bs1": ["output_throughput_token_s"],
                "test_torch_compile_tp2_bs1": ["output_throughput_token_s"],
                "test_moe_offline_throughput_default": ["output_throughput_token_s"],
                "test_moe_offline_throughput_without_radix_cache": [
                    "output_throughput_token_s"
                ],
                "test_pp_offline_throughput_default_decode": [
                    "output_throughput_token_s"
                ],
                "test_pp_long_context_prefill": ["input_throughput_token_s"],
            },
        }

        # Performance metric patterns - only keep metrics needed in images
        self.perf_patterns = {
            # Key metrics shown in images
            "output_throughput_token_s": r"Output token throughput \(tok/s\):\s*([\d.]+)",
            "Output_throughput_token_s": r"Output throughput:\s*([\d.]+)\s*token/s",
            "median_e2e_latency_ms": r"Median E2E Latency \(ms\):\s*([\d.]+)",
            "median_ttft_ms": r"Median TTFT \(ms\):\s*([\d.]+)",
            "accept_length": r"Accept length:\s*([\d.]+)",
            "input_throughput_token_s": r"Input token throughput \(tok/s\):\s*([\d.]+)",
        }

        # Pre-compile regex patterns for better performance
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.perf_patterns.items()
        }

        # Pre-compile test pattern
        self.test_pattern = re.compile(
            r"python3 -m unittest (test_bench_\w+\.TestBench\w+\.test_\w+)"
        )

        # Setup matplotlib fonts and styles
        self._setup_matplotlib()

        # GitHub data repository settings
        self.data_repo = "sglang-bot/sglang-ci-data"
        self.data_branch = "main"

    def _setup_matplotlib(self):
        """Setup matplotlib fonts and styles"""
        # Set fonts
        rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
        rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

        # Set chart styles
        plt.style.use("default")
        rcParams["figure.figsize"] = (12, 6)
        rcParams["font.size"] = 10
        rcParams["axes.grid"] = True
        rcParams["grid.alpha"] = 0.3

    def get_recent_runs(
        self, limit: int = 100, start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """Get recent CI run data with multiple collection strategies"""

        # If date range is specified, get all data in that range
        if start_date or end_date:
            return self._get_date_range_runs(start_date, end_date)

        print(f"Getting PR Test runs (limit: {limit})...")

        # Use sampling strategy if limit >= 500, otherwise use sequential
        if limit >= 500:
            print(f"Using uniform sampling for {limit} runs to cover ~30 days...")
            return self._get_sampled_runs(limit)
        else:
            return self._get_sequential_runs(limit)

    def _get_sequential_runs(self, limit: int) -> List[Dict]:
        """Original sequential method for smaller limits"""
        print(f"Using sequential sampling for {limit} runs...")

        pr_test_runs = []
        page = 1
        per_page = 100

        while len(pr_test_runs) < limit:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": per_page, "page": page}

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                # Filter PR Test runs
                current_pr_tests = [
                    run for run in data["workflow_runs"] if run.get("name") == "PR Test"
                ]

                # Add to result list, but not exceed limit
                for run in current_pr_tests:
                    if len(pr_test_runs) < limit:
                        pr_test_runs.append(run)
                    else:
                        break

                print(f"Got {len(pr_test_runs)} PR test runs...")

                # Exit if no more data on this page or reached limit
                if len(data["workflow_runs"]) < per_page or len(pr_test_runs) >= limit:
                    break

                page += 1
                time.sleep(0.1)  # Avoid API rate limiting

            except requests.exceptions.RequestException as e:
                print(f"Error getting CI data: {e}")
                break

        return pr_test_runs

    def _get_sampled_runs(self, limit: int) -> List[Dict]:
        """Uniform sampling method for 30-day coverage"""
        from datetime import datetime, timedelta

        # Uniform sampling across 30 days
        sampled_runs = self._sample_time_period(limit, days_back=30, uniform=True)

        print(
            f"Sampled {len(sampled_runs)} runs from 30-day period (requested: {limit})"
        )
        return sampled_runs

    def _sample_time_period(
        self,
        target_samples: int,
        days_back: int,
        skip_recent_days: int = 0,
        uniform: bool = False,
    ) -> List[Dict]:
        """Sample runs from a specific time period"""
        from datetime import datetime, timedelta

        # Calculate time range
        end_time = datetime.utcnow() - timedelta(days=skip_recent_days)
        start_time = end_time - timedelta(days=days_back - skip_recent_days)

        sampling_type = "uniform" if uniform else "systematic"
        print(
            f"  {sampling_type.title()} sampling {target_samples} runs from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
        )

        collected_runs = []
        page = 1
        per_page = 100
        total_in_period = 0

        while True:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": per_page, "page": page}

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                period_runs = []
                for run in data["workflow_runs"]:
                    if run.get("name") != "PR Test":
                        continue

                    created_at = run.get("created_at", "")
                    if created_at:
                        try:
                            run_time = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if start_time <= run_time <= end_time:
                                period_runs.append(run)
                                total_in_period += 1
                        except:
                            continue

                collected_runs.extend(period_runs)

                # Progress indicator every 5 pages
                if page % 5 == 0:
                    print(
                        f"    Page {page}: Found {total_in_period} runs in target period, collected {len(collected_runs)} total"
                    )

                # Check if we've gone past our time window
                if data["workflow_runs"]:
                    last_run_time_str = data["workflow_runs"][-1].get("created_at", "")
                    if last_run_time_str:
                        try:
                            last_run_time = datetime.fromisoformat(
                                last_run_time_str.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if last_run_time < start_time:
                                print(f"  Reached time boundary at page {page}")
                                break
                        except:
                            pass

                if len(data["workflow_runs"]) < per_page:
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"  Error getting data for time period: {e}")
                break

        print(
            f"  Found {total_in_period} runs in time period, collected {len(collected_runs)} for sampling"
        )

        # Debug: Show time range of collected data
        if collected_runs:
            collected_runs_sorted = sorted(
                collected_runs, key=lambda x: x.get("created_at", "")
            )
            earliest = (
                collected_runs_sorted[0].get("created_at", "")[:10]
                if collected_runs_sorted
                else "N/A"
            )
            latest = (
                collected_runs_sorted[-1].get("created_at", "")[:10]
                if collected_runs_sorted
                else "N/A"
            )
            print(f"  Collected data spans from {earliest} to {latest}")

        # Sample from collected runs
        if len(collected_runs) <= target_samples:
            return collected_runs

        if uniform:
            # Uniform sampling: sort by time and select evenly distributed samples
            collected_runs.sort(key=lambda x: x.get("created_at", ""))
            step = len(collected_runs) / target_samples
            sampled_runs = []

            for i in range(target_samples):
                index = int(i * step)
                if index < len(collected_runs):
                    sampled_runs.append(collected_runs[index])
        else:
            # Systematic sampling for even distribution
            step = len(collected_runs) / target_samples
            sampled_runs = []

            for i in range(target_samples):
                index = int(i * step)
                if index < len(collected_runs):
                    sampled_runs.append(collected_runs[index])

        print(
            f"  Sampled {len(sampled_runs)} runs from {len(collected_runs)} available"
        )

        # Debug: Show time range of sampled data
        if sampled_runs:
            sampled_runs_sorted = sorted(
                sampled_runs, key=lambda x: x.get("created_at", "")
            )
            earliest = (
                sampled_runs_sorted[0].get("created_at", "")[:10]
                if sampled_runs_sorted
                else "N/A"
            )
            latest = (
                sampled_runs_sorted[-1].get("created_at", "")[:10]
                if sampled_runs_sorted
                else "N/A"
            )
            print(f"  Sampled data spans from {earliest} to {latest}")

        return sampled_runs

    def _get_date_range_runs(
        self, start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """Get all CI runs within specified date range"""
        from datetime import datetime, timedelta

        # Parse dates
        if start_date:
            try:
                start_time = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Invalid start_date format. Use YYYY-MM-DD, got: {start_date}"
                )
        else:
            # Default to 30 days ago if no start date
            start_time = datetime.utcnow() - timedelta(days=30)

        if end_date:
            try:
                end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(
                    days=1
                )  # Include the end date
            except ValueError:
                raise ValueError(
                    f"Invalid end_date format. Use YYYY-MM-DD, got: {end_date}"
                )
        else:
            # Default to now if no end date
            end_time = datetime.utcnow()

        # Validate date range
        if start_time >= end_time:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )

        print(
            f"Getting ALL CI runs from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
        )

        collected_runs = []
        page = 1
        per_page = 100
        total_in_period = 0

        while True:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": per_page, "page": page}

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                # Filter runs in date range and PR Test runs
                period_runs = []
                for run in data["workflow_runs"]:
                    if run.get("name") != "PR Test":
                        continue

                    created_at = run.get("created_at", "")
                    if created_at:
                        try:
                            run_time = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if start_time <= run_time <= end_time:
                                period_runs.append(run)
                                total_in_period += 1
                        except:
                            continue

                collected_runs.extend(period_runs)

                # Progress indicator every 5 pages
                if page % 5 == 0:
                    print(
                        f"    Page {page}: Found {total_in_period} runs in date range, collected {len(collected_runs)} total"
                    )

                # Check if we've gone past our time window
                if data["workflow_runs"]:
                    last_run_time_str = data["workflow_runs"][-1].get("created_at", "")
                    if last_run_time_str:
                        try:
                            last_run_time = datetime.fromisoformat(
                                last_run_time_str.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if last_run_time < start_time:
                                print(f"  Reached time boundary at page {page}")
                                break
                        except:
                            pass

                if len(data["workflow_runs"]) < per_page:
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"  Error getting data for date range: {e}")
                break

        print(
            f"Found {total_in_period} runs in date range {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
        )

        # Sort by creation time (newest first)
        collected_runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return collected_runs

    def get_job_logs(self, run_id: int, job_name: str) -> Optional[str]:
        """Get logs for specific job with early exit optimization"""
        try:
            # First get job list
            jobs_url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
            response = self.session.get(jobs_url)
            response.raise_for_status()
            jobs_data = response.json()

            # Find matching job with early exit
            target_job = None
            for job in jobs_data.get("jobs", []):
                if job_name in job.get("name", ""):
                    # Early exit if job failed or was skipped
                    if job.get("conclusion") not in ["success", "neutral"]:
                        return None
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
            # Reduce verbose error logging for common failures
            if "404" not in str(e):
                print(f"Failed to get job {job_name} logs: {e}")
            return None

    def get_all_job_logs_parallel(self, run_id: int) -> Dict[str, Optional[str]]:
        """Get logs for all performance jobs in parallel"""

        def fetch_job_logs(job_name: str) -> tuple[str, Optional[str]]:
            """Fetch logs for a single job"""
            logs = self.get_job_logs(run_id, job_name)
            return job_name, logs

        results = {}
        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:  # Increased concurrent requests
            # Submit all job log requests
            future_to_job = {
                executor.submit(fetch_job_logs, job_name): job_name
                for job_name in self.performance_jobs
            }

            # Collect results as they complete
            for future in as_completed(future_to_job):
                job_name, logs = future.result()
                results[job_name] = logs

        return results

    def parse_performance_data(
        self, log_content: str, job_name: str
    ) -> Dict[str, Dict[str, str]]:
        """Parse specified performance data from logs"""
        if not log_content:
            return {}

        test_data = {}

        # Get target tests for current job
        target_tests = self.target_tests_and_metrics.get(job_name, {})
        if not target_tests:
            return test_data

        # Find all unittest tests using pre-compiled pattern
        test_matches = self.test_pattern.findall(log_content)

        for test_match in test_matches:
            test_name = test_match.split(".")[-1]  # Extract test name

            # Only process target tests
            if test_name not in target_tests:
                continue

            # Find performance data after this test
            test_section = self._extract_test_section(log_content, test_match)
            if test_section:
                # Only find metrics needed for this test
                target_metrics = target_tests[test_name]
                perf_data = {}

                for metric_name in target_metrics:
                    if metric_name in self.compiled_patterns:
                        compiled_pattern = self.compiled_patterns[metric_name]
                        matches = compiled_pattern.findall(test_section)
                        if matches:
                            perf_data[metric_name] = matches[-1]  # Take the last match

                if perf_data:
                    test_data[test_name] = perf_data

        return test_data

    def _extract_test_section(self, log_content: str, test_pattern: str) -> str:
        """Extract log section for specific test"""
        lines = log_content.split("\n")
        test_start = -1
        test_end = len(lines)

        # Find test start position
        for i, line in enumerate(lines):
            if test_pattern in line:
                test_start = i
                break

        if test_start == -1:
            return ""

        # Find test end position (next test start or major separator)
        for i in range(test_start + 1, len(lines)):
            line = lines[i]
            if (
                "python3 -m unittest" in line and "test_" in line
            ) or "##[group]" in line:
                test_end = i
                break

        return "\n".join(lines[test_start:test_end])

    def collect_performance_data(self, runs: List[Dict]) -> Dict[str, List[Dict]]:
        """Collect all performance data"""
        print("Starting performance data collection...")

        # Create data list for each test
        all_test_data = {}

        total_runs = len(runs)
        for i, run in enumerate(runs, 1):
            if not isinstance(run, dict):
                print(f"  Warning: run #{i} is not a dict, skipping.")
                continue

            run_info = {
                "run_number": run.get("run_number"),
                "created_at": run.get("created_at"),
                "head_sha": (run.get("head_sha") or "")[:8],
                "author": "Unknown",
                "pr_number": None,
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }
            head_commit = run.get("head_commit", {})
            if isinstance(head_commit, dict):
                run_info["author"] = head_commit.get("author", {}).get(
                    "name", "Unknown"
                )

            # Extract PR number
            pull_requests = run.get("pull_requests", [])
            if pull_requests:
                run_info["pr_number"] = pull_requests[0].get("number")

            # Get all job logs in parallel
            all_job_logs = self.get_all_job_logs_parallel(run.get("id"))

            # Process each performance test job
            for job_name, logs in all_job_logs.items():
                if not logs:
                    continue

                # Parse performance data
                test_results = self.parse_performance_data(logs, job_name)

                for test_name, perf_data in test_results.items():
                    # Create full test name including job info
                    full_test_name = f"{job_name}_{test_name}"

                    if full_test_name not in all_test_data:
                        all_test_data[full_test_name] = []

                    test_entry = {**run_info, **perf_data}
                    all_test_data[full_test_name].append(test_entry)
                    print(
                        f"    Found {test_name} performance data: {list(perf_data.keys())}"
                    )

            time.sleep(0.2)
        return all_test_data

    def generate_performance_tables(
        self, test_data: Dict[str, List[Dict]], output_dir: str = "performance_tables"
    ):
        """Generate performance data tables"""
        print(f"Generating performance tables to directory: {output_dir}")

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for each job
        job_dirs = {}
        for job_name in self.performance_jobs:
            job_dir = os.path.join(output_dir, f"{job_name}_summary")
            os.makedirs(job_dir, exist_ok=True)
            job_dirs[job_name] = job_dir

        # Generate table for each test
        for full_test_name, data_list in test_data.items():
            if not data_list:
                continue

            # Determine which job this test belongs to
            job_name = None
            test_name = full_test_name
            for job in self.performance_jobs:
                if full_test_name.startswith(job):
                    job_name = job
                    test_name = full_test_name[len(job) + 1 :]  # Remove job prefix
                    break

            if not job_name:
                continue

            job_dir = job_dirs[job_name]
            table_file = os.path.join(job_dir, f"{test_name}.csv")

            # Generate CSV table
            self._write_csv_table(table_file, test_name, data_list)

            # Generate corresponding chart
            print(f"    Generating chart for {test_name}...")
            self._generate_chart(table_file, test_name, data_list, job_dir)

        print("Performance tables and charts generation completed!")

    def _write_csv_table(self, file_path: str, test_name: str, data_list: List[Dict]):
        """Write CSV table"""
        if not data_list:
            return

        # Get all possible columns
        all_columns = set()
        for entry in data_list:
            all_columns.update(entry.keys())

        # Define column order
        base_columns = ["created_at", "run_number", "pr_number", "author", "head_sha"]
        perf_columns = [col for col in all_columns if col not in base_columns + ["url"]]
        columns = base_columns + sorted(perf_columns) + ["url"]

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(columns)

            # Write data rows
            for entry in sorted(
                data_list, key=lambda x: x.get("created_at", ""), reverse=True
            ):
                row = []
                for col in columns:
                    value = entry.get(col, "")
                    if col == "created_at" and value:
                        # Format time to consistent format
                        try:
                            # Handle ISO 8601 format: "2025-09-26T11:16:40Z"
                            if "T" in value and "Z" in value:
                                dt = datetime.fromisoformat(
                                    value.replace("Z", "+00:00")
                                )
                                value = dt.strftime("%Y-%m-%d %H:%M")
                            # If already in desired format, keep it
                            elif len(value) == 16 and " " in value:
                                # Validate format
                                datetime.strptime(value, "%Y-%m-%d %H:%M")
                            else:
                                # Try to parse and reformat
                                dt = datetime.fromisoformat(value)
                                value = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            # If all parsing fails, keep original value
                            pass
                    elif col == "pr_number" and value:
                        value = f"#{value}"
                    row.append(str(value))
                writer.writerow(row)

        print(f"  Generated table: {file_path} ({len(data_list)} records)")

    def _generate_chart(
        self, csv_file_path: str, test_name: str, data_list: List[Dict], output_dir: str
    ):
        """Generate corresponding time series charts for tables"""
        print(
            f"      Starting chart generation for {test_name} with {len(data_list)} data points"
        )

        if not data_list or len(data_list) < 2:
            print(
                f"      Skipping chart for {test_name}: insufficient data ({len(data_list) if data_list else 0} records)"
            )
            return

        try:
            # Prepare data
            timestamps = []
            metrics_data = {}

            # Get performance metric columns (exclude basic info columns)
            base_columns = {
                "created_at",
                "run_number",
                "pr_number",
                "author",
                "head_sha",
                "url",
            }
            perf_metrics = []

            for entry in data_list:
                for key in entry.keys():
                    if key not in base_columns and key not in perf_metrics:
                        perf_metrics.append(key)

            if not perf_metrics:
                print(
                    f"      Skipping chart for {test_name}: no performance metrics found"
                )
                return

            print(f"      Found performance metrics: {perf_metrics}")

            # Parse data
            for entry in data_list:
                # Parse time
                try:
                    time_str = entry.get("created_at", "")
                    if time_str:
                        # Handle different time formats
                        timestamp = None

                        # Try ISO 8601 format first (from GitHub API): "2025-09-26T11:16:40Z"
                        if "T" in time_str and "Z" in time_str:
                            try:
                                # Parse and convert to naive datetime (remove timezone info)
                                dt_with_tz = datetime.fromisoformat(
                                    time_str.replace("Z", "+00:00")
                                )
                                timestamp = dt_with_tz.replace(tzinfo=None)
                            except:
                                # Fallback for older Python versions
                                timestamp = datetime.strptime(
                                    time_str, "%Y-%m-%dT%H:%M:%SZ"
                                )

                        # Try CSV format: "2025-09-26 08:43"
                        elif " " in time_str and len(time_str) == 16:
                            timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

                        # Try other common formats
                        else:
                            formats_to_try = [
                                "%Y-%m-%d %H:%M:%S",
                                "%Y-%m-%dT%H:%M:%S",
                                "%Y-%m-%d",
                            ]
                            for fmt in formats_to_try:
                                try:
                                    timestamp = datetime.strptime(time_str, fmt)
                                    break
                                except:
                                    continue

                        if timestamp:
                            timestamps.append(timestamp)

                            # Collect metric data
                            for metric in perf_metrics:
                                if metric not in metrics_data:
                                    metrics_data[metric] = []

                                value = entry.get(metric, "")
                                try:
                                    numeric_value = float(value)
                                    metrics_data[metric].append(numeric_value)
                                except:
                                    metrics_data[metric].append(None)
                        else:
                            print(
                                f"      Failed to parse timestamp format: '{time_str}'"
                            )

                except Exception as e:
                    print(f"      Error processing entry: {e}")
                    continue

            if not timestamps:
                print(
                    f"      Skipping chart for {test_name}: no valid timestamps found"
                )
                return

            print(f"      Parsed {len(timestamps)} timestamps")

            # Sort by time
            sorted_data = sorted(
                zip(timestamps, *[metrics_data[m] for m in perf_metrics])
            )
            timestamps = [item[0] for item in sorted_data]
            for i, metric in enumerate(perf_metrics):
                metrics_data[metric] = [item[i + 1] for item in sorted_data]

            # Create chart for each metric
            for metric in perf_metrics:
                values = metrics_data[metric]
                valid_data = [
                    (t, v) for t, v in zip(timestamps, values) if v is not None
                ]

                if len(valid_data) < 2:
                    print(
                        f"      Skipping chart for {test_name}_{metric}: insufficient valid data ({len(valid_data)} points)"
                    )
                    continue

                valid_timestamps, valid_values = zip(*valid_data)

                # Create chart
                plt.figure(figsize=(12, 6))
                plt.plot(
                    valid_timestamps,
                    valid_values,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                )

                # Set title and labels
                title = f"{test_name} - {self._format_metric_name(metric)}"
                plt.title(title, fontsize=14, fontweight="bold")
                plt.xlabel("Time", fontsize=12)
                plt.ylabel(self._get_metric_unit(metric), fontsize=12)

                # Format x-axis
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                plt.gca().xaxis.set_major_locator(
                    mdates.HourLocator(interval=max(1, len(valid_timestamps) // 10))
                )
                plt.xticks(rotation=45)

                # Add grid
                plt.grid(True, alpha=0.3)

                # Adjust layout
                plt.tight_layout()

                # Save chart
                chart_filename = f"{test_name}_{metric}.png"
                chart_path = os.path.join(output_dir, chart_filename)
                plt.savefig(chart_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"      Generated chart: {chart_path}")

        except Exception as e:
            print(f"      Failed to generate chart for {test_name}: {e}")
            import traceback

            traceback.print_exc()

    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display"""
        name_mapping = {
            "output_throughput_token_s": "Output Throughput",
            "median_e2e_latency_ms": "Median E2E Latency",
            "median_ttft_ms": "Median TTFT",
            "accept_length": "Accept Length",
            "input_throughput_token_s": "Input Throughput",
        }
        return name_mapping.get(metric, metric)

    def _get_metric_unit(self, metric: str) -> str:
        """Get metric unit"""
        if "throughput" in metric and "token_s" in metric:
            return "token/s"
        elif "latency" in metric and "ms" in metric:
            return "ms"
        elif "accept_length" in metric:
            return "length"
        else:
            return "value"

    def generate_summary_report(self, test_data: Dict[str, List[Dict]]):
        """Generate summary report"""
        print("\n" + "=" * 60)
        print("SGLang CI Performance Data Collection Report")
        print("=" * 60)

        total_tests = len([test for test, data in test_data.items() if data])
        total_records = sum(len(data) for data in test_data.values())

        print(f"\nOverall Statistics:")
        print(f"  Number of tests collected: {total_tests}")
        print(f"  Total records: {total_records}")

        print(f"\nStatistics by job:")
        for job_name in self.performance_jobs:
            job_tests = [test for test in test_data.keys() if test.startswith(job_name)]
            job_records = sum(len(test_data[test]) for test in job_tests)
            print(f"  {job_name}: {len(job_tests)} tests, {job_records} records")

            for test in job_tests:
                data = test_data[test]
                test_short_name = test[len(job_name) + 1 :]
                print(f"    - {test_short_name}: {len(data)} records")

        print("\n" + "=" * 60)

    def upload_file_to_github(
        self, file_path: str, github_path: str, commit_message: str
    ) -> bool:
        """Upload a file to GitHub repository with retry logic"""
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Read file content
                with open(file_path, "rb") as f:
                    content = f.read()

                # Encode content to base64
                content_encoded = base64.b64encode(content).decode("utf-8")

                # Check if file exists to get SHA
                check_url = (
                    f"{self.base_url}/repos/{self.data_repo}/contents/{github_path}"
                )
                check_response = self.session.get(check_url)

                sha = None
                if check_response.status_code == 200:
                    sha = check_response.json().get("sha")

                # Prepare upload data
                upload_data = {
                    "message": commit_message,
                    "content": content_encoded,
                    "branch": self.data_branch,
                }

                if sha:
                    upload_data["sha"] = sha

                # Upload file
                response = self.session.put(check_url, json=upload_data)

                if response.status_code in [200, 201]:
                    print(f"    ✅ Uploaded: {github_path}")
                    return True
                elif response.status_code == 403:
                    retry_count += 1
                    wait_time = min(2**retry_count, 30)
                    print(
                        f"    ⚠️ Upload forbidden (403) for {github_path}, retrying in {wait_time}s... (attempt {retry_count}/{max_retries})"
                    )
                    if retry_count >= max_retries:
                        print(
                            f"    ❌ Failed to upload {github_path} after {max_retries} attempts (403 Forbidden)"
                        )
                        return False
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = min(2**retry_count, 30)
                print(
                    f"    ⚠️ Upload error for {github_path} (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    print(
                        f"    ❌ Failed to upload {github_path} after {max_retries} attempts: {e}"
                    )
                    return False
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"    ❌ Failed to upload {github_path}: {e}")
                return False

        return False

    def upload_performance_data_to_github(self, output_dir: str):
        """Upload performance_tables to GitHub with original structure"""
        print("📤 Uploading performance data to GitHub...")

        # Check if target repository exists with retry logic
        repo_url = f"{self.base_url}/repos/{self.data_repo}"
        max_retries = 30
        retry_count = 0

        print(f"🔍 Checking repository access to {self.data_repo}...")

        while retry_count < max_retries:
            try:
                repo_response = self.session.get(repo_url)

                if repo_response.status_code == 200:
                    print(f"✅ Repository {self.data_repo} is accessible")
                    break
                elif repo_response.status_code == 404:
                    print(
                        f"❌ Repository {self.data_repo} does not exist or is not accessible"
                    )
                    print("   Please ensure:")
                    print("   1. The repository exists")
                    print("   2. Your GitHub token has access to this repository")
                    print("   3. Your token has 'contents:write' permission")
                    return
                elif repo_response.status_code == 403:
                    retry_count += 1
                    wait_time = min(2**retry_count, 60)  # Exponential backoff, max 60s
                    print(
                        f"⚠️ Repository access forbidden (403), retrying in {wait_time}s... (attempt {retry_count}/{max_retries})"
                    )
                    if retry_count >= max_retries:
                        print(
                            f"❌ Failed to access repository after {max_retries} attempts"
                        )
                        print("   This might be due to:")
                        print("   1. GitHub API rate limiting")
                        print("   2. Token permissions issue")
                        print("   3. Repository access restrictions")
                        return
                    time.sleep(wait_time)
                else:
                    retry_count += 1
                    wait_time = min(2**retry_count, 60)
                    print(
                        f"⚠️ Repository access failed with status {repo_response.status_code}, retrying in {wait_time}s... (attempt {retry_count}/{max_retries})"
                    )
                    if retry_count >= max_retries:
                        print(
                            f"❌ Failed to access repository {self.data_repo} after {max_retries} attempts"
                        )
                        return
                    time.sleep(wait_time)

            except Exception as e:
                retry_count += 1
                wait_time = min(2**retry_count, 60)
                print(
                    f"⚠️ Error checking repository (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count >= max_retries:
                    print(
                        f"❌ Failed to check repository after {max_retries} attempts: {e}"
                    )
                    return
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # Generate timestamp for this upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        uploaded_count = 0

        # Upload all files maintaining original structure
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)

                # Keep original directory structure
                rel_path = os.path.relpath(local_path, output_dir)
                github_path = f"performance_data/{timestamp}/{rel_path}".replace(
                    "\\", "/"
                )

                # Upload file
                commit_msg = f"Add performance data: {rel_path} ({timestamp})"
                if self.upload_file_to_github(local_path, github_path, commit_msg):
                    uploaded_count += 1

        print(f"📤 Uploaded {uploaded_count} files to GitHub")

        # Print access info
        base_url = f"https://github.com/{self.data_repo}/tree/{self.data_branch}/performance_data/{timestamp}"
        print(f"🔗 View uploaded data at: {base_url}")

        # Generate GitHub Actions summary
        self._generate_github_summary(output_dir, timestamp)

    def _generate_github_summary(self, output_dir: str, timestamp: str):
        """Generate GitHub Actions summary with performance data"""
        try:
            # Check if running in GitHub Actions
            github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if not github_step_summary:
                print("ℹ️  Not running in GitHub Actions, skipping summary generation")
                return

            print("📊 Generating GitHub Actions summary...")

            # Collect all CSV and PNG files
            csv_files = []
            png_files = []

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, output_dir)

                    if file.endswith(".csv"):
                        csv_files.append((file_path, rel_path))
                    elif file.endswith(".png"):
                        png_files.append((file_path, rel_path))

            # Sort files by job and test name
            csv_files.sort(key=lambda x: x[1])
            png_files.sort(key=lambda x: x[1])

            # Generate markdown summary
            summary_lines = []
            summary_lines.append("# 📊 SGLang Performance Analysis Report")
            summary_lines.append("")
            summary_lines.append(f"**Analysis Timestamp:** {timestamp}")
            summary_lines.append(f"**Total CSV Files:** {len(csv_files)}")
            summary_lines.append(f"**Total Chart Files:** {len(png_files)}")
            summary_lines.append("")

            # GitHub data repository link
            base_url = f"https://github.com/{self.data_repo}/tree/{self.data_branch}/performance_data/{timestamp}"
            summary_lines.append(f"🔗 **[View All Data on GitHub]({base_url})**")
            summary_lines.append("")

            # Group by job
            job_groups = {}
            for csv_path, rel_path in csv_files:
                # Extract job name from path: job_summary/test_name.csv
                parts = rel_path.split("/")
                if len(parts) >= 2:
                    job_name = parts[0].replace("_summary", "")
                    test_name = parts[1].replace(".csv", "")

                    if job_name not in job_groups:
                        job_groups[job_name] = []
                    job_groups[job_name].append((csv_path, test_name, rel_path))

            # Generate summary for each job
            for job_name in sorted(job_groups.keys()):
                summary_lines.append(f"## 🚀 {job_name}")
                summary_lines.append("")

                tests = job_groups[job_name]
                tests.sort(key=lambda x: x[1])  # Sort by test name

                for csv_path, test_name, rel_path in tests:
                    summary_lines.append(f"### 📈 {test_name}")

                    # Add CSV data preview
                    try:
                        with open(csv_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # Has header and data
                                summary_lines.append("")
                                summary_lines.append("**Recent Performance Data:**")
                                summary_lines.append("")

                                # Show header
                                header = lines[0].strip()
                                summary_lines.append(
                                    f"| {' | '.join(header.split(','))} |"
                                )
                                summary_lines.append(
                                    f"| {' | '.join(['---'] * len(header.split(',')))} |"
                                )

                                # Show most recent 5 records (CSV is already sorted newest first)
                                data_lines = lines[1:]
                                for line in data_lines[
                                    :5
                                ]:  # Take first 5 lines (most recent)
                                    if line.strip():
                                        summary_lines.append(
                                            f"| {' | '.join(line.strip().split(','))} |"
                                        )

                                summary_lines.append("")
                    except Exception as e:
                        summary_lines.append(f"*Error reading CSV data: {e}*")
                        summary_lines.append("")

                    # Add chart image if exists
                    test_prefix = rel_path.replace(".csv", "")
                    matching_charts = [
                        (png_path, png_rel)
                        for png_path, png_rel in png_files
                        if png_rel.startswith(test_prefix)
                    ]

                    for png_path, chart_rel_path in matching_charts:
                        chart_url = f"https://github.com/{self.data_repo}/raw/{self.data_branch}/performance_data/{timestamp}/{chart_rel_path}"
                        # Extract metric name from filename: test_name_metric_name.png
                        filename = os.path.basename(chart_rel_path)
                        metric_name = filename.replace(f"{test_name}_", "").replace(
                            ".png", ""
                        )
                        summary_lines.append(
                            f"**{self._format_metric_name(metric_name)} Trend:**"
                        )
                        summary_lines.append("")
                        summary_lines.append(
                            f"![{test_name}_{metric_name}]({chart_url})"
                        )
                        summary_lines.append("")

                    summary_lines.append("---")
                    summary_lines.append("")

            # Write summary to GitHub Actions (append mode to preserve CI Analysis report)
            with open(github_step_summary, "a", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            print("✅ GitHub Actions summary generated successfully")

        except Exception as e:
            print(f"❌ Failed to generate GitHub Actions summary: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="SGLang CI Performance Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of runs to analyze (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="performance_tables",
        help="Output directory (default: performance_tables)",
    )
    parser.add_argument(
        "--upload-to-github",
        action="store_true",
        help="Upload results to sglang-bot/sglang-ci-data repository",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for date range query (YYYY-MM-DD format). When specified with --end-date, gets ALL runs in range.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for date range query (YYYY-MM-DD format). When specified with --start-date, gets ALL runs in range.",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = SGLangPerfAnalyzer(args.token)

    try:
        # Get CI run data
        runs = analyzer.get_recent_runs(args.limit, args.start_date, args.end_date)

        if not runs:
            print("No CI run data found")
            return

        # Collect performance data
        test_data = analyzer.collect_performance_data(runs)

        # Generate performance tables
        analyzer.generate_performance_tables(test_data, args.output_dir)

        # Upload to GitHub if requested
        if args.upload_to_github:
            analyzer.upload_performance_data_to_github(args.output_dir)

        # Generate summary report
        analyzer.generate_summary_report(test_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
