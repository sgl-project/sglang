#!/usr/bin/env python3
"""
SGLang CI Performance Analyzer - Simplified Version
Collect performance data based on actual log format
"""

import argparse
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

    def get_recent_runs(self, limit: int = 100) -> List[Dict]:
        """Get recent CI run data"""
        print(f"Getting recent {limit} PR Test runs...")

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
            print(f"Processing run {i}/{total_runs}: #{run.get('run_number')}")

            run_info = {
                "run_number": run.get("run_number"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha", "")[:8],
                "author": run.get("head_commit", {})
                .get("author", {})
                .get("name", "Unknown"),
                "pr_number": None,
                "url": f"https://github.com/{self.repo}/actions/runs/{run.get('id')}",
            }

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

            time.sleep(0.2)  # Slightly longer delay between runs to be API-friendly

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

    args = parser.parse_args()

    # Create analyzer
    analyzer = SGLangPerfAnalyzer(args.token)

    try:
        # Get CI run data
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No CI run data found")
            return

        # Collect performance data
        test_data = analyzer.collect_performance_data(runs)

        # Generate performance tables
        analyzer.generate_performance_tables(test_data, args.output_dir)

        # Generate summary report
        analyzer.generate_summary_report(test_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
