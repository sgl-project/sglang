#!/usr/bin/env python3
"""
SGLang Auto-tuning Script

This script searches for optimal server configurations by running benchmarks
across different parameter combinations and request rates.

Usage:
    python sglang_autotune.py \
        --model-path meta-llama/Meta-Llama-3-8B \
        --port 30010 \
        --search-server-args '{"tp": [1, 2, 4], "attention_backend": ["triton", "flashinfer"]}' \
        --request-rates 4 8 16 32 \
        --num-prompts 512
"""

import argparse
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single run."""
    config: Dict[str, Any]
    request_rate: Optional[float]  # None when using max_concurrency
    max_concurrency: Optional[int]  # None when using request_rate
    num_prompts: int
    input_token_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    success_rate: float
    timestamp: str
    status: str = "success"  # "success", "server_failed", "benchmark_failed"
    error_message: str = ""


class AutoTuner:
    """Auto-tuner for SGLang server configurations."""

    def __init__(
        self,
        model_path: str,
        static_server_args: Dict[str, Any],
        search_server_args: Dict[str, List[Any]],
        request_rates: Optional[List[float]] = None,
        max_concurrency_list: Optional[List[int]] = None,
        num_prompts: int = 512,
        dataset_name: str = "random",
        dataset_path: Optional[str] = None,
        random_input_len: int = 1024,
        random_output_len: int = 1024,
        random_range_ratio: float = 0.0,
        sharegpt_output_len: Optional[int] = None,
        sharegpt_context_len: Optional[int] = None,
        output_dir: str = "autotune_results",
        warmup_prompts: int = 10,
        server_timeout: int = 120,
        benchmark_timeout: int = 600,
        verbose_benchmark: bool = True,
        save_server_logs: bool = False,
        server_log_dir: str = "server_logs",
        tail_server_logs: bool = False,
        visualize: bool = False,
    ):
        # Validate that exactly one of request_rates or max_concurrency_list is provided
        if (request_rates is None and max_concurrency_list is None) or \
           (request_rates is not None and max_concurrency_list is not None):
            raise ValueError("Must specify exactly one of --request-rates or --max-concurrency-list")

        self.model_path = model_path
        self.static_server_args = static_server_args
        self.search_server_args = search_server_args
        self.request_rates = request_rates
        self.max_concurrency_list = max_concurrency_list
        self.use_concurrency_mode = max_concurrency_list is not None
        self.num_prompts = num_prompts
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.random_input_len = random_input_len
        self.random_output_len = random_output_len
        self.random_range_ratio = random_range_ratio
        self.sharegpt_output_len = sharegpt_output_len
        self.sharegpt_context_len = sharegpt_context_len
        self.output_dir = Path(output_dir)
        self.warmup_prompts = warmup_prompts
        self.server_timeout = server_timeout
        self.benchmark_timeout = benchmark_timeout
        self.verbose_benchmark = verbose_benchmark
        self.save_server_logs = save_server_logs
        self.server_log_dir = Path(server_log_dir)
        self.tail_server_logs = tail_server_logs
        self.visualize = visualize

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.results: List[BenchmarkResult] = []

        # Current server process and log files
        self.server_process: Optional[subprocess.Popen] = None
        self.server_stdout_file = None
        self.server_stderr_file = None

        # Setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        _ = signum, frame  # Unused but required by signal handler interface
        print("\n\nReceived interrupt signal. Cleaning up...")
        self._stop_server()
        sys.exit(0)

    def _generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test."""
        if not self.search_server_args:
            return [{}]

        # Get all parameter names and their values
        param_names = list(self.search_server_args.keys())
        param_values = [self.search_server_args[name] for name in param_names]

        # Generate all combinations
        configurations = []
        for values in product(*param_values):
            config = dict(zip(param_names, values))
            configurations.append(config)

        return configurations

    def _build_server_command(self, config: Dict[str, Any]) -> List[str]:
        """Build the server launch command with given configuration."""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
        ]

        # Add static parameters
        for key, value in self.static_server_args.items():
            # Special handling for attention_backend - don't replace underscores
            if key == "attention_backend":
                key_formatted = f"--{key.replace('_', '-')}"
            else:
                key_formatted = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(key_formatted)
            elif value is not None:
                cmd.extend([key_formatted, str(value)])

        # Add search parameters for this configuration
        for key, value in config.items():
            # Special handling for attention_backend - don't replace underscores
            if key == "attention_backend":
                key_formatted = f"--{key.replace('_', '-')}"
            else:
                key_formatted = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(key_formatted)
            elif value is not None:
                cmd.extend([key_formatted, str(value)])

        return cmd

    def _start_server(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Start the SGLang server with given configuration.

        Returns:
            (success, error_message) tuple
        """
        cmd = self._build_server_command(config)

        print(f"\nStarting server with command:")
        print(" ".join(cmd))

        # Prepare log files if requested
        if self.save_server_logs:
            # Create log directory
            self.server_log_dir.mkdir(parents=True, exist_ok=True)

            # Create unique log file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = "_".join(f"{k}={v}" for k, v in config.items()).replace("/", "_")

            stdout_path = self.server_log_dir / f"server_stdout_{config_str}_{timestamp}.log"
            stderr_path = self.server_log_dir / f"server_stderr_{config_str}_{timestamp}.log"

            self.server_stdout_file = open(stdout_path, 'w')
            self.server_stderr_file = open(stderr_path, 'w')

            print(f"Server logs will be saved to:")
            print(f"  stdout: {stdout_path}")
            print(f"  stderr: {stderr_path}")

            # Store paths for later reference
            self.current_stdout_path = stdout_path
            self.current_stderr_path = stderr_path

        try:
            # Use DEVNULL when not saving logs to avoid pipe buffer issues
            stdout_dest = self.server_stdout_file if self.server_stdout_file else subprocess.DEVNULL
            stderr_dest = self.server_stderr_file if self.server_stderr_file else subprocess.DEVNULL

            self.server_process = subprocess.Popen(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Wait for server to be ready
            port = self.static_server_args.get("port", 30000)
            success, error_msg = self._wait_for_server(port)
            return success, error_msg

        except Exception as e:
            error_msg = f"Failed to start server: {e}"
            print(error_msg)
            # Close log files on error
            if self.server_stdout_file:
                self.server_stdout_file.close()
                self.server_stdout_file = None
            if self.server_stderr_file:
                self.server_stderr_file.close()
                self.server_stderr_file = None
            return False, error_msg

    def _wait_for_server(self, port: int, timeout: int = None) -> tuple[bool, str]:
        """Wait for the server to be ready.

        Returns:
            (success, error_message) tuple
        """
        timeout = timeout or self.server_timeout
        url = f"http://localhost:{port}/health"

        print(f"Waiting for server to be ready at {url}...")
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print("Server is ready!")
                    # Additional warmup time
                    time.sleep(5)
                    return True, ""
            except requests.exceptions.RequestException:
                pass

            # Check if process has terminated
            if self.server_process and self.server_process.poll() is not None:
                error_msg = f"Server process terminated unexpectedly. "

                if self.save_server_logs:
                    # Flush and close log files to ensure all data is written
                    if self.server_stdout_file:
                        self.server_stdout_file.flush()
                    if self.server_stderr_file:
                        self.server_stderr_file.flush()

                    # Display tail of logs if requested
                    if self.tail_server_logs and hasattr(self, 'current_stderr_path'):
                        print("\n--- Last 50 lines of server stderr ---")
                        try:
                            with open(self.current_stderr_path, 'r') as f:
                                lines = f.readlines()
                                print(''.join(lines[-50:]))
                        except:
                            pass
                        print("--- End of server stderr ---\n")

                    # Extract error from log file
                    if hasattr(self, 'current_stderr_path'):
                        try:
                            with open(self.current_stderr_path, 'r') as f:
                                stderr_content = f.read()
                                stderr_lines = stderr_content.split('\n')
                                for line in reversed(stderr_lines):
                                    if 'error:' in line.lower() or 'exception' in line.lower():
                                        error_msg += f"Error: {line.strip()}"
                                        break
                                else:
                                    error_msg += f"Last stderr: {stderr_content[-500:]}"
                        except:
                            pass
                else:
                    # When not saving logs, we use DEVNULL so no output to read
                    print(f"Server process terminated unexpectedly!")

                return False, error_msg

            time.sleep(2)

        error_msg = f"Server failed to start within {timeout} seconds"
        print(error_msg)
        return False, error_msg

    def _stop_server(self):
        """Stop the currently running server."""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't terminate gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
            time.sleep(2)  # Give time for port to be released

        # Close log files if they exist
        if self.server_stdout_file:
            self.server_stdout_file.close()
            self.server_stdout_file = None
        if self.server_stderr_file:
            self.server_stderr_file.close()
            self.server_stderr_file = None

    def _run_benchmark(
        self,
        config: Dict[str, Any],
        request_rate: Optional[float] = None,
        max_concurrency: Optional[int] = None
    ) -> Optional[BenchmarkResult]:
        """Run benchmark for a specific configuration with either request rate or max concurrency."""
        port = self.static_server_args.get("port", 30000)

        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--port", str(port),
            "--model", self.model_path,
            "--dataset-name", self.dataset_name,
            "--num-prompts", str(self.num_prompts),
        ]

        # Add dataset-specific parameters
        if self.dataset_name == "random":
            cmd.extend(["--random-input-len", str(self.random_input_len)])
            cmd.extend(["--random-output-len", str(self.random_output_len)])
            if self.random_range_ratio > 0:
                cmd.extend(["--random-range-ratio", str(self.random_range_ratio)])
        elif self.dataset_name == "sharegpt":
            if self.sharegpt_output_len:
                cmd.extend(["--sharegpt-output-len", str(self.sharegpt_output_len)])
            if self.sharegpt_context_len:
                cmd.extend(["--sharegpt-context-len", str(self.sharegpt_context_len)])

        if self.dataset_path:
            cmd.extend(["--dataset-path", self.dataset_path])

        # Add either request rate or max concurrency
        if request_rate is not None:
            cmd.extend(["--request-rate", str(request_rate)])
            print(f"\nRunning benchmark with request rate {request_rate}...")
        else:
            cmd.extend(["--max-concurrency", str(max_concurrency)])
            print(f"\nRunning benchmark with max concurrency {max_concurrency}...")

        print(" ".join(cmd))

        try:
            # Use Popen for real-time streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Read as bytes for better control
                bufsize=0,  # Unbuffered
            )

            # Collect output while optionally streaming
            stdout_data = []

            # Stream stdout in real-time if verbose
            if self.verbose_benchmark:
                print("\n--- Benchmark Output ---", flush=True)

            # Read stdout and stderr for real-time progress bars
            # Set timeout for select
            timeout = self.benchmark_timeout
            start_time = time.monotonic()
            stderr_data = []

            # Make stderr and stdout non-blocking
            fcntl.fcntl(process.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
            fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break

                # Check for timeout
                if time.monotonic() - start_time > timeout:
                    process.kill()
                    process.wait()
                    print(f"Benchmark timed out after {self.benchmark_timeout} seconds")
                    return None

                # Use select to check if data is available on stdout or stderr
                ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.01)

                for stream in ready:
                    try:
                        if stream == process.stdout:
                            chunk = os.read(process.stdout.fileno(), 65536)
                            if chunk:
                                stdout_data.append(chunk)
                                if self.verbose_benchmark:
                                    sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                                    sys.stdout.flush()
                        elif stream == process.stderr:
                            chunk = os.read(process.stderr.fileno(), 65536)
                            if chunk:
                                stderr_data.append(chunk)
                                if self.verbose_benchmark:
                                    # Print stderr output in real-time (this includes progress bars)
                                    sys.stderr.write(chunk.decode('utf-8', errors='replace'))
                                    sys.stderr.flush()
                    except OSError:
                        # No data available or pipe closed
                        pass

            # Get return code
            return_code = process.wait()

            # Read any remaining stdout
            try:
                while True:
                    remaining = os.read(process.stdout.fileno(), 65536)
                    if not remaining:
                        break
                    stdout_data.append(remaining)
                    if self.verbose_benchmark:
                        sys.stdout.write(remaining.decode('utf-8', errors='replace'))
                        sys.stdout.flush()
            except OSError:
                pass

            # Read any remaining stderr
            try:
                while True:
                    remaining = os.read(process.stderr.fileno(), 65536)
                    if not remaining:
                        break
                    stderr_data.append(remaining)
                    if self.verbose_benchmark:
                        sys.stderr.write(remaining.decode('utf-8', errors='replace'))
                        sys.stderr.flush()
            except OSError:
                pass

            # Join stderr data
            stderr = b''.join(stderr_data).decode('utf-8', errors='replace')

            if self.verbose_benchmark:
                print("\n--- End Benchmark Output ---\n")

            # Join collected output for parsing
            stdout = b''.join(stdout_data).decode('utf-8', errors='replace')

            if return_code != 0:
                print(f"Benchmark failed with return code {return_code}")
                if stderr and not self.verbose_benchmark:
                    print(f"STDERR: {stderr[-1000:]}")
                return None

            # Parse the output
            return self._parse_benchmark_output(
                stdout, config, request_rate, max_concurrency
            )

        except Exception as e:
            print(f"Benchmark failed with error: {e}")
            return None

    def _create_failed_result(
        self,
        config: Dict[str, Any],
        request_rate: Optional[float],
        max_concurrency: Optional[int],
        status: str,
        error_message: str
    ) -> BenchmarkResult:
        """Create a BenchmarkResult for failed configurations."""
        return BenchmarkResult(
            config=config,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            num_prompts=self.num_prompts,
            input_token_throughput=0,
            output_token_throughput=0,
            total_token_throughput=0,
            mean_ttft_ms=0,
            median_ttft_ms=0,
            p99_ttft_ms=0,
            mean_itl_ms=0,
            median_itl_ms=0,
            p99_itl_ms=0,
            mean_e2e_latency_ms=0,
            median_e2e_latency_ms=0,
            success_rate=0,
            timestamp=datetime.now().isoformat(),
            status=status,
            error_message=error_message
        )

    def _parse_benchmark_output(
        self,
        output: str,
        config: Dict[str, Any],
        request_rate: Optional[float],
        max_concurrency: Optional[int]
    ) -> Optional[BenchmarkResult]:
        """Parse benchmark output to extract metrics."""
        try:
            lines = output.split('\n')
            metrics = {}

            for line in lines:
                line = line.strip()

                # Parse token throughput metrics
                if "Input token throughput (tok/s):" in line:
                    metrics['input_token_throughput'] = float(line.split()[-1])
                elif "Output token throughput (tok/s):" in line:
                    metrics['output_token_throughput'] = float(line.split()[-1])
                elif "Total token throughput (tok/s):" in line:
                    metrics['total_token_throughput'] = float(line.split()[-1])

                # Parse TTFT - new format
                if "Mean TTFT (ms):" in line:
                    metrics['mean_ttft_ms'] = float(line.split()[-1])
                elif "Median TTFT (ms):" in line:
                    metrics['median_ttft_ms'] = float(line.split()[-1])
                elif "P99 TTFT (ms):" in line:
                    metrics['p99_ttft_ms'] = float(line.split()[-1])

                # Parse ITL - new format
                if "Mean ITL (ms):" in line:
                    metrics['mean_itl_ms'] = float(line.split()[-1])
                elif "Median ITL (ms):" in line:
                    metrics['median_itl_ms'] = float(line.split()[-1])
                elif "P99 ITL (ms):" in line:
                    metrics['p99_itl_ms'] = float(line.split()[-1])

                # Parse E2E latency - new format
                if "Mean E2E Latency (ms):" in line:
                    metrics['mean_e2e_latency_ms'] = float(line.split()[-1])
                elif "Median E2E Latency (ms):" in line:
                    metrics['median_e2e_latency_ms'] = float(line.split()[-1])

            # Calculate success rate (assuming all successful if we got here)
            metrics['success_rate'] = 1.0

            # Create result object
            return BenchmarkResult(
                config=config,
                request_rate=request_rate,
                max_concurrency=max_concurrency,
                num_prompts=self.num_prompts,
                input_token_throughput=metrics.get('input_token_throughput', 0),
                output_token_throughput=metrics.get('output_token_throughput', 0),
                total_token_throughput=metrics.get('total_token_throughput', 0),
                mean_ttft_ms=metrics.get('mean_ttft_ms', 0),
                median_ttft_ms=metrics.get('median_ttft_ms', 0),
                p99_ttft_ms=metrics.get('p99_ttft_ms', 0),
                mean_itl_ms=metrics.get('mean_itl_ms', 0),
                median_itl_ms=metrics.get('median_itl_ms', 0),
                p99_itl_ms=metrics.get('p99_itl_ms', 0),
                mean_e2e_latency_ms=metrics.get('mean_e2e_latency_ms', 0),
                median_e2e_latency_ms=metrics.get('median_e2e_latency_ms', 0),
                success_rate=metrics.get('success_rate', 1.0),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"Failed to parse benchmark output: {e}")
            return None

    def _save_results(self):
        """Save results to files."""
        if not self.results:
            print("No results to save")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Save detailed CSV
        csv_path = self.output_dir / f"autotune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")

        # Save summary JSON
        json_path = self.output_dir / f"autotune_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Count successful and failed runs
        success_count = len([r for r in self.results if r.status == "success"])
        failed_count = len([r for r in self.results if r.status != "success"])

        summary = {
            "model_path": self.model_path,
            "static_server_args": self.static_server_args,
            "search_server_args": self.search_server_args,
            "request_rates": self.request_rates,
            "max_concurrency_list": self.max_concurrency_list,
            "num_prompts": self.num_prompts,
            "dataset_name": self.dataset_name,
            "summary_stats": {
                "total_runs": len(self.results),
                "successful_runs": success_count,
                "failed_runs": failed_count
            },
            "results": [asdict(r) for r in self.results]
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {json_path}")
        print(f"Total runs: {len(self.results)} | Successful: {success_count} | Failed: {failed_count}")

        # Generate visualizations if requested
        if self.visualize:
            self._generate_visualizations(df)

    def _print_summary(self):
        """Print a summary of the results."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*80)
        print("AUTOTUNE RESULTS SUMMARY")
        print("="*80)

        # Group by configuration
        df = pd.DataFrame([asdict(r) for r in self.results])

        # First, show failed configurations
        failed_df = df[df['status'] != 'success']
        if not failed_df.empty:
            print("\n" + "="*80)
            print("FAILED CONFIGURATIONS")
            print("="*80)

            for _, row in failed_df.iterrows():
                config = row['config']
                print(f"\nConfiguration: {config}")
                print(f"  Status: {row['status']}")
                print(f"  Error: {row['error_message'][:200]}")  # Truncate long messages

        # Show successful configurations
        success_df = df[df['status'] == 'success']
        if not success_df.empty:
            print("\n" + "="*80)
            print("SUCCESSFUL CONFIGURATIONS")
            print("="*80)

            for config_str in success_df['config'].apply(json.dumps).unique():
                config = json.loads(config_str)
                config_df = success_df[success_df['config'].apply(json.dumps) == config_str]

                print(f"\nConfiguration: {config}")
                print("-" * 40)

                # Create summary table - show either request_rate or max_concurrency
                if self.use_concurrency_mode:
                    summary = config_df[['max_concurrency', 'total_token_throughput', 'median_ttft_ms',
                                        'median_itl_ms', 'median_e2e_latency_ms']].round(2)
                else:
                    summary = config_df[['request_rate', 'total_token_throughput', 'median_ttft_ms',
                                        'median_itl_ms', 'median_e2e_latency_ms']].round(2)
                print(summary.to_string(index=False))

            # Find best configuration for each metric (only from successful runs)
            print("\n" + "="*80)
            print("BEST CONFIGURATIONS BY METRIC (successful runs only)")
            print("="*80)

            metrics = [
                ('total_token_throughput', 'max'),
                ('median_ttft_ms', 'min'),
                ('median_itl_ms', 'min'),
                ('median_e2e_latency_ms', 'min')
            ]

            for metric, optimize in metrics:
                # Filter out zero values for failed runs
                valid_df = success_df[success_df[metric] > 0] if optimize == 'min' else success_df

                if valid_df.empty:
                    print(f"\nNo valid data for {metric}")
                    continue

                if optimize == 'max':
                    best_idx = valid_df[metric].idxmax()
                else:
                    best_idx = valid_df[metric].idxmin()

                best_row = valid_df.loc[best_idx]
                print(f"\nBest {metric}: {best_row[metric]:.2f}")
                print(f"  Config: {best_row['config']}")
                print(f"  Request Rate: {best_row['request_rate']}")
        else:
            print("\nNo successful configurations found!")

    def _generate_visualizations(self, df):
        """Generate grouped bar charts for benchmark metrics."""

        print("Generating visulaizations...")

        # Filter out failed runs
        df_success = df[df['status'] == 'success'].copy()
        if df_success.empty:
            print("No successful runs to visualize")
            return

        # Determine the x-axis label and values
        if self.use_concurrency_mode:
            x_col = 'max_concurrency'
            x_label = 'Max Concurrency'
        else:
            x_col = 'request_rate'
            x_label = 'Request Rate'

        # Get unique x values and configurations
        x_values = sorted(df_success[x_col].unique())
        configs = df_success['config'].apply(str).unique()

        # Define metrics to plot
        metrics = [
            ('total_token_throughput', 'Total Token Throughput (tok/s)', True),  # Higher is better
            ('median_ttft_ms', 'Median TTFT (ms)', False),  # Lower is better
            ('median_itl_ms', 'Median ITL (ms)', False),  # Lower is better
            ('median_e2e_latency_ms', 'Median E2E Latency (ms)', False),  # Lower is better
        ]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Benchmark Results: {self.model_path}', fontsize=16)
        axes = axes.flatten()

        for idx, (metric, metric_label, higher_better) in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for grouped bar chart
            bar_width = 0.8 / len(configs)
            x_positions = np.arange(len(x_values))

            for config_idx, config in enumerate(configs):
                config_data = df_success[df_success['config'].apply(str) == config]
                values = []
                for x_val in x_values:
                    data_point = config_data[config_data[x_col] == x_val]
                    if not data_point.empty:
                        values.append(data_point[metric].iloc[0])
                    else:
                        values.append(0)  # No data for this combination

                # Position bars
                positions = x_positions + config_idx * bar_width - (len(configs) - 1) * bar_width / 2

                # Create bars
                bars = ax.bar(positions, values, bar_width, label=config)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if value > 0:  # Only label bars with data
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom', fontsize=8)

            # Customize subplot
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(x) for x in x_values])
            ax.legend(title='Configuration', loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add indicator for optimization direction
            if higher_better:
                ax.text(0.02, 0.98, '↑ Higher is better', transform=ax.transAxes,
                       va='top', fontsize=8, color='green')
            else:
                ax.text(0.02, 0.98, '↓ Lower is better', transform=ax.transAxes,
                       va='top', fontsize=8, color='blue')

        plt.tight_layout()

        # Save figure
        viz_path = self.output_dir / f"benchmark_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        print(f"\nVisualization saved to: {viz_path}")

        # Also show the plot if running interactively
        try:
            plt.show()
        except:
            pass  # Might fail in non-interactive environments

    def run(self):
        """Run the auto-tuning process."""
        configurations = self._generate_configurations()

        # Determine which mode we're in and calculate total runs
        if self.use_concurrency_mode:
            test_values = self.max_concurrency_list
            test_label = "Max concurrency values"
        else:
            test_values = self.request_rates
            test_label = "Request rates"

        total_runs = len(configurations) * len(test_values)

        print(f"\nStarting auto-tuning with:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Configurations to test: {len(configurations)}")
        print(f"  - {test_label}: {test_values}")
        print(f"  - Total runs: {total_runs}")
        print(f"  - Static server args: {self.static_server_args}")
        print(f"  - Search server args: {self.search_server_args}")

        run_count = 0

        for config in configurations:
            print(f"\n{'='*80}")
            print(f"Testing configuration: {config}")
            print(f"{'='*80}")

            # Start server with this configuration
            server_success, server_error = self._start_server(config)
            if not server_success:
                print(f"Failed to start server with config: {config}")
                # Record failure for all request rates for this config
                for test_value in test_values:
                    run_count += 1
                    if self.use_concurrency_mode:
                        print(f"\n[Run {run_count}/{total_runs}] Max concurrency: {test_value} - SKIPPED (server failed)")
                        failed_result = self._create_failed_result(
                            config, None, test_value, "server_failed", server_error
                        )
                    else:
                        print(f"\n[Run {run_count}/{total_runs}] Request rate: {test_value} - SKIPPED (server failed)")
                        failed_result = self._create_failed_result(
                            config, test_value, None, "server_failed", server_error
                        )
                    self.results.append(failed_result)
                continue

            # Run benchmarks for each test value (request rate or max concurrency)
            for test_value in test_values:
                run_count += 1

                if self.use_concurrency_mode:
                    print(f"\n[Run {run_count}/{total_runs}] Max concurrency: {test_value}")
                    result = self._run_benchmark(config, request_rate=None, max_concurrency=test_value)
                else:
                    print(f"\n[Run {run_count}/{total_runs}] Request rate: {test_value}")
                    result = self._run_benchmark(config, request_rate=test_value, max_concurrency=None)

                if result:
                    self.results.append(result)
                    print(f"✓ Benchmark completed successfully")
                    print(f"  Total token throughput: {result.total_token_throughput:.2f} tok/s")
                    print(f"  Median TTFT: {result.median_ttft_ms:.2f} ms")
                    print(f"  Median E2E: {result.median_e2e_latency_ms:.2f} ms")
                else:
                    print(f"✗ Benchmark failed")
                    # Record benchmark failure
                    if self.use_concurrency_mode:
                        failed_result = self._create_failed_result(
                            config, None, test_value, "benchmark_failed", "Benchmark execution failed"
                        )
                    else:
                        failed_result = self._create_failed_result(
                            config, test_value, None, "benchmark_failed", "Benchmark execution failed"
                        )
                    self.results.append(failed_result)

            # Stop server before next configuration
            self._stop_server()

        # Save and display results
        self._save_results()
        self._print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Auto-tuning Script - Find optimal server configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with tensor parallelism tuning
  %(prog)s --model-path meta-llama/Meta-Llama-3-8B \\
           --port 30010 \\
           --search-server-args '{"tp": [1, 2, 4]}' \\
           --request-rates 4 8 16

  # Tune multiple parameters with static args
  %(prog)s --model-path meta-llama/Meta-Llama-3-70B \\
           --port 30010 --dtype float16 \\
           --search-server-args '{"tp": [2, 4], "attention_backend": ["triton", "flashinfer"]}' \\
           --request-rates 4 8 16 32 \\
           --num-prompts 1000

  # Load configurations from file
  %(prog)s --model-path meta-llama/Meta-Llama-3-8B \\
           --config config.json
        """
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or name of the model to benchmark"
    )

    # Search parameter configuration
    parser.add_argument(
        "--search-server-args",
        type=str,
        default="{}",
        help='JSON string of server parameters to search over (e.g., \'{"tp": [1, 2, 4], "attention_backend": ["triton", "flashinfer"]}\')'
    )

    # Benchmark configuration
    # Load testing mode - mutually exclusive
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--request-rates",
        type=float,
        nargs="+",
        default=None,
        help="Request rates to test (requests per second)"
    )
    load_group.add_argument(
        "--max-concurrency-list",
        type=int,
        nargs="+",
        default=None,
        help="List of max concurrency values to test (alternative to request-rates)"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=512,
        help="Number of prompts to send in each benchmark"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["sharegpt", "random", "random-ids", "generated-shared-prefix", "mmmu", "random-image", "mooncake"],
        help="Name of the dataset to benchmark on"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file"
    )

    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Input token length for random dataset (matches bench_serving)"
    )

    parser.add_argument(
        "--random-output-len",
        type=int,
        default=1024,
        help="Output token length for random dataset (matches bench_serving)"
    )

    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length for random dataset"
    )

    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Override output length for ShareGPT dataset"
    )

    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="Context length for ShareGPT dataset"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="autotune_results",
        help="Directory to save results"
    )

    # Advanced options
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=10,
        help="Number of warmup prompts before benchmark"
    )

    parser.add_argument(
        "--server-timeout",
        type=int,
        default=120,
        help="Timeout for server startup (seconds)"
    )

    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=600,
        help="Timeout for each benchmark run (seconds)"
    )

    parser.add_argument(
        "--verbose-benchmark",
        action="store_true",
        default=True,
        help="Show real-time benchmark output during execution (default: True)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress real-time benchmark output"
    )

    # Server logging options
    parser.add_argument(
        "--save-server-logs",
        action="store_true",
        help="Save server stdout/stderr to files for debugging"
    )

    parser.add_argument(
        "--server-log-dir",
        type=str,
        default="server_logs",
        help="Directory to save server logs (default: server_logs)"
    )

    parser.add_argument(
        "--tail-server-logs",
        action="store_true",
        help="Display last lines of server logs when server fails to start"
    )

    # Visualization option
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate grouped bar charts for benchmark metrics"
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (overrides other arguments)"
    )

    # Parse known args and collect unknown args for server
    args, unknown = parser.parse_known_args()

    # Process unknown arguments as server args
    static_server_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:].replace('-', '_')
            # Check if next item is a value or another flag
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                # It's a key-value pair
                value = unknown[i + 1]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    value = int(value)
                except ValueError:
                    try:
                        # Try float
                        value = float(value)
                    except ValueError:
                        # Keep as string, but check for boolean
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                static_server_args[key] = value
                i += 2
            else:
                # It's a boolean flag
                static_server_args[key] = True
                i += 1
        else:
            print(f"Warning: Ignoring unknown argument: {unknown[i]}")
            i += 1

    # Load configuration from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

        model_path = config.get("model_path", args.model_path)
        # Merge static args from config with command line unknown args
        config_static = config.get("static_server_args", {})
        config_static.update(static_server_args)
        static_server_args = config_static
        search_server_args = config.get("search_server_args", json.loads(args.search_server_args))
        request_rates = config.get("request_rates", args.request_rates)
        max_concurrency_list = config.get("max_concurrency_list", args.max_concurrency_list)
        num_prompts = config.get("num_prompts", args.num_prompts)
        dataset_name = config.get("dataset_name", args.dataset_name)
        dataset_path = config.get("dataset_path", args.dataset_path)
        random_input_len = config.get("random_input_len", config.get("random_input", args.random_input_len))  # Support old config files
        random_output_len = config.get("random_output_len", config.get("random_output", args.random_output_len))  # Support old config files
        random_range_ratio = config.get("random_range_ratio", args.random_range_ratio)
        sharegpt_output_len = config.get("sharegpt_output_len", args.sharegpt_output_len)
        sharegpt_context_len = config.get("sharegpt_context_len", args.sharegpt_context_len)
        output_dir = config.get("output_dir", args.output_dir)
        warmup_prompts = config.get("warmup_prompts", args.warmup_prompts)
        server_timeout = config.get("server_timeout", args.server_timeout)
        benchmark_timeout = config.get("benchmark_timeout", args.benchmark_timeout)
        # Handle quiet flag overriding verbose (quiet takes precedence)
        verbose_benchmark = False if args.quiet else config.get("verbose_benchmark", True)
    else:
        model_path = args.model_path
        search_server_args = json.loads(args.search_server_args)
        request_rates = args.request_rates
        max_concurrency_list = args.max_concurrency_list
        num_prompts = args.num_prompts
        dataset_name = args.dataset_name
        dataset_path = args.dataset_path
        random_input_len = args.random_input_len
        random_output_len = args.random_output_len
        random_range_ratio = args.random_range_ratio
        sharegpt_output_len = args.sharegpt_output_len
        sharegpt_context_len = args.sharegpt_context_len
        output_dir = args.output_dir
        warmup_prompts = args.warmup_prompts
        server_timeout = args.server_timeout
        benchmark_timeout = args.benchmark_timeout
        # Handle quiet flag (quiet takes precedence over default verbose=True)
        verbose_benchmark = False if args.quiet else True

    # Create and run auto-tuner
    tuner = AutoTuner(
        model_path=model_path,
        static_server_args=static_server_args,
        search_server_args=search_server_args,
        request_rates=request_rates,
        max_concurrency_list=max_concurrency_list,
        num_prompts=num_prompts,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=random_range_ratio,
        sharegpt_output_len=sharegpt_output_len,
        sharegpt_context_len=sharegpt_context_len,
        output_dir=output_dir,
        warmup_prompts=warmup_prompts,
        server_timeout=server_timeout,
        benchmark_timeout=benchmark_timeout,
        verbose_benchmark=verbose_benchmark,
        save_server_logs=args.save_server_logs,
        server_log_dir=args.server_log_dir,
        tail_server_logs=args.tail_server_logs,
        visualize=args.visualize
    )

    tuner.run()


if __name__ == "__main__":
    main()