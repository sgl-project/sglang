"""Utilities for running nightly performance benchmarks with profiling."""

import json
import os
import subprocess
import time
from typing import List, Optional, Tuple

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.nightly_bench_utils import BenchmarkResult, generate_markdown_report
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class NightlyBenchmarkRunner:
    """Helper class for running nightly performance benchmarks with profiling.

    This class encapsulates common patterns used across nightly performance tests,
    including profile directory management, benchmark command construction,
    result parsing, and report generation.
    """

    def __init__(
        self,
        profile_dir: str,
        test_name: str,
        base_url: str,
        gpu_config: str = None,
    ):
        """Initialize the benchmark runner.

        Args:
            profile_dir: Directory to store performance profiles
            test_name: Name of the test (used for reporting)
            base_url: Base URL for the server
            gpu_config: Optional GPU configuration string (e.g., "2-gpu-h100", "8-gpu-b200")
        """
        self.profile_dir = profile_dir
        self.test_name = test_name
        self.base_url = base_url
        self.gpu_config = gpu_config or os.environ.get("GPU_CONFIG", "")

        # Include GPU config in report header if available
        header = f"## {test_name}"
        if self.gpu_config:
            header += f" ({self.gpu_config})"
        header += "\n"
        self.full_report = header + BenchmarkResult.help_str()

    def setup_profile_directory(self) -> None:
        """Create the profile directory if it doesn't exist."""
        os.makedirs(self.profile_dir, exist_ok=True)

    def generate_profile_filename(
        self, model_path: str, variant: str = ""
    ) -> Tuple[str, str]:
        """Generate unique profile filename and path for the model.

        Args:
            model_path: Path to the model (e.g., "deepseek-ai/DeepSeek-V3.1")
            variant: Optional variant suffix (e.g., "basic", "mtp", "nsa")

        Returns:
            Tuple of (profile_path_prefix, json_output_file)
        """
        timestamp = int(time.time())
        model_safe_name = model_path.replace("/", "_")

        # Build filename with optional variant
        if variant:
            profile_filename = f"{model_safe_name}_{variant}_{timestamp}"
            json_filename = f"results_{model_safe_name}_{variant}_{timestamp}.json"
        else:
            profile_filename = f"{model_safe_name}_{timestamp}"
            json_filename = f"results_{model_safe_name}_{timestamp}.json"

        profile_path_prefix = os.path.join(self.profile_dir, profile_filename)

        return profile_path_prefix, json_filename

    def build_benchmark_command(
        self,
        model_path: str,
        batch_sizes: List[int],
        input_lens: Tuple[int, ...],
        output_lens: Tuple[int, ...],
        profile_path_prefix: str,
        json_output_file: str,
        extra_args: Optional[List[str]] = None,
        server_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the benchmark command with all required arguments.

        Args:
            model_path: Path to the model
            batch_sizes: List of batch sizes to test
            input_lens: Tuple of input lengths to test
            output_lens: Tuple of output lengths to test
            profile_path_prefix: Prefix for profile output files
            json_output_file: Path to JSON output file
            extra_args: Optional extra arguments to append to command
            server_args: Optional server launch arguments to record in metrics

        Returns:
            List of command arguments ready for subprocess.run()
        """
        command = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            model_path,
            "--base-url",
            self.base_url,
            "--batch-size",
            *[str(x) for x in batch_sizes],
            "--input-len",
            *[str(x) for x in input_lens],
            "--output-len",
            *[str(x) for x in output_lens],
            "--show-report",
            "--profile",
            "--profile-by-stage",
            "--profile-output-dir",
            profile_path_prefix,
            f"--pydantic-result-filename={json_output_file}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]

        if extra_args:
            command.extend(extra_args)

        # Record server launch arguments in metrics for tracking configurations
        if server_args:
            command.append("--server-args-for-metrics")
            command.extend(server_args)

        return command

    def run_benchmark_command(
        self, command: List[str], model_description: str = ""
    ) -> Tuple[subprocess.CompletedProcess, bool]:
        """Execute the benchmark command and return the result.

        Args:
            command: Command to execute
            model_description: Description for logging (e.g., "model_name (variant)")

        Returns:
            Tuple of (CompletedProcess, success_bool)
        """
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            desc = model_description or "benchmark"
            print(f"Error running benchmark for {desc}:")
            print(result.stderr)
            return result, False

        return result, True

    def load_benchmark_results(
        self, json_output_file: str, model_description: str = ""
    ) -> Tuple[List[BenchmarkResult], bool]:
        """Load and parse benchmark results from JSON file.

        Args:
            json_output_file: Path to JSON output file
            model_description: Description for logging

        Returns:
            Tuple of (list of BenchmarkResult objects, success_bool)
        """
        benchmark_results = []

        if not os.path.exists(json_output_file):
            desc = model_description or "model"
            print(f"Warning: JSON output file {json_output_file} not found for {desc}")
            return benchmark_results, False

        try:
            with open(json_output_file, "r") as f:
                json_data = json.load(f)

            # Convert JSON data to BenchmarkResult objects
            for data in json_data:
                benchmark_result = BenchmarkResult(**data)
                benchmark_results.append(benchmark_result)

            print(
                f"Loaded {len(benchmark_results)} benchmark results from {json_output_file}"
            )

            # Note: JSON files are preserved for metrics collection by CI scripts
            # They will be collected by scripts/ci/save_metrics.py

            return benchmark_results, True

        except Exception as e:
            desc = model_description or "model"
            print(f"Error loading benchmark results for {desc}: {e}")
            return benchmark_results, False

    def run_benchmark_for_model(
        self,
        model_path: str,
        batch_sizes: List[int],
        input_lens: Tuple[int, ...],
        output_lens: Tuple[int, ...],
        other_args: Optional[List[str]] = None,
        variant: str = "",
        extra_bench_args: Optional[List[str]] = None,
    ) -> Tuple[List[BenchmarkResult], bool, Optional[float]]:
        """Run a complete benchmark for a single model with server management.

        This method handles:
        - Server launch and cleanup
        - Profile filename generation
        - Benchmark command construction and execution
        - Result loading and parsing
        - Fetching speculative decoding accept length (for MTP/EAGLE)

        Args:
            model_path: Path to the model
            batch_sizes: List of batch sizes to test
            input_lens: Tuple of input lengths
            output_lens: Tuple of output lengths
            other_args: Arguments to pass to server launch
            variant: Optional variant suffix (e.g., "basic", "mtp")
            extra_bench_args: Extra arguments for the benchmark command

        Returns:
            Tuple of (list of BenchmarkResult objects, success_bool, avg_spec_accept_length or None)
        """
        benchmark_results = []
        avg_spec_accept_length = None
        model_description = f"{model_path}" + (f" ({variant})" if variant else "")

        # Launch server
        process = popen_launch_server(
            model=model_path,
            base_url=self.base_url,
            other_args=other_args or [],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        try:
            # Generate filenames
            profile_path_prefix, json_output_file = self.generate_profile_filename(
                model_path, variant
            )

            # Build and run benchmark command
            # Prepare extra args with run_name if variant is specified
            bench_args = list(extra_bench_args) if extra_bench_args else []
            if variant:
                bench_args.extend(["--run-name", variant])

            command = self.build_benchmark_command(
                model_path,
                batch_sizes,
                input_lens,
                output_lens,
                profile_path_prefix,
                json_output_file,
                extra_args=bench_args,
                server_args=other_args,
            )

            result, cmd_success = self.run_benchmark_command(command, model_description)

            if not cmd_success:
                return benchmark_results, False, None

            # Load results
            benchmark_results, load_success = self.load_benchmark_results(
                json_output_file, model_description
            )

            # Fetch speculative decoding accept length before killing server
            avg_spec_accept_length = self._get_spec_accept_length()

            return benchmark_results, load_success, avg_spec_accept_length

        finally:
            # Always clean up server process
            kill_process_tree(process.pid)

    def _get_spec_accept_length(self) -> Optional[float]:
        """Query the server for avg_spec_accept_length metric.

        Returns:
            The average speculative decoding accept length, or None if not available.
        """
        try:
            response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
            if response.status_code == 200:
                server_info = response.json()
                internal_states = server_info.get("internal_states", [])
                if internal_states and len(internal_states) > 0:
                    accept_length = internal_states[0].get("avg_spec_accept_length")
                    if accept_length is not None:
                        print(f"  avg_spec_accept_length={accept_length:.2f}")
                        return accept_length
        except Exception as e:
            print(f"  Warning: Could not fetch spec accept length: {e}")
        return None

    def add_report(
        self, results: List[BenchmarkResult], variant: Optional[str] = None
    ) -> None:
        """Add benchmark results to the full report.

        Args:
            results: List of BenchmarkResult objects to add to report
        """
        if results:
            report_part = generate_markdown_report(self.profile_dir, results, variant)
            self.full_report += report_part + "\n"

    def write_final_report(self) -> None:
        """Write the final report to GitHub summary if in CI."""
        if is_in_ci():
            write_github_step_summary(self.full_report)
        print(self.full_report)

    def get_full_report(self) -> str:
        """Get the accumulated full report.

        Returns:
            The full markdown report as a string
        """
        return self.full_report
