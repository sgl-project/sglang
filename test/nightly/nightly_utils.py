"""Utilities for running nightly performance benchmarks with profiling."""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.bench_tool_call import ToolCallBenchmarkResult

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
        ]

        if extra_args:
            command.extend(extra_args)

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

            # Clean up JSON file
            os.remove(json_output_file)

            return benchmark_results, True

        except Exception as e:
            desc = model_description or "model"
            print(f"Error loading benchmark results for {desc}: {e}")
            # Try to clean up the file anyway
            if os.path.exists(json_output_file):
                os.remove(json_output_file)
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
    ) -> Tuple[List[BenchmarkResult], bool]:
        """Run a complete benchmark for a single model with server management.

        This method handles:
        - Server launch and cleanup
        - Profile filename generation
        - Benchmark command construction and execution
        - Result loading and parsing

        Args:
            model_path: Path to the model
            batch_sizes: List of batch sizes to test
            input_lens: Tuple of input lengths
            output_lens: Tuple of output lengths
            other_args: Arguments to pass to server launch
            variant: Optional variant suffix (e.g., "basic", "mtp")
            extra_bench_args: Extra arguments for the benchmark command

        Returns:
            Tuple of (list of BenchmarkResult objects, success_bool)
        """
        benchmark_results = []
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
            )

            result, cmd_success = self.run_benchmark_command(command, model_description)

            if not cmd_success:
                return benchmark_results, False

            # Load results
            benchmark_results, load_success = self.load_benchmark_results(
                json_output_file, model_description
            )

            return benchmark_results, load_success

        finally:
            # Always clean up server process
            try:
                if process and process.poll() is None:
                    kill_process_tree(process.pid)
            except Exception as e:
                print(f"Warning: Failed to kill process {process.pid}: {e}")

    def add_report(self, results: List[BenchmarkResult]) -> None:
        """Add benchmark results to the full report.

        Args:
            results: List of BenchmarkResult objects to add to report
        """
        if results:
            report_part = generate_markdown_report(self.profile_dir, results)
            self.full_report += report_part + "\n"

    def add_tool_call_report(self, tool_call_result) -> None:
        """Add tool call benchmark results to the full report.

        Args:
            tool_call_result: ToolCallBenchmarkResult object to add to report
        """
        if tool_call_result:
            self.full_report += "\n" + tool_call_result.to_markdown_report() + "\n"

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

    def run_perf_and_tool_call_benchmark(
        self,
        model_path: str,
        batch_sizes: List[int],
        input_lens: Tuple[int, ...],
        output_lens: Tuple[int, ...],
        other_args: Optional[List[str]] = None,
        variant: str = "",
        tool_call_parser: Optional[str] = None,
        extra_bench_args: Optional[List[str]] = None,
    ) -> Tuple[List[BenchmarkResult], Optional[ToolCallBenchmarkResult], bool, bool]:
        """Run performance AND tool call benchmarks with single server launch.

        This method launches the server once and runs both benchmark types,
        avoiding multiple server starts.

        Args:
            model_path: Path to the model
            batch_sizes: List of batch sizes to test
            input_lens: Tuple of input lengths
            output_lens: Tuple of output lengths
            other_args: Arguments to pass to server launch
            variant: Optional variant suffix (e.g., "basic", "mtp")
            tool_call_parser: Tool call parser name (e.g., "llama3", "qwen")
            extra_bench_args: Extra arguments for the benchmark command

        Returns:
            Tuple of (perf_results, tool_call_results, perf_success, tool_call_success)
        """
        from sglang.bench_tool_call import ToolCallBenchmark, ToolCallParser

        benchmark_results = []
        tool_call_result = None
        model_description = f"{model_path}" + (f" ({variant})" if variant else "")

        # Build server args, adding tool-call-parser if provided
        server_args = list(other_args) if other_args else []
        if tool_call_parser:
            server_args.extend(["--tool-call-parser", tool_call_parser])

        # Launch server ONCE
        process = popen_launch_server(
            model=model_path,
            base_url=self.base_url,
            other_args=server_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        perf_success = False
        tool_call_success = True  # Default to True if no tool call benchmark

        try:
            # 1. Run performance benchmark
            profile_path_prefix, json_output_file = self.generate_profile_filename(
                model_path, variant
            )

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
            )

            result, cmd_success = self.run_benchmark_command(command, model_description)

            if cmd_success:
                benchmark_results, perf_success = self.load_benchmark_results(
                    json_output_file, model_description
                )
            else:
                perf_success = False

            # 2. Run tool call benchmark (if parser provided)
            if tool_call_parser:
                try:
                    print(f"Running tool call benchmark for {model_description}...")
                    benchmark = ToolCallBenchmark(
                        base_url=f"{self.base_url}/v1",
                        model=model_path,
                        parser=ToolCallParser(tool_call_parser),
                    )
                    tool_call_result = benchmark.run_benchmark()
                    tool_call_success = tool_call_result.success_rate >= 0.7
                    print(
                        f"Tool call benchmark: {tool_call_result.passed_tests}/{tool_call_result.total_tests} "
                        f"passed ({tool_call_result.success_rate:.1%})"
                    )
                except Exception as e:
                    print(
                        f"Error running tool call benchmark for {model_description}: {e}"
                    )
                    tool_call_success = False

            return benchmark_results, tool_call_result, perf_success, tool_call_success

        finally:
            # Always clean up server process
            try:
                if process and process.poll() is None:
                    kill_process_tree(process.pid)
            except Exception as e:
                print(f"Warning: Failed to kill process {process.pid}: {e}")
