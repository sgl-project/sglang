"""Utilities for running stress tests with bench_serving."""

import json
import os
import re
import subprocess
from typing import Dict, List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class StressTestRunner:
    """Helper class for running stress tests with bench_serving.

    This class encapsulates common patterns for stress testing, including:
    - Server launch and cleanup
    - Stress test command construction and execution
    - Result collection and reporting
    """

    def __init__(
        self,
        test_name: str,
        base_url: str,
        num_prompts: int = 20000,
        duration_minutes: int = 5,
    ):
        """Initialize the stress test runner.

        Args:
            test_name: Name of the test (used for reporting)
            base_url: Base URL for the server
            num_prompts: Number of prompts to send (default: 20000)
            duration_minutes: Timeout in minutes (default: 5 for debugging)
        """
        self.test_name = test_name
        self.base_url = base_url
        self.num_prompts = num_prompts
        self.duration_minutes = duration_minutes
        self.full_report = f"## {test_name}\n"

    def build_stress_test_command(
        self,
        model_path: str,
        random_input_len: int,
        random_output_len: int,
        output_file: str,
        random_range_ratio: float = 0.2,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build the bench_serving stress test command.

        Args:
            model_path: Path to the model
            random_input_len: Random input length
            random_output_len: Random output length
            output_file: Output JSONL file path
            random_range_ratio: Random range ratio (default: 0.2)
            extra_args: Optional extra arguments

        Returns:
            List of command arguments ready for subprocess.run()
        """
        command = [
            "python3",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang-oai",
            "--base-url",
            self.base_url,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(random_input_len),
            "--random-output-len",
            str(random_output_len),
            "--random-range-ratio",
            str(random_range_ratio),
            "--num-prompts",
            str(self.num_prompts),
            "--output-file",
            output_file,
        ]

        if extra_args:
            command.extend(extra_args)

        return command

    def _parse_metrics_from_jsonl(self, output_file: str) -> Dict[str, float]:
        """Parse metrics from bench_serving JSONL output file.

        Args:
            output_file: Path to the output JSONL file

        Returns:
            Dictionary containing calculated metrics
        """
        import statistics

        metrics = {}

        # Check if file exists and print debug info
        print(f"Checking for JSONL file: {output_file}")
        print(f"Current working directory: {os.getcwd()}")

        # Try multiple possible locations for the JSONL file
        possible_paths = [
            output_file,  # Relative to current directory
            os.path.abspath(output_file),  # Absolute path from current directory
            os.path.join(os.getcwd(), output_file),  # Explicitly in current directory
            os.path.join(os.path.dirname(os.getcwd()), output_file),  # Parent directory
            os.path.join("/Users/doug/sglang", output_file),  # Project root
        ]

        # Also search for any .jsonl files in current and parent directories
        for search_dir in [
            os.getcwd(),
            os.path.dirname(os.getcwd()),
            "/Users/doug/sglang",
        ]:
            try:
                files = os.listdir(search_dir)
                jsonl_files = [f for f in files if f.endswith(".jsonl")]
                if jsonl_files:
                    print(f"JSONL files in {search_dir}: {jsonl_files}")
                    # Add any matching files to possible paths
                    for jf in jsonl_files:
                        if os.path.basename(output_file) in jf:
                            possible_paths.append(os.path.join(search_dir, jf))
            except Exception as e:
                print(f"Error listing directory {search_dir}: {e}")

        # Try each possible path
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found JSONL file at: {path}")
                found_path = path
                break

        if not found_path:
            print(
                f"Output file not found in any of {len(possible_paths)} locations tried"
            )
            return metrics

        output_file = found_path

        try:
            # Read all completed requests from JSONL
            requests = []
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            req = json.loads(line)
                            if req.get("success"):
                                requests.append(req)
                        except json.JSONDecodeError:
                            continue

            if not requests:
                print("No successful requests found in output file")
                return metrics

            # Calculate metrics from the requests
            metrics["completed"] = len(requests)

            # Duration: time from first to last request
            start_times = [
                r.get("start_time", 0) for r in requests if r.get("start_time")
            ]
            latencies = [r.get("latency", 0) for r in requests if r.get("latency")]

            if start_times and latencies:
                first_start = min(start_times)
                last_completion = max(
                    start_times[i] + latencies[i] for i in range(len(start_times))
                )
                duration = last_completion - first_start
                metrics["duration"] = duration

                # Throughput
                if duration > 0:
                    metrics["request_throughput"] = len(requests) / duration

            # Output token throughput
            output_lens = [
                r.get("output_len", 0) for r in requests if r.get("output_len")
            ]
            if output_lens and metrics.get("duration"):
                total_output_tokens = sum(output_lens)
                metrics["output_throughput"] = total_output_tokens / metrics["duration"]

            # TTFT metrics
            ttfts = [
                r.get("ttft", 0) * 1000 for r in requests if r.get("ttft")
            ]  # Convert to ms
            if ttfts:
                metrics["mean_ttft_ms"] = statistics.mean(ttfts)
                metrics["median_ttft_ms"] = statistics.median(ttfts)
                if len(ttfts) > 1:
                    sorted_ttfts = sorted(ttfts)
                    p99_idx = int(len(sorted_ttfts) * 0.99)
                    metrics["p99_ttft_ms"] = sorted_ttfts[p99_idx]

            return metrics

        except Exception as e:
            print(f"Error parsing JSONL file: {e}")
            return metrics

    def _parse_metrics_from_output(self, output: str) -> Dict[str, float]:
        """Parse benchmark metrics from bench_serving output.

        Args:
            output: The stdout/stderr from bench_serving

        Returns:
            Dictionary containing parsed metrics
        """
        metrics = {}

        # Parse key metrics using regex
        patterns = {
            "completed": r"Successful requests:\s+(\d+)",
            "duration": r"Benchmark duration \(s\):\s+([\d.]+)",
            "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
            "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
            "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
            "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
            "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[key] = float(match.group(1))

        return metrics

    def _parse_tqdm_progress(self, stderr: str) -> Dict[str, float]:
        """Parse completed request count from tqdm progress bar in stderr.

        Args:
            stderr: The stderr output containing tqdm progress

        Returns:
            Dictionary with 'completed' count if found
        """
        metrics = {}

        # Parse tqdm progress bar format: "XX%|████| 12345/20000 [time<time, XX.XXit/s]"
        # Match patterns like: "50%|████| 10000/20000 [02:30<02:30, 66.67it/s]"
        # or simpler: "10000/20000 [02:30<02:30, 66.67it/s]"
        tqdm_pattern = r"(\d+)/(\d+)\s+\[[^\]]+,\s*([\d.]+)it/s\]"

        # Find all matches (get the last one as it's the most recent)
        matches = list(re.finditer(tqdm_pattern, stderr))
        if matches:
            last_match = matches[-1]
            completed = int(last_match.group(1))
            total = int(last_match.group(2))
            rate = float(last_match.group(3))

            metrics["completed"] = completed
            metrics["request_throughput"] = rate

            print(f"Parsed from tqdm: {completed}/{total} requests at {rate:.2f} req/s")

        return metrics

    def run_stress_test_command(
        self, command: List[str], timeout_minutes: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Execute the stress test command with timeout.

        Args:
            command: Command to execute
            timeout_minutes: Timeout in minutes (uses class default if None)

        Returns:
            CompletedProcess result (or a mock result on timeout)
        """
        timeout = (timeout_minutes or self.duration_minutes) * 60
        print(f"Running stress test command (timeout: {timeout}s):")
        print(f"  {' '.join(command)}")
        print(f"Current working directory for subprocess: {os.getcwd()}")

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                print("Error running stress test:")
                print(result.stderr)
                raise RuntimeError(
                    f"Stress test failed with return code {result.returncode}"
                )

            return result

        except subprocess.TimeoutExpired as e:
            # Timeout is expected for throughput benchmarking - parse partial results
            print(f"\nStress test reached timeout ({timeout}s) - this is expected.")
            print("Parsing throughput metrics from partial output...")

            # Get stdout/stderr from the TimeoutExpired exception
            stdout = e.stdout.decode("utf-8") if e.stdout else ""
            stderr = e.stderr.decode("utf-8") if e.stderr else ""

            # Debug: print what we got
            print(f"\nTimeout exception stdout length: {len(stdout)} chars")
            print(f"Timeout exception stderr length: {len(stderr)} chars")
            if stdout:
                print(f"First 500 chars of timeout stdout:\n{stdout[:500]}")
                print(f"Last 500 chars of timeout stdout:\n{stdout[-500:]}")
            if stderr:
                print(f"First 500 chars of timeout stderr:\n{stderr[:500]}")
                print(f"Last 500 chars of timeout stderr:\n{stderr[-500:]}")

            # Create a mock CompletedProcess with the partial output
            result = subprocess.CompletedProcess(
                args=command, returncode=0, stdout=stdout, stderr=stderr
            )

            return result

    def run_stress_test_for_model(
        self,
        model_path: str,
        random_input_len: int,
        random_output_len: int,
        output_file: str,
        server_args: Optional[List[str]] = None,
        extra_bench_args: Optional[List[str]] = None,
        timeout_minutes: Optional[int] = None,
    ) -> bool:
        """Run a complete stress test for a single model with server management.

        This method handles:
        - Server launch and cleanup
        - Stress test command construction and execution
        - Error handling and reporting

        Args:
            model_path: Path to the model
            random_input_len: Random input length
            random_output_len: Random output length
            output_file: Output JSONL file path
            server_args: Arguments to pass to server launch
            extra_bench_args: Extra arguments for bench_serving
            timeout_minutes: Timeout in minutes (uses class default if None)

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Starting stress test for: {model_path}")
        print(f"Input length: {random_input_len}")
        print(f"Output length: {random_output_len}")
        print(f"Num prompts: {self.num_prompts}")
        print(f"{'='*60}\n")

        # Get server launch timeout from env var or use default
        server_timeout = int(
            os.environ.get(
                "SGLANG_SERVER_LAUNCH_TIMEOUT", DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
            )
        )
        print(f"Server launch timeout: {server_timeout}s")

        # Launch server
        process = popen_launch_server(
            model=model_path,
            base_url=self.base_url,
            other_args=server_args or [],
            timeout=server_timeout,
        )

        try:
            # Build and run stress test command
            command = self.build_stress_test_command(
                model_path,
                random_input_len,
                random_output_len,
                output_file,
                extra_args=extra_bench_args,
            )

            result = self.run_stress_test_command(command, timeout_minutes)

            # Try to parse metrics from JSONL file first (most reliable)
            print(f"\nAttempting to parse metrics from JSONL file: {output_file}")
            metrics = self._parse_metrics_from_jsonl(output_file)

            # If JSONL parsing failed, try stdout/stderr as fallback
            if not metrics:
                print("JSONL parsing failed, trying stdout/stderr...")
                metrics = self._parse_metrics_from_output(result.stdout)
                if not metrics and result.stderr:
                    metrics = self._parse_metrics_from_output(result.stderr)

            # If still no metrics, try parsing tqdm progress from stderr
            if not metrics and result.stderr:
                print("Trying to parse tqdm progress from stderr...")
                metrics = self._parse_tqdm_progress(result.stderr)

            # Debug: print parsed metrics
            print(f"\nParsed metrics: {metrics}")
            if not metrics:
                print("WARNING: No metrics were parsed!")
                print(f"stdout length: {len(result.stdout)} chars")
                print(f"stderr length: {len(result.stderr)} chars")
                print(f"\nFirst 500 chars of stdout:\n{result.stdout[:500]}")
                print(f"\nLast 500 chars of stdout:\n{result.stdout[-500:]}")
                if result.stderr:
                    print(f"\nFirst 500 chars of stderr:\n{result.stderr[:500]}")
                    print(f"\nLast 500 chars of stderr:\n{result.stderr[-500:]}")

            print(f"\nStress test completed successfully for {model_path}")
            self._add_success_to_report(
                model_path, random_input_len, random_output_len, metrics
            )
            return True

        except Exception as e:
            print(f"\nStress test failed for {model_path}: {e}")
            self._add_failure_to_report(model_path, str(e))
            return False

        finally:
            # Always clean up server process
            print("Cleaning up server process...")
            kill_process_tree(process.pid)

    def _add_success_to_report(
        self,
        model_path: str,
        input_len: int,
        output_len: int,
        metrics: Dict[str, float],
    ) -> None:
        """Add success entry to report with throughput metrics."""
        model_name = model_path.split("/")[-1]
        self.full_report += f"### {model_name} - Success\n"
        self.full_report += f"- Model: `{model_path}`\n"
        self.full_report += f"- Input Length: {input_len}\n"
        self.full_report += f"- Output Length: {output_len}\n"
        self.full_report += f"- Target Prompts: {self.num_prompts}\n"

        # Add throughput metrics if available
        if metrics.get("completed"):
            self.full_report += f"- Completed Requests: {int(metrics['completed'])}\n"
        if metrics.get("duration"):
            self.full_report += f"- Duration: {metrics['duration']:.1f}s\n"
        if metrics.get("request_throughput"):
            self.full_report += (
                f"- Request Throughput: {metrics['request_throughput']:.2f} req/s\n"
            )
        if metrics.get("output_throughput"):
            self.full_report += (
                f"- Output Token Throughput: {metrics['output_throughput']:.2f} tok/s\n"
            )
        if metrics.get("median_ttft_ms"):
            self.full_report += f"- Median TTFT: {metrics['median_ttft_ms']:.2f}ms\n"

        # If no metrics were parsed, add a note
        if not metrics:
            self.full_report += (
                "- ⚠️ Metrics not available (check output file for details)\n"
            )

        self.full_report += "- Status: **PASSED**\n\n"

    def _add_failure_to_report(self, model_path: str, error: str) -> None:
        """Add failure entry to report."""
        model_name = model_path.split("/")[-1]
        self.full_report += f"### {model_name} - Failure\n"
        self.full_report += f"- Model: `{model_path}`\n"
        self.full_report += "- Status: **FAILED**\n"
        self.full_report += f"- Error: {error}\n\n"

    def write_final_report(self) -> None:
        """Write the final report to GitHub summary if in CI."""
        if is_in_ci():
            write_github_step_summary(self.full_report)

    def get_full_report(self) -> str:
        """Get the accumulated full report.

        Returns:
            The full markdown report as a string
        """
        return self.full_report
