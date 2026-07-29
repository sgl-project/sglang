"""Utilities for running stress tests with bench_serving."""

import subprocess
from typing import List, Optional

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
        num_prompts: int = 50000,
        duration_minutes: int = 45,
    ):
        """Initialize the stress test runner.

        Args:
            test_name: Name of the test (used for reporting)
            base_url: Base URL for the server
            num_prompts: Number of prompts to send (default: 50000)
            duration_minutes: Timeout in minutes (default: 45)
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

    def run_stress_test_command(
        self, command: List[str], timeout_minutes: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Execute the stress test command with timeout.

        Args:
            command: Command to execute
            timeout_minutes: Timeout in minutes (uses class default if None)

        Returns:
            CompletedProcess result
        """
        timeout = (timeout_minutes or self.duration_minutes) * 60
        print(f"Running stress test command (timeout: {timeout}s):")
        print(f"  {' '.join(command)}")

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            print(f"Error running stress test:")
            print(result.stderr)
            raise RuntimeError(
                f"Stress test failed with return code {result.returncode}"
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

        # Launch server
        process = popen_launch_server(
            model=model_path,
            base_url=self.base_url,
            other_args=server_args or [],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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

            self.run_stress_test_command(command, timeout_minutes)

            print(f"\nStress test completed successfully for {model_path}")
            self._add_success_to_report(model_path, random_input_len, random_output_len)
            return True

        except Exception as e:
            print(f"\nStress test failed for {model_path}: {e}")
            self._add_failure_to_report(model_path, str(e))
            return False

        finally:
            # Always clean up server process
            print(f"Cleaning up server process...")
            kill_process_tree(process.pid)

    def _add_success_to_report(
        self, model_path: str, input_len: int, output_len: int
    ) -> None:
        """Add success entry to report."""
        model_name = model_path.split("/")[-1]
        self.full_report += f"### {model_name} - Success\n"
        self.full_report += f"- Model: `{model_path}`\n"
        self.full_report += f"- Input Length: {input_len}\n"
        self.full_report += f"- Output Length: {output_len}\n"
        self.full_report += f"- Num Prompts: {self.num_prompts}\n"
        self.full_report += f"- Status: **PASSED**\n\n"

    def _add_failure_to_report(self, model_path: str, error: str) -> None:
        """Add failure entry to report."""
        model_name = model_path.split("/")[-1]
        self.full_report += f"### {model_name} - Failure\n"
        self.full_report += f"- Model: `{model_path}`\n"
        self.full_report += f"- Status: **FAILED**\n"
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
