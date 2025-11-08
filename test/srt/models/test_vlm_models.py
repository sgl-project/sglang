import argparse
import glob
import json
import os
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

_is_hip = is_hip()
# VLM models for testing
if _is_hip:
    MODELS = [SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4)]
else:
    MODELS = [
        SimpleNamespace(model="google/gemma-3-27b-it", mmmu_accuracy=0.45),
        SimpleNamespace(model="Qwen/Qwen2.5-VL-3B-Instruct", mmmu_accuracy=0.4),
        SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4),
    ]

# Set default mem_fraction_static to 0.8
DEFAULT_MEM_FRACTION_STATIC = 0.8


class TestVLMModels(CustomTestCase):
    parsed_args = None  # Class variable to store args

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.time_out = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

        if cls.parsed_args is None:
            cls.parsed_args = SimpleNamespace(
                mem_fraction_static=DEFAULT_MEM_FRACTION_STATIC
            )

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

    def _detect_eviction_in_logs(self, log_output):
        """Detect if eviction events occurred in the log output."""
        eviction_keywords = ["Cache eviction: evicted"]

        eviction_detected = False
        eviction_count = 0

        for line in log_output.split("\n"):
            if any(keyword in line for keyword in eviction_keywords):
                eviction_detected = True
                eviction_count += 1
                print(f"Eviction detected: {line.strip()}")

        return eviction_detected, eviction_count

    def run_mmmu_eval(
        self,
        model_version: str,
        output_path: str,
        *,
        env: dict | None = None,
    ):
        """
        Evaluate a VLM on the MMMU validation set with lmms‑eval.
        Only `model_version` (checkpoint) and `chat_template` vary;
        We are focusing only on the validation set due to resource constraints.
        """
        # -------- fixed settings --------
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 32
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        # -------- compose --model_args --------
        model_args = f'model_version="{model_version}",' f"tp={tp}"

        # -------- build command list --------
        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            model,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            log_suffix,
            "--output_path",
            str(output_path),
        ]

        subprocess.run(
            cmd,
            check=True,
            timeout=3600,
        )

    def _run_vlm_mmmu_test(
        self,
        model,
        output_path,
        test_name="",
        custom_env=None,
        log_level="info",
        capture_output=False,
    ):
        """
        Common method to run VLM MMMU benchmark test.

        Args:
            model: Model to test
            output_path: Path for output logs
            test_name: Optional test name for logging
            custom_env: Optional custom environment variables
            log_level: Log level for server (default: "info")
            capture_output: Whether to capture server stdout/stderr
        """
        print(f"\nTesting model: {model.model}{test_name}")

        process = None
        mmmu_accuracy = 0  # Initialize to handle potential exceptions
        server_output = ""

        try:
            # Prepare environment variables
            process_env = os.environ.copy()
            if custom_env:
                process_env.update(custom_env)
            # if test vlm with cuda_ipc feature, open this env_var
            process_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"

            # Prepare stdout/stderr redirection if needed
            stdout_file = None
            stderr_file = None
            if capture_output:
                stdout_file = open("/tmp/server_stdout.log", "w")
                stderr_file = open("/tmp/server_stderr.log", "w")

            # Launch server for testing
            process = popen_launch_server(
                model.model,
                base_url=self.base_url,
                timeout=self.time_out,
                api_key=self.api_key,
                other_args=[
                    "--trust-remote-code",
                    "--cuda-graph-max-bs",
                    "32",
                    "--enable-multimodal",
                    "--mem-fraction-static",
                    str(self.parsed_args.mem_fraction_static),  # Use class variable
                    "--log-level",
                    log_level,
                ],
                env=process_env,
                return_stdout_stderr=(
                    (stdout_file, stderr_file) if capture_output else None
                ),
            )

            # Run evaluation
            self.run_mmmu_eval(model.model, output_path)

            # Get the result file
            # Search recursively for JSON result files (lmms-eval v0.4.1+ creates subdirectories)
            result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
            if not result_files:
                result_files = glob.glob(f"{output_path}/*.json")

            if not result_files:
                raise FileNotFoundError(f"No JSON result files found in {output_path}")

            result_file_path = result_files[0]

            with open(result_file_path, "r") as f:
                result = json.load(f)
                print(f"Result{test_name}\n: {result}")

            # Process the result
            mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
            print(
                f"Model {model.model} achieved accuracy{test_name}: {mmmu_accuracy:.4f}"
            )

            # Capture server output if requested
            if capture_output and process:
                server_output = self._read_output_from_files()

            # Assert performance meets expected threshold
            self.assertGreaterEqual(
                mmmu_accuracy,
                model.mmmu_accuracy,
                f"Model {model.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({model.mmmu_accuracy:.4f}){test_name}",
            )

            return server_output

        except Exception as e:
            print(f"Error testing {model.model}{test_name}: {e}")
            self.fail(f"Test failed for {model.model}{test_name}: {e}")

        finally:
            # Ensure process cleanup happens regardless of success/failure
            if process is not None and process.poll() is None:
                print(f"Cleaning up process {process.pid}")
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")

            # clean up temporary files
            if capture_output:
                if stdout_file:
                    stdout_file.close()
                if stderr_file:
                    stderr_file.close()
                for filename in ["/tmp/server_stdout.log", "/tmp/server_stderr.log"]:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception as e:
                        print(f"Error removing {filename}: {e}")

    def _read_output_from_files(self):
        output_lines = []

        log_files = [
            ("/tmp/server_stdout.log", "[STDOUT]"),
            ("/tmp/server_stderr.log", "[STDERR]"),
        ]
        for filename, tag in log_files:
            try:
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        for line in f:
                            output_lines.append(f"{tag} {line.rstrip()}")
            except Exception as e:
                print(f"Error reading {tag.lower()} file: {e}")

        return "\n".join(output_lines)

    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            self._run_vlm_mmmu_test(model, "./logs")

    def test_vlm_mmmu_benchmark_with_small_cache(self):
        """Test VLM models against MMMU benchmark with a small embedding cache to force eviction."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            custom_env = {"SGLANG_VLM_CACHE_SIZE_MB": "5"}

            # Run the test with output capture
            server_output = self._run_vlm_mmmu_test(
                model,
                "./logs_small_cache",
                test_name=" with small embedding cache (evict test)",
                custom_env=custom_env,
                log_level="debug",  # Enable debug logging for eviction detection
                capture_output=True,  # Capture server output
            )

            # Print server output for debugging
            print("Server output:\n", server_output)

            # Analyze server output for eviction events
            eviction_detected, eviction_count = self._detect_eviction_in_logs(
                server_output
            )

            # Assert that eviction was detected (since we're using small cache)
            self.assertTrue(
                eviction_detected,
                f"Expected eviction events to be detected with small cache (5MB), but none found. "
                f"Cache size may be too large for the workload or eviction logic may not be working. "
                f"Total log content length: {len(server_output)} characters",
            )

            print(
                f"Eviction detection summary: {eviction_count} eviction events detected"
            )

            # Additional assertion: if eviction was detected, the test passed
            if eviction_detected:
                print("✅ Eviction logic successfully triggered and detected!")


if __name__ == "__main__":
    # Define and parse arguments here, before unittest.main
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Static memory fraction for the model",
        default=DEFAULT_MEM_FRACTION_STATIC,
    )

    # Parse args intended for unittest
    args = parser.parse_args()

    # Store the parsed args object on the class
    TestVLMModels.parsed_args = args

    # Pass args to unittest
    unittest.main(argv=[sys.argv[0]])
