import glob
import json
import os
import random
import subprocess

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestVLMModels(CustomTestCase):
    models = []
    tp_size = 4
    mem_fraction_static = 0.35
    parsed_args = None  # Class variable to store args

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.time_out = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

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
        limit: str,
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
        batch_size = 2
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        # -------- compose --model_args --------
        model_args = f'model_version="{model_version}",' f"tp={tp}"

        # -------- build command list --------
        current_dir = os.path.dirname(os.path.abspath(_file_))

        config_path = os.path.join(current_dir,'mmmu-val.yaml')
        
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
            "--limit",
            limit,
            "--config",
            config_path
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
        limit="50",
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

            # Prepare stdout/stderr redirection if needed
            stdout_file = None
            stderr_file = None
            if capture_output:
                stdout_file = open("/tmp/server_stdout.log", "w")
                stderr_file = open("/tmp/server_stderr.log", "w")

            # Launch server for testing
            if is_npu():
                other_args = [
                    "--trust-remote-code",
                    "--cuda-graph-max-bs",
                    "32",
                    "--enable-multimodal",
                    "--mem-fraction-static",
                    self.mem_fraction_static,
                    "--log-level",
                    log_level,
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--tp-size",
                    self.tp_size,
                ]
            else:
                other_args = [
                    "--trust-remote-code",
                    "--cuda-graph-max-bs",
                    "32",
                    "--enable-multimodal",
                    "--mem-fraction-static",
                    str(self.parsed_args.mem_fraction_static),  # Use class variable
                    "--log-level",
                    log_level,
                ]
            process = popen_launch_server(
                model.model,
                base_url=self.base_url,
                timeout=self.time_out,
                api_key=self.api_key,
                other_args=other_args,
                env=process_env,
                return_stdout_stderr=(
                    (stdout_file, stderr_file) if capture_output else None
                ),
            )

            # Run evaluation
            self.run_mmmu_eval(model.model, output_path, limit)

            # Get the result file
            result_file_path = glob.glob(f"{output_path}/*.json")[0]

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

    def vlm_mmmu_benchmark(self):
        """Test VLM  against MMMU benchmark."""
        models_to_test = self.models

        if not is_npu():
            if is_in_ci():
                models_to_test = [random.choice(self.models)]

        for model in models_to_test:
            self._run_vlm_mmmu_test(model, "./logs")

    def vlm_mmmu_benchmark_with_small_cache(self):
        """Test VLM models against MMMU benchmark with a small embedding cache to force eviction."""
        models_to_test = self.models

        if not is_npu():
            if is_in_ci():
                models_to_test = [random.choice(self.models)]

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

    #         # Print server output for debugging
    #         print("Server output:\n", server_output)

    #         # Analyze server output for eviction events
    #         eviction_detected, eviction_count = self._detect_eviction_in_logs(
    #             server_output
    #         )

    #         # Assert that eviction was detected (since we're using small cache)
    #         self.assertTrue(
    #             eviction_detected,
    #             f"Expected eviction events to be detected with small cache (5MB), but none found. "
    #             f"Cache size may be too large for the workload or eviction logic may not be working. "
    #             f"Total log content length: {len(server_output)} characters",
    #         )

    #         print(
    #             f"Eviction detection summary: {eviction_count} eviction events detected"
    #         )

    #         # Additional assertion: if eviction was detected, the test passed
    #         if eviction_detected:
    #             print("✅ Eviction logic successfully triggered and detected!")
