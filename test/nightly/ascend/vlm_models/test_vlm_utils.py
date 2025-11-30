import glob
import json
import os
import subprocess

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestVLMModels(CustomTestCase):
    model = ""
    mmmu_accuracy = 0.00
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.35,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
    ]

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.time_out = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

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
            "/__w/sglang/sglang/test/nightly/ascend/vlm_models/mmmu-val.yaml",
        ]

        subprocess.run(
            cmd,
            check=True,
            timeout=3600,
        )

    def _run_vlm_mmmu_test(
        self,
        output_path="./logs",
        test_name="",
        custom_env=None,
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
            capture_output: Whether to capture server stdout/stderr
        """
        print(f"\nTesting model: {self.model}{test_name}")

        process = None
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

            process = popen_launch_server(
                self.model,
                base_url=self.base_url,
                timeout=self.time_out,
                api_key=self.api_key,
                other_args=self.other_args,
                env=process_env,
                return_stdout_stderr=(
                    (stdout_file, stderr_file) if capture_output else None
                ),
            )

            # Run evaluation
            self.run_mmmu_eval(self.model, output_path, limit)

            # Get the result file
            result_file_path = glob.glob(f"{output_path}/*.json")[0]

            with open(result_file_path, "r") as f:
                result = json.load(f)
                print(f"Result{test_name}\n: {result}")

            # Process the result
            mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
            print(
                f"Model {self.model} achieved accuracy{test_name}: {mmmu_accuracy:.4f}"
            )

            # Capture server output if requested
            if capture_output and process:
                server_output = self._read_output_from_files()

            # Assert performance meets expected threshold
            self.assertGreaterEqual(
                mmmu_accuracy,
                self.mmmu_accuracy,
                f"Model {self.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({self.mmmu_accuracy:.4f}){test_name}",
            )

            return server_output

        except Exception as e:
            print(f"Error testing {self.model}{test_name}: {e}")
            self.fail(f"Test failed for {self.model}{test_name}: {e}")

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
