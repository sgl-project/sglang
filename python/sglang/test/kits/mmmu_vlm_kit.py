import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.environ import temp_set_env
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Set default mem_fraction_static to 0.8
DEFAULT_MEM_FRACTION_STATIC = 0.8


def _is_mmmu_parquet_corruption(error_output: str) -> bool:
    """Check if error is due to MMMU parquet file corruption."""
    return (
        "ArrowInvalid" in error_output
        and "Parquet magic bytes not found" in error_output
        and ("MMMU" in error_output or "lmms-lab--MMMU" in error_output)
    )


def _cleanup_mmmu_dataset_cache():
    """Clean up corrupted MMMU dataset cache to allow fresh download."""
    # Priority 1: Check CI convention path /hf_home first (used in Docker containers)
    ci_hf_home = Path("/hf_home/hub/datasets--lmms-lab--MMMU")
    if ci_hf_home.exists():
        mmmu_cache_path = ci_hf_home
    else:
        # Priority 2: Use HF_HOME env var or default user cache
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        mmmu_cache_path = Path(hf_home) / "hub" / "datasets--lmms-lab--MMMU"

    if mmmu_cache_path.exists():
        print(f"Detected corrupted MMMU parquet cache. Cleaning up: {mmmu_cache_path}")
        try:
            shutil.rmtree(mmmu_cache_path)
            print(f"Successfully removed corrupted cache: {mmmu_cache_path}")
            return True
        except OSError as e:
            print(f"Warning: Failed to remove cache {mmmu_cache_path}: {e}")
            return False
    else:
        print(f"MMMU cache not found at {mmmu_cache_path}, skipping cleanup")
        return False


def _run_lmms_eval_with_retry(cmd: list[str], timeout: int = 3600) -> None:
    """Run lmms_eval command with automatic retry on MMMU parquet corruption."""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        # Check for errors in output even if exit code is 0
        # lmms_eval sometimes returns 0 even when errors occur
        combined_output = result.stdout + result.stderr
        if _is_mmmu_parquet_corruption(combined_output):
            print(
                "Detected MMMU parquet corruption error in output. Attempting recovery..."
            )
            if _cleanup_mmmu_dataset_cache():
                print("Retrying lmms_eval with fresh download...")
                with temp_set_env(
                    HF_HUB_OFFLINE="0",
                    HF_DATASETS_DOWNLOAD_MODE="force_redownload",
                ):
                    retry_result = subprocess.run(
                        cmd, check=True, timeout=timeout, capture_output=True, text=True
                    )
                    # Print retry output
                    if retry_result.stdout:
                        print(retry_result.stdout, end="")
                    if retry_result.stderr:
                        print(retry_result.stderr, end="")
            else:
                print(
                    f"Failed to cleanup corrupted MMMU cache. Output from lmms_eval:\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
                )
                raise RuntimeError("Failed to cleanup corrupted MMMU cache")
        else:
            # Print captured output to maintain visibility of successful runs
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
    except subprocess.CalledProcessError as e:
        error_output = e.stderr + e.stdout
        if _is_mmmu_parquet_corruption(error_output):
            print("Detected MMMU parquet corruption error. Attempting recovery...")
            if _cleanup_mmmu_dataset_cache():
                print("Retrying lmms_eval with fresh download...")
                with temp_set_env(
                    HF_HUB_OFFLINE="0",
                    HF_DATASETS_DOWNLOAD_MODE="force_redownload",
                ):
                    retry_result = subprocess.run(
                        cmd, check=True, timeout=timeout, capture_output=True, text=True
                    )
                    # Print retry output
                    if retry_result.stdout:
                        print(retry_result.stdout, end="")
                    if retry_result.stderr:
                        print(retry_result.stderr, end="")
            else:
                print(
                    f"Failed to cleanup corrupted MMMU cache. Error from lmms_eval:\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
                )
                raise
        else:
            print(
                f"lmms_eval failed with an unhandled error.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
            )
            raise


class MMMUMixin:
    """Mixin for MMMU evaluation.

    Use with MMMUServerBase for single-model tests:
        class TestMyModel(MMMUMixin, MMMUServerBase):
            model = "my/model"
            accuracy = 0.4
    """

    accuracy: float
    mmmu_args: list[str] = []

    # For OpenAI API settings
    api_key = "sk-123456"

    def run_mmmu_eval(
        self: CustomTestCase,
        model_version: str,
        output_path: str,
    ):
        """
        Evaluate a VLM on the MMMU validation set with lmms-eval.
        Only `model_version` (checkpoint) and `chat_template` vary;
        We are focusing only on the validation set due to resource constraints.
        """
        # -------- fixed settings --------
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 64
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
            *self.mmmu_args,
        ]

        # Set OpenAI API key and base URL environment variables.
        # Needed for lmms-eval to work.
        with temp_set_env(
            OPENAI_API_KEY=self.api_key,
            OPENAI_API_BASE=f"{self.base_url}/v1",
        ):
            _run_lmms_eval_with_retry(cmd)

    def test_mmmu(self: CustomTestCase):
        """Run MMMU evaluation test."""
        with tempfile.TemporaryDirectory() as output_path:
            # Run evaluation
            self.run_mmmu_eval(self.model, output_path)

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
                print(f"Result: {result}")

            # Process the result
            mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
            print(f"Model {self.model} achieved accuracy: {mmmu_accuracy:.4f}")

            # Assert performance meets expected threshold
            self.assertGreaterEqual(
                mmmu_accuracy,
                self.accuracy,
                f"Model {self.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({self.accuracy:.4f})",
            )


class MMMUMultiModelTestBase(CustomTestCase):
    """Base class for multi-model MMMU tests.

    This class is for tests that need to evaluate multiple models,
    starting and stopping a server for each model within the test method.
    For single-model tests, use MMMUMixin with MMMUServerBase instead.
    """

    parsed_args = None  # Class variable to store args
    other_args = []
    mmmu_args = []

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

        # Save original environment variables for restoration in tearDownClass
        cls._original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        cls._original_openai_api_base = os.environ.get("OPENAI_API_BASE")

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

    @classmethod
    def tearDownClass(cls):
        # Restore original environment variables
        if cls._original_openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = cls._original_openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if cls._original_openai_api_base is not None:
            os.environ["OPENAI_API_BASE"] = cls._original_openai_api_base
        elif "OPENAI_API_BASE" in os.environ:
            del os.environ["OPENAI_API_BASE"]

    def run_mmmu_eval(
        self,
        model_version: str,
        output_path: str,
        *,
        env: dict | None = None,
    ):
        """
        Evaluate a VLM on the MMMU validation set with lmms-eval.
        Only `model_version` (checkpoint) and `chat_template` vary;
        We are focusing only on the validation set due to resource constraints.
        """
        # -------- fixed settings --------
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 64
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
            *self.mmmu_args,
        ]

        _run_lmms_eval_with_retry(cmd)

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
                    "64",
                    "--enable-multimodal",
                    "--mem-fraction-static",
                    str(self.parsed_args.mem_fraction_static),  # Use class variable
                    "--log-level",
                    log_level,
                    *self.other_args,
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


# Backward compatibility alias
MMMUVLMTestBase = MMMUMultiModelTestBase
