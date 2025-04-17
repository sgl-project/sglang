import glob
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (  # add_common_sglang_args_and_parse,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# CI VLM model for testing
CI_MODELS = [
    SimpleNamespace(
        model="google/gemma-3-27b-it", chat_template="gemma-it", mmmu_accuracy=0.39
    ),
    # SimpleNamespace(
    #     model="Qwen/Qwen2.5-VL-7B-Instruct",
    #     chat_template="qwen2-vl",
    #     mmmu_accuracy=0.45,
    # ),
    # SimpleNamespace(
    #     model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     chat_template="llama_3_vision",
    #     mmmu_accuracy=0.31,
    # ),
    # SimpleNamespace(
    #     model="openbmb/MiniCPM-V-2_6", chat_template="minicpmv", mmmu_accuracy=0.4
    # ),
]


class TestVLMModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):  # Fixed method name (was setUPClass)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.time_out = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = cls.api_key
        os.environ["OPENAI_COMPATIBLE_API_URL"] = f"{cls.base_url}/v1"

    def run_mmmu_eval(
        self,
        model_version: str,
        chat_template: str,
        output_path: str,
        *,
        env: dict | None = None,
    ):
        """
        Evaluate a VLM on the MMMU validation set with lmms‑eval.
        Only `model_version` (checkpoint) and `chat_template` vary;
        """
        # -------- fixed settings --------
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 1
        log_suffix = "openai_compatible"
        output_path.mkdir(exist_ok=True, parents=True)

        # -------- compose --model_args --------
        model_args = (
            f'model_version="{model_version}",'
            f'chat_template="{chat_template}",'
            f"tp={tp}"
        )

        # -------- build command list --------
        cmd = [
            sys.executable,
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

        # -------- environment --------
        env_combined = os.environ.copy()
        if env:
            env_combined.update(env)  # e.g. {"OPENAI_API_KEY": "..."}

        subprocess.run(cmd, check=True, env=env_combined)

    def test_ci_models(self):
        """Test CI models against MMMU benchmark."""
        for cli_model in CI_MODELS:
            print(f"\nTesting model: {cli_model.model}")

            process = None
            mmmu_accuracy = 0  # Initialize to handle potential exceptions

            try:
                # Launch server for testing
                process = popen_launch_server(
                    cli_model.model,
                    base_url=self.base_url,
                    timeout=self.time_out,
                    api_key=self.api_key,
                    other_args=[
                        "--chat-template",
                        cli_model.chat_template,
                        "--trust-remote-code",
                    ],
                )

                # Run evaluation
                self.run_mmmu_eval(cli_model.model, cli_model.chat_template, "./logs")

                # Get the result file
                result_file_path = glob.glob("./*.json")[0]

                with open(result_file_path, "r") as f:
                    result = json.load(f)
                # Process the result
                mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
                print(f"Model {cli_model.model} achieved accuracy: {mmmu_accuracy:.4f}")

                # Assert performance meets expected threshold
                self.assertGreaterEqual(
                    mmmu_accuracy,
                    cli_model.mmmu_accuracy,
                    f"Model {cli_model.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({cli_model.mmmu_accuracy:.4f})",
                )

            except Exception as e:
                print(f"Error testing {cli_model.model}: {e}")
                self.fail(f"Test failed for {cli_model.model}: {e}")

            finally:
                # Ensure process cleanup happens regardless of success/failure
                if process is not None and process.poll() is None:
                    print(f"Cleaning up process {process.pid}")
                    try:
                        kill_process_tree(process.pid)
                    except Exception as e:
                        print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
