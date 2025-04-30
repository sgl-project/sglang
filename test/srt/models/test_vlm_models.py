import argparse
import glob
import json
import os
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

# VLM models for testing
MODELS = [
    SimpleNamespace(
        model="google/gemma-3-27b-it", chat_template="gemma-it", mmmu_accuracy=0.45
    ),
    SimpleNamespace(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        chat_template="qwen2-vl",
        mmmu_accuracy=0.4,
    ),
    SimpleNamespace(
        model="openbmb/MiniCPM-V-2_6", chat_template="minicpmv", mmmu_accuracy=0.4
    ),
]


class TestVLMModels(CustomTestCase):
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
        model_args = (
            f'model_version="{model_version}",'
            f'chat_template="{chat_template}",'
            f"tp={tp}"
        )

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

    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            print(f"\nTesting model: {model.model}")

            process = None
            mmmu_accuracy = 0  # Initialize to handle potential exceptions

            try:
                # Launch server for testing
                process = popen_launch_server(
                    model.model,
                    base_url=self.base_url,
                    timeout=self.time_out,
                    api_key=self.api_key,
                    other_args=[
                        "--chat-template",
                        model.chat_template,
                        "--trust-remote-code",
                        "--cuda-graph-max-bs",
                        "32",
                        "--enable-multimodal",
                        "--mem-fraction-static",
                        str(self.parsed_args.mem_fraction_static),  # Use class variable
                    ],
                )

                # Run evaluation
                self.run_mmmu_eval(model.model, model.chat_template, "./logs")

                # Get the result file
                result_file_path = glob.glob("./logs/*.json")[0]

                with open(result_file_path, "r") as f:
                    result = json.load(f)
                    print(f"Result \n: {result}")
                # Process the result
                mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
                print(f"Model {model.model} achieved accuracy: {mmmu_accuracy:.4f}")

                # Assert performance meets expected threshold
                self.assertGreaterEqual(
                    mmmu_accuracy,
                    model.mmmu_accuracy,
                    f"Model {model.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({model.mmmu_accuracy:.4f})",
                )

            except Exception as e:
                print(f"Error testing {model.model}: {e}")
                self.fail(f"Test failed for {model.model}: {e}")

            finally:
                # Ensure process cleanup happens regardless of success/failure
                if process is not None and process.poll() is None:
                    print(f"Cleaning up process {process.pid}")
                    try:
                        kill_process_tree(process.pid)
                    except Exception as e:
                        print(f"Error killing process: {e}")


if __name__ == "__main__":
    # Define and parse arguments here, before unittest.main
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Static memory fraction for the model",
        default=0.8,
    )

    # Parse args intended for unittest
    args = parser.parse_args()

    # Store the parsed args object on the class
    TestVLMModels.parsed_args = args

    # Pass args to unittest
    unittest.main(argv=[sys.argv[0]])
