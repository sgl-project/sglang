"""
Benchmark script for evaluating VLMs using lmms-eval.
This script launches a model server using SGLang and runs evaluations against
specified multimodal tasks. It handles server setup, evaluation running,
and result collection in one streamlined process.

Prerequisites:
    You need to install lmms-eval first:
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
    cd lmms-eval
    pip install -e .

Example usage:
    python bench_lmm_evals_sglang.py \
  --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --chat-template "qwen2-vl" \
  --tasks "mmmu_pro,mmmu_val" \
  --mem-fraction-static 0.6 \
  --timeout 300 \
  --batch-size 1 \
  --tp 1

Supported tasks list: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks
"""

import argparse
import glob
import json
import os
import subprocess

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def execute_lmms_evaluation(
    model_version: str,
    chat_template: str,
    output_path: str,
    tasks: str,
    batch_size: int,
    tp: int,
):
    """
    Evaluate a VLM on the selected tasks with lmms‑eval.
    Only `model_version` (checkpoint) and `chat_template` vary;
    """
    # -------- fixed settings --------
    model = "openai_compatible"
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


def run_eval(args):
    """
    Evaluate a VLM on the selected tasks with lmms‑eval.
    """
    print(f"\nTesting model: {args.model_path}")

    try:
        # Launch server for testing
        process = popen_launch_server(
            args.model_path,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=args.timeout,
            api_key="sk-456",
            other_args=[
                "--chat-template",
                args.chat_template,
                "--trust-remote-code",
                "--mem-fraction-static",
                str(args.mem_fraction_static),  # Use class variable
            ],
        )

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = "sk-456"
        os.environ["OPENAI_API_BASE"] = f"{DEFAULT_URL_FOR_TEST}/v1"

        # Run evaluation using lmm-evals
        execute_lmms_evaluation(
            args.model_path,
            args.chat_template,
            "./logs",
            args.tasks,
            args.batch_size,
            args.tp,
        )
        print("Evaluation completed.")

        # Get the result file
        result_files = glob.glob("./logs/*.json")
        if result_files:
            result_file_path = result_files[0]

            with open(result_file_path, "r") as f:
                try:
                    result = json.load(f)
                    print(f"Result:\n{result}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON file: {e}")
        else:
            print(
                "No result files found in the logs directory. Evaluation may have failed."
            )

    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Kill the server process
        if process and process.poll() is None:
            kill_process_tree(process.pid)
            print("Server process killed.")
        else:
            print("Server process already terminated.")


if __name__ == "__main__":
    # Define and parse arguments here, before unittest.main
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model",
        required=True,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Chat template to use",
        required=True,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Multi Modal task to be evaluated. Tasks should be comma seperated values without space. Example: --tasks=mmmu_pro,mmmu_val",
        required=True,
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Static memory fraction for the model",
        default=0.6,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for server launch",
        default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for evaluation",
        default=1,
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="Tensor parallelism degree",
        default=1,
    )

    # Parse args intended for unittest
    args = parser.parse_args()

    run_eval(args)
