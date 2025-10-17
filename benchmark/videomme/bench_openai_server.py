"""
Bench an OpenAI-compatible server with benchmark Video-MME using lmms-eval.

Usage:
    Step 1: Launch a server
    (e.g.) python -m sglang.launch_server --model-path <path-to-vlm-model> --port 30000

    Step 2: Run benchmark
    python benchmark/videomme/bench_openai_server.py --port 30000 --model-path <path-to-vlm-model>
"""

import argparse
import glob
import json
import os
import subprocess
from argparse import Namespace

from eval_utils import EvalArgs

from sglang.test.test_utils import add_common_sglang_args_and_parse


def main(args: Namespace):
    # Set OpenAI API key and base URL environment variables.
    # lmms-eval's openai_compatible model uses these env vars.
    api_key = "sk-123456"
    base_url = f"{args.host}:{args.port}/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    print(f"Using OpenAI API Base: {base_url}")

    # lmms_eval settings from test_vlm_models.py
    model = "openai_compatible"
    tasks = "videomme"
    tasks = "video_mmmu_adaptation_question_only"
    log_suffix = "openai_compatible"
    os.makedirs(args.output_path, exist_ok=True)

    # compose --model_args
    # `model_version` is passed to the `model` parameter in OpenAI API call.
    # SGLang server will use the model it was launched with.
    # model_args = f""

    # build command list
    cmd = [
        "python3",
        "-m",
        "lmms_eval",
        "--model",
        model,
        "--tasks",
        tasks,
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--log_samples_suffix",
        log_suffix,
        "--output_path",
        str(args.output_path),
        f"--limit={args.limit}",
    ]

    print("\nRunning lmms_eval command:")
    print(" ".join(cmd))

    try:
        subprocess.run(
            cmd,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError running lmms_eval: {e}")
        print("Please make sure 'lmms-eval' is installed and in your PATH.")
        print(
            "You can install it with: pip install 'lmms-eval @ git+https://github.com/EvolvingLMMs-Community/lm-evaluation-harness.git@v0.0.1'"
        )
        return

    # Print results
    result_files = glob.glob(f"{args.output_path}/*.json")
    if not result_files:
        print("\nNo result file found in the output directory.")
        return

    # The result file is usually named after the model, tasks, etc.
    # and contains a summary. We find the most recent one.
    result_files.sort(key=os.path.getmtime, reverse=True)
    result_file_path = result_files[0]

    print(f"\nReading results from: {result_file_path}")
    with open(result_file_path, "r") as f:
        result = json.load(f)
        print("\n--- Evaluation Result ---")
        print(json.dumps(result, indent=2))
        print("--- End of Result ---")


def parse_args():
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    args = add_common_sglang_args_and_parse(parser)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
