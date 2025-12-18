"""
AMD GSM8K Completion Evaluation Test

This test uses the completion-based gsm8k benchmark (few-shot prompting)
which works with base models that don't have chat templates.

This complements test_gsm8k_eval_amd.py which uses mgsm_en (chat completions)
for instruction-tuned models.

Base models tested here:
- GPT-OSS series (lmsys/gpt-oss-*-bf16)
- GROK series (amd--grok-1-*, lmzheng-grok-1, grok-2)

Reference: benchmark/gsm8k/bench_sglang.py
"""

import ast
import json
import os
import re
import subprocess
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


@dataclass
class BaseModelConfig:
    """Configuration for a base model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    tokenizer_path: Optional[str] = None
    timeout: Optional[int] = None  # Custom timeout for server launch (seconds)

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


# AMD TP=8 base models for gsm8k completion benchmark
# These models work with completion API but not chat completions
#
# NOTE: These models require in-house CI resources (not public HuggingFace models)
# The upstream CI runner (linux-mi325-gpu-8) may not have access to these models.
# If models are not available, the test will be skipped.
#
# Model path mapping for different platforms:
# - MI300X (in-house): lmsys/gpt-oss-*-bf16
# - MI325X (in-house): openai/gpt-oss-*
AMD_BASE_MODELS_TP8 = [
    # GPT-OSS-20B - smaller model, run first for faster feedback
    BaseModelConfig(
        model_path="lmsys/gpt-oss-20b-bf16",
        tp_size=8,
        accuracy_threshold=0.50,
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
    # GPT-OSS-120B - large model, needs longer timeout
    BaseModelConfig(
        model_path="lmsys/gpt-oss-120b-bf16",
        tp_size=8,
        accuracy_threshold=0.82,
        timeout=900,  # 15 minutes for 120B model
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
    # GROK1-FP8 - uses aiter backend, needs extended timeout for kernel compilation
    BaseModelConfig(
        model_path="lmzheng-grok-1",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=600,  # 10 minutes for kernel compilation
        tokenizer_path="Xenova--grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "0",
        },
    ),
    # GROK1-IN4 - INT4 quantized version
    BaseModelConfig(
        model_path="amd--grok-1-W4A8KV8",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=600,  # 10 minutes for kernel compilation
        tokenizer_path="Xenova--grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "1",
        },
    ),
]


def check_model_available(model_path: str) -> bool:
    """Check if a model is available (either locally cached or on HuggingFace).
    
    Returns True if model can be loaded, False otherwise.
    """
    if not HF_HUB_AVAILABLE:
        # If huggingface_hub not available, assume model exists and let server fail if not
        return True
    
    try:
        # Try to get the model path - this checks local cache first
        from huggingface_hub import hf_hub_download, HfFileSystem
        fs = HfFileSystem()
        # Check if model exists by listing files (works for both local cache and remote)
        files = fs.ls(model_path, detail=False)
        return len(files) > 0
    except Exception as e:
        # Model not found or error accessing it
        print(f"Model {model_path} not available: {e}")
        return False


# For 2-GPU testing (scaled down from TP=8)
AMD_BASE_MODELS_TP2 = [
    BaseModelConfig(
        model_path="lmsys/gpt-oss-20b-bf16",
        tp_size=2,
        accuracy_threshold=0.50,
        other_args=[
            "--chunked-prefill-size",
            "32543",  # Scaled from 130172 / 4
            "--max-running-requests",
            "32",  # Scaled from 128 / 4
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "0"},
    ),
]


def get_one_example(lines, i, include_answer):
    """Format a single GSM8K example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """Get k few-shot examples for prompting."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Extract numerical answer from response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """
    Run GSM8K few-shot completion benchmark.

    Returns:
        Tuple of (accuracy, invalid_rate, latency)
    """
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    # Download and load data
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    # Construct prompts
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    # Define sglang function
    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    # Set backend
    backend = RuntimeEndpoint(base_url)
    sgl.set_default_backend(backend)

    # Run benchmark
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Extract predictions
    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute metrics
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


def popen_launch_server_for_base_model(
    base_url: str,
    config: BaseModelConfig,
) -> "subprocess.Popen":
    """Launch server for a base model with appropriate configuration."""
    # Build environment - start with current env and add config-specific vars
    env = os.environ.copy()
    for key, value in config.env_vars.items():
        env[key] = value
        print(f"Setting env: {key}={value}")

    # Build other_args
    other_args = list(config.other_args)
    other_args.extend(["--tp", str(config.tp_size)])
    other_args.extend(["--log-level-http", "warning"])

    if config.tokenizer_path:
        other_args.extend(["--tokenizer-path", config.tokenizer_path])

    # Use custom timeout if provided, otherwise use default
    timeout = config.timeout if config.timeout else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    process = popen_launch_server(
        model=config.model_path,
        base_url=base_url,
        timeout=timeout,
        other_args=other_args,
        env=env,  # Pass environment explicitly
    )
    return process


class TestNightlyGsm8kCompletionEvalAMD(unittest.TestCase):
    """
    AMD GSM8K Completion Evaluation Test

    Tests base models using few-shot completion benchmark.
    This is different from mgsm_en which uses chat completions.
    """

    @classmethod
    def setUpClass(cls):
        # Select models based on available GPUs
        # For now, default to TP=8 models (8-GPU runner)
        # Can be overridden by environment variable
        gpu_count = int(os.environ.get("AMD_GPU_COUNT", "8"))
        if gpu_count >= 8:
            cls.models = AMD_BASE_MODELS_TP8
        else:
            cls.models = AMD_BASE_MODELS_TP2

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_gsm8k_completion_all_models(self):
        """Test all configured base models with GSM8K completion benchmark."""
        all_results = []
        summary = "| Model | TP | Accuracy | Threshold | Invalid | Latency | Status |\n"
        summary += "| ----- | -- | -------- | --------- | ------- | ------- | ------ |\n"

        for config in self.models:
            with self.subTest(model=config.model_path):
                print(f"\n{'='*60}")
                print(f"Testing: {config.model_path} (TP={config.tp_size})")
                print(f"{'='*60}")

                error_message = None
                acc, invalid, latency = None, None, None
                skipped = False

                # Check if model is available before trying to launch
                if not check_model_available(config.model_path):
                    print(f"⏭️ SKIPPED: Model {config.model_path} not available")
                    status = "⏭️ SKIPPED"
                    skipped = True
                    all_results.append({
                        "model": config.model_path,
                        "tp_size": config.tp_size,
                        "accuracy": None,
                        "threshold": config.accuracy_threshold,
                        "invalid": None,
                        "latency": None,
                        "passed": True,  # Don't count as failure
                        "skipped": True,
                        "error": "Model not available",
                    })
                else:
                    try:
                        # Launch server
                        process = popen_launch_server_for_base_model(
                            self.base_url, config
                        )

                        try:
                            # Run benchmark
                            acc, invalid, latency = run_gsm8k_benchmark(
                                self.base_url,
                                num_questions=self.num_questions,
                                num_shots=5,
                                parallel=64,
                            )

                            print(f"Accuracy: {acc:.3f} (threshold: {config.accuracy_threshold})")
                            print(f"Invalid: {invalid:.3f}")
                            print(f"Latency: {latency:.1f}s")

                            passed = acc >= config.accuracy_threshold
                            status = "✅ PASS" if passed else "❌ FAIL"

                            all_results.append({
                                "model": config.model_path,
                                "tp_size": config.tp_size,
                                "accuracy": acc,
                                "threshold": config.accuracy_threshold,
                                "invalid": invalid,
                                "latency": latency,
                                "passed": passed,
                                "skipped": False,
                                "error": None,
                            })

                        except Exception as e:
                            error_message = str(e)
                            print(f"Error during benchmark: {error_message}")
                            status = "❌ ERROR"
                            all_results.append({
                                "model": config.model_path,
                                "tp_size": config.tp_size,
                                "accuracy": None,
                                "threshold": config.accuracy_threshold,
                                "invalid": None,
                                "latency": None,
                                "passed": False,
                                "skipped": False,
                                "error": error_message,
                            })

                        finally:
                            kill_process_tree(process.pid)

                    except Exception as e:
                        error_message = str(e)
                        print(f"Error launching server: {error_message}")
                        status = "❌ ERROR"
                        all_results.append({
                            "model": config.model_path,
                            "tp_size": config.tp_size,
                            "accuracy": None,
                            "threshold": config.accuracy_threshold,
                            "invalid": None,
                            "latency": None,
                            "passed": False,
                            "skipped": False,
                            "error": error_message,
                        })

                # Add to summary
                acc_str = f"{acc:.3f}" if acc is not None else "N/A"
                invalid_str = f"{invalid:.3f}" if invalid is not None else "N/A"
                latency_str = f"{latency:.1f}s" if latency is not None else "N/A"
                summary += f"| {config.model_path} | {config.tp_size} | {acc_str} | {config.accuracy_threshold} | {invalid_str} | {latency_str} | {status} |\n"

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(summary)

        # Write GitHub step summary
        if is_in_ci():
            write_github_step_summary(f"### TestNightlyGsm8kCompletionEvalAMD\n{summary}")

        # Check for failures (exclude skipped models)
        failed_models = [r for r in all_results if not r["passed"] and not r.get("skipped", False)]
        skipped_models = [r for r in all_results if r.get("skipped", False)]
        
        if skipped_models:
            print(f"\n⏭️ Skipped {len(skipped_models)} model(s) (not available)")
        
        if failed_models:
            failure_msg = "\n".join([
                f"- {r['model']}: accuracy={r['accuracy']}, threshold={r['threshold']}, error={r['error']}"
                for r in failed_models
            ])
            raise AssertionError(f"The following models failed:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()

