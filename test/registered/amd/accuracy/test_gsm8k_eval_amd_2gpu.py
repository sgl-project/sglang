"""
AMD GSM8K Evaluation Test - 2-GPU Models (TP=2) - MI30x Only

This test evaluates instruction-tuned models on the mgsm_en benchmark using chat completions.
Models are tested with TP=2 configuration on 2 AMD MI300X/MI325X GPUs.

Note: This test runs only on MI30x runners (linux-mi325-gpu-2), not on MI35x.

Registry: nightly-amd suite (2-GPU tests)
"""

import json
import os
import time
import unittest
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
    write_results_to_json,
)

# Register for AMD CI - GSM8K 2-GPU evaluation tests (~30 min)
register_amd_ci(est_time=1800, suite="nightly-amd-accuracy-2-gpu", nightly=True)

# Models known to fail on AMD - not cached locally or other issues
FAILING_MODELS = {
    # Models not cached locally on CI runner
    "Qwen/Qwen2-57B-A14B-Instruct",  # Not cached locally
    "Qwen/Qwen3-30B-A3B-Thinking-2507",  # Not cached locally
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",  # Not cached locally
    "neuralmagic/Qwen2-72B-Instruct-FP8",  # Not cached locally
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",  # Not cached locally
}


def remove_failing_models(models_list):
    """Remove models known to fail on AMD from the test list."""
    return [m for m in models_list if m not in FAILING_MODELS]


MODEL_SCORE_THRESHOLDS = {
    # Llama 3.1 series
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    # Mistral series
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.61,
    # Google Gemma (moved from 1-GPU due to OOM/AITER kernel compile time)
    "google/gemma-2-27b-it": 0.91,
    "google/gemma-2-9b-it": 0.72,
    # Qwen2 series
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    # Qwen3 series
    "Qwen/Qwen3-30B-A3B-Thinking-2507": 0.84,
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.86,
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.62,
    "neuralmagic/gemma-2-2b-it-FP8": 0.50,  # Moved from 1-GPU (OOM)
}

# 2-GPU models (TP=2) - models that require 2 GPUs
_TP2_MODELS_ALL = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # Gemma models (moved from 1-GPU due to OOM/AITER kernel compile time)
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "neuralmagic/gemma-2-2b-it-FP8",
    # Qwen models
    "Qwen/Qwen2-57B-A14B-Instruct",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
    "neuralmagic/Qwen2-72B-Instruct-FP8",
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",
]

# Filter out models that aren't cached locally
TP2_MODELS = remove_failing_models(_TP2_MODELS_ALL)

NO_MOE_PADDING_MODELS = {"neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"}
DISABLE_HF_XET_MODELS = {
    "Qwen/Qwen2-57B-A14B-Instruct",
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",
}
TRITON_MOE_MODELS = {
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


def popen_launch_server_wrapper(base_url, model):
    """Launch server with TP=2 configuration."""
    other_args = ["--log-level-http", "warning", "--trust-remote-code", "--tp", "2"]

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


def check_model_scores(results):
    """Check model scores and generate summary table with pass/fail status."""
    failed_models = []
    passed_count = 0
    failed_count = 0

    summary = "| Model | TP | Score | Threshold | Startup | Eval | Total | Status |\n"
    summary += "| ----- | -- | ----- | --------- | ------- | ---- | ----- | ------ |\n"

    for result in results:
        model = result["model"]
        score = result["score"]
        startup_time = result.get("startup_time")
        eval_time = result.get("eval_time")
        total_time = result.get("total_time")

        threshold = MODEL_SCORE_THRESHOLDS.get(model)
        if threshold is None:
            print(f"Warning: No threshold defined for model {model}")
            status = "⚠️ NO THRESHOLD"
        elif score >= threshold:
            status = "✅ PASS"
            passed_count += 1
        else:
            status = "❌ FAIL"
            failed_count += 1
            failed_models.append(
                f"- {model}: score={score:.4f}, threshold={threshold:.4f}"
            )

        # Format times
        startup_str = f"{startup_time:.0f}s" if startup_time is not None else "N/A"
        eval_str = f"{eval_time:.0f}s" if eval_time is not None else "N/A"
        total_str = f"{total_time:.0f}s" if total_time is not None else "N/A"
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"

        line = f"| {model} | 2 | {score:.3f} | {threshold_str} | {startup_str} | {eval_str} | {total_str} | {status} |\n"
        summary += line

    print(f"\n{'='*60}")
    print("SUMMARY - TP=2 Instruction Models (mgsm_en)")
    print(f"{'='*60}")
    print(summary)
    print(f"\n📊 Final Statistics:")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {failed_count}")

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyGsm8KEval2GPU (TP=2)\n{summary}")

    if failed_models:
        failure_msg = "\n".join(failed_models)
        raise AssertionError(f"The following models failed:\n{failure_msg}")


class TestNightlyGsm8KEval2GPU(unittest.TestCase):
    """AMD GSM8K Evaluation Test for 2-GPU models (TP=2)."""

    @classmethod
    def setUpClass(cls):
        cls.models = TP2_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mgsm_en_2gpu_models(self):
        """Test all 2-GPU models with mgsm_en benchmark."""
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []
        total_test_start = time.time()

        print(f"\n{'='*60}")
        print("AMD GSM8K Evaluation Test (TP=2 Instruction Models)")
        print(f"{'='*60}")
        print(f"Benchmark: mgsm_en (chat completions)")
        print(f"{'='*60}\n")

        for model in self.models:
            with self.subTest(model=model):
                print(f"\n{'='*60}")
                print(f"Testing: {model} (TP=2)")
                print(f"{'='*60}")

                model_start = time.time()

                os.environ["SGLANG_MOE_PADDING"] = (
                    "0" if model in NO_MOE_PADDING_MODELS else "1"
                )
                os.environ["HF_HUB_DISABLE_XET"] = (
                    "1" if model in DISABLE_HF_XET_MODELS else "0"
                )
                os.environ["SGLANG_USE_AITER"] = (
                    "0" if model in TRITON_MOE_MODELS else "1"
                )

                # Launch server with timing
                print(f"🚀 Launching server...")
                server_start = time.time()
                process = popen_launch_server_wrapper(self.base_url, model)
                startup_time = time.time() - server_start
                print(f"⏱️  Server startup: {startup_time:.1f}s")

                args = SimpleNamespace(
                    base_url=self.base_url,
                    model=model,
                    eval_name="mgsm_en",
                    num_examples=None,
                    num_threads=1024,
                )

                # Run eval with timing and retries
                print(f"📊 Running mgsm_en evaluation...")
                eval_start = time.time()
                threshold = MODEL_SCORE_THRESHOLDS.get(model)
                metrics = None
                for attempt in range(3):
                    try:
                        metrics = run_eval(args)
                        score = metrics["score"]
                        if threshold and score >= threshold:
                            break
                    except Exception as e:
                        print(f"   Attempt {attempt + 1} failed with error: {e}")
                eval_time = time.time() - eval_start
                total_time = time.time() - model_start

                # Print results
                score = metrics["score"] if metrics else 0.0
                threshold_str = f"{threshold:.2f}" if threshold else "N/A"
                passed = threshold and score >= threshold

                print(f"\n📈 Results for {model}:")
                print(f"   Score: {score:.3f} (threshold: {threshold_str})")
                print(f"\n⏱️  Runtime breakdown:")
                print(f"   Server startup: {startup_time:.1f}s")
                print(f"   Evaluation: {eval_time:.1f}s")
                print(f"   Total: {total_time:.1f}s")

                if passed:
                    print(f"\n   Status: ✅ PASSED")
                else:
                    print(f"\n   Status: ❌ FAILED")

                write_results_to_json(model, metrics, "w" if is_first else "a")
                is_first = False

                all_results.append(
                    {
                        "model": model,
                        "score": score,
                        "startup_time": startup_time,
                        "eval_time": eval_time,
                        "total_time": total_time,
                    }
                )

                print(f"\n🛑 Stopping server...")
                kill_process_tree(process.pid)

        # Calculate total test runtime
        total_test_time = time.time() - total_test_start

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")

        # Check all scores after collecting all results
        check_model_scores(all_results)
        print(
            f"\n⏱️  Total test runtime: {total_test_time:.1f}s ({total_test_time/60:.1f} min)"
        )


if __name__ == "__main__":
    unittest.main()
