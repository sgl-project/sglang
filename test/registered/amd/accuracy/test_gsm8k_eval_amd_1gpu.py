"""
AMD GSM8K Evaluation Test - 1-GPU Models (TP=1) - MI30x Only

This test evaluates instruction-tuned models on the mgsm_en benchmark using chat completions.
Models are tested with TP=1 configuration on a single AMD MI300X/MI325X GPU.

Note: This test runs only on MI30x runners (linux-mi325-gpu-2), not on MI35x.

Registry: nightly-amd-1-gpu suite
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

# Register for AMD CI - GSM8K 1-GPU evaluation tests (~30 min)
register_amd_ci(est_time=1800, suite="nightly-amd-accuracy-1-gpu", nightly=True)

MODEL_SCORE_THRESHOLDS = {
    # Llama 3.1 series
    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
    # Llama 3.2 series (smaller models)
    "meta-llama/Llama-3.2-3B-Instruct": 0.55,
    # Mistral series
    "mistralai/Mistral-7B-Instruct-v0.3": 0.58,
    # DeepSeek series
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
    # Qwen2 series
    "Qwen/Qwen2.5-7B-Instruct": 0.85,
    # Qwen3 series
    "Qwen/Qwen3-8B": 0.77,
    # Google Gemma
    "google/gemma-2-27b-it": 0.91,
    "google/gemma-2-9b-it": 0.72,
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.8,
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.54,
}

# Models known to fail on AMD
FAILING_MODELS = {
    "google/gemma-2-9b-it",  # OOM on single GPU (exit code -9)
    "neuralmagic/gemma-2-2b-it-FP8",  # OOM on single GPU (exit code -9)
}

# 1-GPU models (TP=1) - models that fit on a single GPU
TP1_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "google/gemma-2-27b-it",
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

TRITON_MOE_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.3",
}


def popen_launch_server_wrapper(base_url, model):
    """Launch server with appropriate configuration."""
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]

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

        line = f"| {model} | 1 | {score:.3f} | {threshold_str} | {startup_str} | {eval_str} | {total_str} | {status} |\n"
        summary += line

    print(f"\n{'='*60}")
    print("SUMMARY - TP=1 Instruction Models (mgsm_en)")
    print(f"{'='*60}")
    print(summary)
    print(f"\n📊 Final Statistics:")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {failed_count}")

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyGsm8KEval1GPU (TP=1)\n{summary}")

    if failed_models:
        failure_msg = "\n".join(failed_models)
        raise AssertionError(f"The following models failed:\n{failure_msg}")


class TestNightlyGsm8KEval1GPU(unittest.TestCase):
    """AMD GSM8K Evaluation Test for 1-GPU models (TP=1)."""

    @classmethod
    def setUpClass(cls):
        cls.models = [m for m in TP1_MODELS if m not in FAILING_MODELS]
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mgsm_en_1gpu_models(self):
        """Test all 1-GPU models with mgsm_en benchmark."""
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []
        total_test_start = time.time()

        print(f"\n{'='*60}")
        print("AMD GSM8K Evaluation Test (TP=1 Instruction Models)")
        print(f"{'='*60}")
        print(f"Benchmark: mgsm_en (chat completions)")
        print(f"{'='*60}\n")

        for model in self.models:
            with self.subTest(model=model):
                print(f"\n{'='*60}")
                print(f"Testing: {model} (TP=1)")
                print(f"{'='*60}")

                model_start = time.time()

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
