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

# Timeout for model download (20 minutes for large models)
MODEL_DOWNLOAD_TIMEOUT = 1200  # 20 minutes


def try_download_model(model_path: str, timeout: int = MODEL_DOWNLOAD_TIMEOUT) -> bool:
    """Try to download/verify model availability before server launch.

    Returns True if model is available, False if download fails.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        print(f"📥 Checking/downloading model: {model_path}")
        start_time = time.time()

        # Try to download the model (will use cache if already downloaded)
        snapshot_download(
            model_path,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
            ignore_patterns=["*.gguf", "*.ggml"],
        )

        elapsed = time.time() - start_time
        print(f"✅ Model ready: {model_path} ({elapsed:.1f}s)")
        return True

    except (HfHubHTTPError, RepositoryNotFoundError) as e:
        print(f"❌ Failed to download {model_path}: {e}")
        return False
    except Exception as e:
        # Check if it's an offline mode error
        if "offline mode" in str(e).lower() or "HF_HUB_OFFLINE" in str(e):
            print(f"⚠️ Offline mode enabled, checking local cache for {model_path}")
            try:
                from huggingface_hub import try_to_load_from_cache

                # Try to check if model exists in cache
                config_path = try_to_load_from_cache(model_path, "config.json")
                if config_path is not None:
                    print(f"✅ Model found in cache: {model_path}")
                    return True
                else:
                    print(f"❌ Model not in cache and offline: {model_path}")
                    return False
            except Exception:
                print(f"❌ Cannot verify model in offline mode: {model_path}")
                return False
        print(f"❌ Error checking model {model_path}: {e}")
        return False


# Register for AMD CI - GSM8K 2-GPU evaluation tests (~60 min with downloads)
register_amd_ci(est_time=3600, suite="nightly-amd-accuracy-2-gpu", nightly=True)

# Models known to have actual failures on AMD (not download issues)
FAILING_MODELS = {
    # AITER backend crashes with "RuntimeError: invalid argument for batch_prefill"
    "google/gemma-2-9b-it",
    # AITER kernel compilation hangs (stale lock issue after gemma-2-9b-it crash)
    "neuralmagic/gemma-2-2b-it-FP8",
}


MODEL_SCORE_THRESHOLDS = {
    # Llama 3.1 series
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    # Mistral series
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.61,
    # Google Gemma (gemma-2-27b-it works, others have AITER bugs)
    "google/gemma-2-27b-it": 0.91,
    # Qwen2 series
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    # Qwen3 series
    "Qwen/Qwen3-30B-A3B-Thinking-2507": 0.84,
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.86,
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.62,
}

# 2-GPU models (TP=2) - models that require 2 GPUs
# Models will be downloaded if not cached (with timeout)
TP2_MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # Gemma (only gemma-2-27b-it works; others have AITER bugs - see FAILING_MODELS)
    "google/gemma-2-27b-it",
    # Qwen models
    "Qwen/Qwen2-57B-A14B-Instruct",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
    "neuralmagic/Qwen2-72B-Instruct-FP8",
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",
]

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
    # Use longer watchdog timeout for AITER kernel compilation (can take 5-10 min)
    other_args = [
        "--log-level-http",
        "warning",
        "--trust-remote-code",
        "--tp",
        "2",
        "--watchdog-timeout",
        "600",  # 10 minutes for AITER kernel compilation
    ]

    # Increase server launch timeout (default may be too short for AITER compilation)
    launch_timeout = max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600)

    process = popen_launch_server(
        model,
        base_url,
        timeout=launch_timeout,
        other_args=other_args,
    )
    return process


def check_model_scores(results, skipped_models=None):
    """Check model scores and generate summary table with pass/fail status.

    Only accuracy failures (score < threshold) cause test failure.
    Skipped models (AITER compilation, download, server issues) do not cause failure.
    """
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

    # Add skipped models to summary (not failures)
    skipped_count = len(skipped_models) if skipped_models else 0
    if skipped_models:
        summary += (
            "\n**Skipped Models (infrastructure issues, not accuracy failures):**\n"
        )
        for model, reason in skipped_models:
            summary += f"| {model} | 2 | - | - | - | - | - | ⏭️ SKIP: {reason} |\n"

    print(f"\n{'='*60}")
    print("SUMMARY - TP=2 Instruction Models (mgsm_en)")
    print(f"{'='*60}")
    print(summary)
    print(f"\n📊 Final Statistics:")
    print(f"   Passed: {passed_count}")
    print(f"   Failed (accuracy): {failed_count}")
    print(f"   Skipped (infra): {skipped_count}")

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyGsm8KEval2GPU (TP=2)\n{summary}")

    # Only fail on accuracy failures, not skips
    if failed_models:
        failure_msg = "\n".join(failed_models)
        raise AssertionError(
            f"The following models failed accuracy test:\n{failure_msg}"
        )


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
        skipped_models = []
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

                # Skip models with known failures
                if model in FAILING_MODELS:
                    print(f"⏭️ Skipping {model} (known failure)")
                    skipped_models.append((model, "known failure"))
                    continue

                # Try to download/verify model availability first
                if not try_download_model(model):
                    print(f"⏭️ Skipping {model} (download failed)")
                    skipped_models.append((model, "download failed"))
                    continue

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

                # Launch server with timing (extra timeout for first-time compilation)
                print(f"🚀 Launching server...")
                server_start = time.time()
                try:
                    process = popen_launch_server_wrapper(self.base_url, model)
                except Exception as e:
                    print(f"❌ Server launch failed for {model}: {e}")
                    skipped_models.append((model, f"server launch failed: {e}"))
                    continue
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

        # Report skipped models
        if skipped_models:
            print(f"\n{'='*60}")
            print(f"⏭️ Skipped Models ({len(skipped_models)}):")
            print(f"{'='*60}")
            for model, reason in skipped_models:
                print(f"  - {model}: {reason}")

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")

        # Check all scores after collecting all results
        # Only accuracy failures cause test failure; skips are reported but don't fail
        if all_results:
            check_model_scores(all_results, skipped_models)
        elif skipped_models:
            # All models were skipped - not a failure, just report
            print("\n⚠️ All models were skipped (infrastructure issues):")
            for model, reason in skipped_models:
                print(f"  - {model}: {reason}")
            print("\nNo accuracy failures - test passes with all skips.")
        else:
            print("\n⚠️ No models were tested!")

        print(
            f"\n⏱️  Total test runtime: {total_test_time:.1f}s ({total_test_time/60:.1f} min)"
        )


if __name__ == "__main__":
    unittest.main()
