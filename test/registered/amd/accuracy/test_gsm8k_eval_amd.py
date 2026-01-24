"""
AMD GSM8K Evaluation Test (Migrated from test/srt/nightly/)

This test evaluates instruction-tuned models on the mgsm_en benchmark using chat completions.
Models are tested with various TP configurations on AMD GPUs.

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
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
    write_results_to_json,
)

# Register for AMD CI - GSM8K evaluation tests (~60 min)
register_amd_ci(est_time=3600, suite="nightly-amd", nightly=True)

MODEL_SCORE_THRESHOLDS = {
    # Llama 3.1 series
    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
    # Llama 3.2 series (smaller models)
    "meta-llama/Llama-3.2-3B-Instruct": 0.55,
    # Mistral series
    "mistralai/Mistral-7B-Instruct-v0.3": 0.58,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.61,
    # DeepSeek series
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
    # Qwen2 series
    "Qwen/Qwen2-57B-A14B-Instruct": 0.86,
    "Qwen/Qwen2.5-7B-Instruct": 0.85,
    # Qwen3 series
    "Qwen/Qwen3-30B-A3B-Thinking-2507": 0.84,  # MoE model verified on MI300X
    "Qwen/Qwen3-8B": 0.77,
    # Google Gemma
    "google/gemma-2-27b-it": 0.91,
    "google/gemma-2-9b-it": 0.72,
    # "neuralmagic/gemma-2-2b-it-FP8": 0.4,  # Small 2B model - OOM on single GPU
    # FP8 quantized models
    "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8": 0.8,
    "neuralmagic/Mistral-7B-Instruct-v0.3-FP8": 0.54,
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-72B-Instruct-FP8": 0.94,
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8": 0.86,
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8": 0.62,
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8": 0.84,
}

failing_models = {
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8",  # RuntimeError: This GEMM is not supported!
    "zai-org/GLM-4.5-Air-FP8",  # TypeError: cannot unpack non-iterable ForwardMetadata object
    "google/gemma-2-9b-it",  # OOM on single GPU (exit code -9)
    "neuralmagic/gemma-2-2b-it-FP8",  # OOM on single GPU (exit code -9)
}


def remove_failing_models(model_str):
    models = model_str.split(",")
    filtered = [m for m in models if m not in failing_models]
    return ",".join(filtered)


DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1
)
DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2 = remove_failing_models(
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2
)

# AMD-specific models verified on MI300X
# TP1 models - smaller models that fit on single GPU
AMD_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 = remove_failing_models(
    "meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B,google/gemma-2-9b-it"
)
# TP2 models - larger models requiring 2 GPUs
AMD_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2 = remove_failing_models(
    "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

NO_MOE_PADDING_MODELS = {"neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8"}
DISABLE_HF_XET_MODELS = {
    "Qwen/Qwen2-57B-A14B-Instruct",
    "neuralmagic/Qwen2-57B-A14B-Instruct-FP8",
}
TRITON_MOE_MODELS = {
    "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8",
    "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
}
# AMD-specific models that need special launch config (matching in-house CI sanity_check.py)
# AMD_SPECIAL_CONFIG_MODELS = {
#     "Qwen/Qwen3-30B-A3B-Thinking-2507", # default config works
# }


def popen_launch_server_wrapper(base_url, model, is_tp2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if is_tp2:
        other_args.extend(["--tp", "2"])

    # Use same config as sanity_check.py for AMD-specific models (scaled for tp=2)
    # Original tp=8: chunked-prefill-size=130172, max-running-requests=128
    # Scaled tp=2:   chunked-prefill-size=32543,  max-running-requests=32
    # if model in AMD_SPECIAL_CONFIG_MODELS:
    #     other_args.extend([
    #         "--chunked-prefill-size", "32543",
    #         "--max-running-requests", "32",
    #         "--mem-fraction-static", "0.85",
    #         "--attention-backend", "aiter",
    #     ])

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
        tp_size = result.get("tp_size", 2)
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

        line = f"| {model} | {tp_size} | {score:.3f} | {threshold_str} | {startup_str} | {eval_str} | {total_str} | {status} |\n"
        summary += line

    print(f"\n{'='*60}")
    print("SUMMARY - TP=2 Instruction Models (mgsm_en)")
    print(f"{'='*60}")
    print(summary)
    print(f"\n📊 Final Statistics:")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {failed_count}")

    if is_in_ci():
        write_github_step_summary(f"### TestNightlyGsm8KEval (TP=2)\n{summary}")

    if failed_models:
        failure_msg = "\n".join(failed_models)
        raise AssertionError(f"The following models failed:\n{failure_msg}")


# Do not use `CustomTestCase` since `test_mgsm_en_all_models` does not want retry
class TestNightlyGsm8KEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
            # AMD-specific models verified on MI300X
            (parse_models(AMD_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            (parse_models(AMD_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mgsm_en_all_models(self):
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

        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):
                    tp_size = 2 if is_tp2 else 1
                    print(f"\n{'='*60}")
                    print(f"Testing: {model} (TP={tp_size}, FP8={is_fp8})")
                    print(f"{'='*60}")

                    model_start = time.time()
                    startup_time = None
                    eval_time = None

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
                    process = popen_launch_server_wrapper(self.base_url, model, is_tp2)
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
                            "tp_size": tp_size,
                            "is_fp8": is_fp8,
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
