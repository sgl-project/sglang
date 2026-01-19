"""
AMD VLM MMMU Evaluation Test - MI30x Only

This test evaluates Vision-Language Models (VLMs) on the MMMU benchmark on AMD GPUs.
Models are selected based on compatibility with AMD/ROCm platform.

VLMs tested here:
- Qwen VL series (Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen3-VL-30B)
- InternVL2 series (InternVL2_5-2B)
- MiniCPM series (MiniCPM-v-2_6, MiniCPM-o-2_6)
- DeepSeek VL series (deepseek-vl2-small, Janus-Pro-7B)
- Kimi VL (Kimi-VL-A3B-Instruct)
- MiMo VL (MiMo-VL-7B-RL)
- GLM VL (GLM-4.1V-9B-Thinking)

Note: NVILA models are excluded (NVIDIA-specific).
Note: This test runs only on MI30x runners (linux-mi325-gpu-2), not on MI35x.

Registry: nightly-amd-accuracy-2-gpu-vlm suite (2-GPU VLM tests)
"""

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

# Register for AMD CI - VLM MMMU evaluation tests (~120 min)
register_amd_ci(est_time=7200, suite="nightly-amd-accuracy-2-gpu-vlm", nightly=True)

# AMD-verified VLM models with conservative thresholds on 100 MMMU samples
# Format: (model_path, tp_size, accuracy_threshold, extra_args)
AMD_VLM_MODELS = [
    # Qwen VL series - well supported on AMD
    {
        "model_path": "Qwen/Qwen2-VL-7B-Instruct",
        "tp_size": 1,
        "accuracy_threshold": 0.30,
        "extra_args": ["--trust-remote-code"],
    },
    {
        "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "tp_size": 1,
        "accuracy_threshold": 0.33,
        "extra_args": ["--trust-remote-code"],
    },
    {
        "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "tp_size": 2,
        "accuracy_threshold": 0.29,
        "extra_args": ["--trust-remote-code"],
    },
    # InternVL2 - smaller model, good for testing
    {
        "model_path": "OpenGVLab/InternVL2_5-2B",
        "tp_size": 1,
        "accuracy_threshold": 0.29,
        "extra_args": ["--trust-remote-code"],
    },
    # MiniCPM series
    {
        "model_path": "openbmb/MiniCPM-v-2_6",
        "tp_size": 1,
        "accuracy_threshold": 0.25,
        "extra_args": ["--trust-remote-code"],
    },
    {
        "model_path": "openbmb/MiniCPM-o-2_6",
        "tp_size": 1,
        "accuracy_threshold": 0.32,
        "extra_args": ["--trust-remote-code"],
    },
    # DeepSeek VL series
    {
        "model_path": "deepseek-ai/deepseek-vl2-small",
        "tp_size": 1,
        "accuracy_threshold": 0.31,
        "extra_args": ["--trust-remote-code"],
    },
    {
        "model_path": "deepseek-ai/Janus-Pro-7B",
        "tp_size": 1,
        "accuracy_threshold": 0.28,
        "extra_args": ["--trust-remote-code"],
    },
    # Kimi VL - MoE
    {
        "model_path": "moonshotai/Kimi-VL-A3B-Instruct",
        "tp_size": 1,
        "accuracy_threshold": 0.26,
        "extra_args": ["--trust-remote-code"],
    },
    # MiMo VL
    {
        "model_path": "XiaomiMiMo/MiMo-VL-7B-RL",
        "tp_size": 1,
        "accuracy_threshold": 0.27,
        "extra_args": ["--trust-remote-code"],
    },
    # GLM VL
    {
        "model_path": "zai-org/GLM-4.1V-9B-Thinking",
        "tp_size": 1,
        "accuracy_threshold": 0.27,
        "extra_args": ["--trust-remote-code"],
    },
]

# Models that need special handling on AMD (MoE models)
TRITON_ATTENTION_MODELS = {
    "deepseek-ai/deepseek-vl2-small",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "moonshotai/Kimi-VL-A3B-Instruct",
}

# Models known to fail on AMD - exclude from testing
AMD_FAILING_VLM_MODELS = {
    # Add models here as they are discovered to fail
}


def get_active_models():
    """Get list of models to test, excluding known failures."""
    return [m for m in AMD_VLM_MODELS if m["model_path"] not in AMD_FAILING_VLM_MODELS]


class TestNightlyVLMMmmuEvalAMD(unittest.TestCase):
    """AMD VLM MMMU Evaluation Test.

    Tests Vision-Language Models on MMMU benchmark using AMD GPUs.
    """

    @classmethod
    def setUpClass(cls):
        cls.models = get_active_models()
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu_vlm_models(self):
        """Test all configured VLM models on MMMU benchmark."""
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []
        total_test_start = time.time()

        print(f"\n{'='*60}")
        print("AMD VLM MMMU Evaluation Test")
        print(f"{'='*60}")
        print(f"Benchmark: MMMU (100 samples)")
        print(f"Models to test: {len(self.models)}")
        for m in self.models:
            print(f"  - {m['model_path']} (TP={m['tp_size']})")
        print(f"{'='*60}\n")

        for model_config in self.models:
            model_path = model_config["model_path"]
            tp_size = model_config["tp_size"]
            accuracy_threshold = model_config["accuracy_threshold"]
            extra_args = model_config.get("extra_args", [])
            error_message = None

            with self.subTest(model=model_path):
                print(f"\n{'='*60}")
                print(f"Testing: {model_path} (TP={tp_size})")
                print(f"{'='*60}")

                model_start = time.time()
                startup_time = None
                eval_time = None
                score = None

                # Set AMD-specific environment variables
                if model_path in TRITON_ATTENTION_MODELS:
                    os.environ["SGLANG_USE_AITER"] = "0"
                else:
                    os.environ["SGLANG_USE_AITER"] = "1"

                # Build launch args
                other_args = list(extra_args)
                other_args.extend(["--log-level-http", "warning"])
                if tp_size > 1:
                    other_args.extend(["--tp", str(tp_size)])

                # Launch server with timing
                print(f"🚀 Launching server...")
                server_start = time.time()
                process = popen_launch_server(
                    model=model_path,
                    base_url=self.base_url,
                    other_args=other_args,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
                startup_time = time.time() - server_start
                print(f"⏱️  Server startup: {startup_time:.1f}s")

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model_path,
                        eval_name="mmmu",
                        num_examples=100,
                        num_threads=64,
                        max_tokens=30,
                    )

                    # Run evaluation with timing
                    print(f"📊 Running MMMU evaluation (100 samples)...")
                    eval_start = time.time()

                    # Retry up to 3 times
                    metrics = None
                    for attempt in range(3):
                        try:
                            metrics = run_eval(args)
                            score = metrics["score"]
                            if score >= accuracy_threshold:
                                break
                        except Exception as e:
                            print(f"   Attempt {attempt + 1} failed with error: {e}")
                            if attempt == 2:
                                raise

                    eval_time = time.time() - eval_start
                    total_time = time.time() - model_start

                    # Print results
                    print(f"\n📈 Results for {model_path}:")
                    print(
                        f"   Score: {score:.3f} (threshold: {accuracy_threshold:.2f})"
                    )
                    print(f"\n⏱️  Runtime breakdown:")
                    print(f"   Server startup: {startup_time:.1f}s")
                    print(f"   Evaluation: {eval_time:.1f}s")
                    print(f"   Total: {total_time:.1f}s")

                    passed = score >= accuracy_threshold
                    if passed:
                        print(f"\n   Status: ✅ PASSED")
                    else:
                        print(f"\n   Status: ❌ FAILED")

                    write_results_to_json(model_path, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append(
                        {
                            "model": model_path,
                            "tp_size": tp_size,
                            "score": score,
                            "threshold": accuracy_threshold,
                            "startup_time": startup_time,
                            "eval_time": eval_time,
                            "total_time": total_time,
                            "passed": passed,
                            "error": None,
                        }
                    )

                except Exception as e:
                    error_message = str(e)
                    total_time = time.time() - model_start
                    print(f"\n❌ Error evaluating {model_path}: {error_message}")
                    all_results.append(
                        {
                            "model": model_path,
                            "tp_size": tp_size,
                            "score": None,
                            "threshold": accuracy_threshold,
                            "startup_time": startup_time,
                            "eval_time": None,
                            "total_time": total_time,
                            "passed": False,
                            "error": error_message,
                        }
                    )

                finally:
                    print(f"\n🛑 Stopping server...")
                    kill_process_tree(process.pid)

        # Calculate total test runtime
        total_test_time = time.time() - total_test_start

        # Generate summary
        self._check_results(all_results, total_test_time)

    def _check_results(self, results, total_test_time):
        """Check results and generate summary."""
        failed_models = []
        passed_count = 0
        failed_count = 0

        summary = (
            "| Model | TP | Score | Threshold | Startup | Eval | Total | Status |\n"
        )
        summary += (
            "| ----- | -- | ----- | --------- | ------- | ---- | ----- | ------ |\n"
        )

        for result in results:
            model = result["model"]
            score = result["score"]
            tp_size = result["tp_size"]
            threshold = result["threshold"]
            startup_time = result.get("startup_time")
            eval_time = result.get("eval_time")
            total_time = result.get("total_time")
            error = result.get("error")

            if error:
                status = "❌ ERROR"
                failed_count += 1
                failed_models.append(f"- {model}: ERROR - {error[:100]}")
            elif result["passed"]:
                status = "✅ PASS"
                passed_count += 1
            else:
                status = "❌ FAIL"
                failed_count += 1
                failed_models.append(
                    f"- {model}: score={score:.4f}, threshold={threshold:.4f}"
                )

            # Format values
            score_str = f"{score:.3f}" if score is not None else "N/A"
            startup_str = f"{startup_time:.0f}s" if startup_time is not None else "N/A"
            eval_str = f"{eval_time:.0f}s" if eval_time is not None else "N/A"
            total_str = f"{total_time:.0f}s" if total_time is not None else "N/A"

            summary += f"| {model} | {tp_size} | {score_str} | {threshold:.2f} | {startup_str} | {eval_str} | {total_str} | {status} |\n"

        print(f"\n{'='*60}")
        print("SUMMARY - AMD VLM MMMU Evaluation")
        print(f"{'='*60}")
        print(summary)
        print(f"\n📊 Final Statistics:")
        print(f"   Passed: {passed_count}")
        print(f"   Failed: {failed_count}")
        print(
            f"\n⏱️  Total test runtime: {total_test_time:.1f}s ({total_test_time/60:.1f} min)"
        )

        if is_in_ci():
            write_github_step_summary(
                f"### TestNightlyVLMMmmuEvalAMD\n{summary}\n\n"
                f"**Total Runtime:** {total_test_time:.1f}s ({total_test_time/60:.1f} min)"
            )

        if failed_models:
            failure_msg = "\n".join(failed_models)
            raise AssertionError(f"The following models failed:\n{failure_msg}")


if __name__ == "__main__":
    unittest.main()
