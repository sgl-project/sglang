"""MI35x Nightly Evaluation Tests - 2 GPU

This test module runs model evaluation tests on MI35x (gfx950) GPUs with 2 GPU (TP=2) configuration.
Tests larger models that require tensor parallelism across 2 GPUs.

Runs on: linux-mi35x-gpu-2 runner
"""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD MI35x CI - 2 GPU evaluation tests (~60 min)
register_amd_ci(est_time=3600, suite="nightly-2-gpu-mi35x", nightly=True)

# Model score thresholds for MI35x 2-GPU tests
MODEL_SCORE_THRESHOLDS = {
    "meta-llama/Llama-3.1-70B-Instruct": 0.90,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": 0.80,
    "google/gemma-2-27b-it": 0.88,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.58,
}

# Models requiring TP=2 on MI35x
MI35X_2GPU_MODELS = [
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


class TestMI35xEval2GPU(CustomTestCase):
    """Evaluation tests for MI35x with 2 GPU (TP=2) configuration."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.all_results = []

    def _launch_server(self, model: str) -> "subprocess.Popen":
        """Launch server with MI35x-optimized settings for TP=2."""
        other_args = [
            "--log-level-http",
            "warning",
            "--trust-remote-code",
            "--tp",
            "2",
        ]
        return popen_launch_server(
            model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _run_model_eval(self, model: str) -> dict:
        """Run mgsm_en evaluation for a model with timing."""
        model_start = time.time()
        server_start = time.time()
        process = self._launch_server(model)
        startup_time = time.time() - server_start

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=model,
                eval_name="mgsm_en",
                num_examples=None,  # Full evaluation
                num_threads=1024,
            )

            eval_start = time.time()
            metrics = run_eval(args)
            eval_time = time.time() - eval_start
            total_time = time.time() - model_start

            result = {
                "model": model,
                "score": metrics["score"],
                "tp_size": 2,
                "startup_time": startup_time,
                "eval_time": eval_time,
                "total_time": total_time,
            }
            self.all_results.append(result)

            print(f"\n📈 Results for {model} (TP=2):")
            print(f"   Score: {metrics['score']:.3f}")
            print(f"   Server startup: {startup_time:.1f}s")
            print(f"   Evaluation: {eval_time:.1f}s")
            print(f"   Total: {total_time:.1f}s")

            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_qwen3_30b_moe(self):
        """Test Qwen3-30B-A3B MoE model on MI35x with TP=2."""
        model = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        metrics = self._run_model_eval(model)
        threshold = MODEL_SCORE_THRESHOLDS[model]
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} score {metrics['score']:.3f} below threshold {threshold}",
        )

    def test_mixtral_8x7b(self):
        """Test Mixtral-8x7B MoE model on MI35x with TP=2."""
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        metrics = self._run_model_eval(model)
        threshold = MODEL_SCORE_THRESHOLDS[model]
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} score {metrics['score']:.3f} below threshold {threshold}",
        )

    @classmethod
    def tearDownClass(cls):
        """Generate summary report after all tests."""
        if cls.all_results:
            summary = "### MI35x 2-GPU Evaluation Results\n\n"
            summary += "| Model | TP | Score | Threshold | Status |\n"
            summary += "| ----- | -- | ----- | --------- | ------ |\n"

            for result in cls.all_results:
                model = result["model"]
                score = result["score"]
                threshold = MODEL_SCORE_THRESHOLDS.get(model, 0)
                status = "✅ PASS" if score >= threshold else "❌ FAIL"
                summary += f"| {model} | 2 | {score:.3f} | {threshold:.2f} | {status} |\n"

            print(summary)
            if is_in_ci():
                write_github_step_summary(summary)


if __name__ == "__main__":
    unittest.main()
