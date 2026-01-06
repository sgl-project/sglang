"""MI35x Nightly Large Model Tests - 8 GPU

This test module runs large model tests on MI35x (gfx950) GPUs with 8 GPU (TP=8) configuration.
Tests large models like DeepSeek-V3, Grok, and other models requiring full 8-GPU tensor parallelism.

Runs on: linux-mi35x-gpu-8 runner

The model path can be configured via environment variables:
- DEEPSEEK_V3_MODEL_PATH: Path to DeepSeek-V3 model (default: deepseek-ai/DeepSeek-V3-0324)
"""

import os
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

# Register for AMD MI35x CI - 8 GPU large model tests (~120 min)
register_amd_ci(est_time=7200, suite="nightly-8-gpu-mi35x", nightly=True)

# Model score thresholds for MI35x 8-GPU tests
MODEL_SCORE_THRESHOLDS = {
    "deepseek-ai/DeepSeek-V3-0324": 0.90,
    "deepseek-ai/DeepSeek-V3.1": 0.90,
    "xai-org/grok-1": 0.85,
}

# Environment variable for model path override
DEEPSEEK_V3_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V3_MODEL_PATH", "deepseek-ai/DeepSeek-V3-0324"
)


class TestMI35xLarge8GPU(CustomTestCase):
    """Large model tests for MI35x with 8 GPU (TP=8) configuration."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.all_results = []

    def _launch_server(
        self, model: str, extra_args: list = None
    ) -> "subprocess.Popen":
        """Launch server with MI35x-optimized settings for TP=8."""
        other_args = [
            "--log-level-http",
            "warning",
            "--trust-remote-code",
            "--tp",
            "8",
            "--mem-fraction-static",
            "0.85",
        ]
        if extra_args:
            other_args.extend(extra_args)

        return popen_launch_server(
            model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,  # Longer timeout for large models
            other_args=other_args,
        )

    def _run_model_eval(
        self, model: str, extra_args: list = None, eval_name: str = "mgsm_en"
    ) -> dict:
        """Run evaluation for a model with timing."""
        model_start = time.time()
        server_start = time.time()
        process = self._launch_server(model, extra_args)
        startup_time = time.time() - server_start

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=model,
                eval_name=eval_name,
                num_examples=100,  # Subset for faster testing
                num_threads=512,
            )

            eval_start = time.time()
            metrics = run_eval(args)
            eval_time = time.time() - eval_start
            total_time = time.time() - model_start

            result = {
                "model": model,
                "score": metrics["score"],
                "tp_size": 8,
                "startup_time": startup_time,
                "eval_time": eval_time,
                "total_time": total_time,
            }
            self.all_results.append(result)

            print(f"\n📈 Results for {model} (TP=8):")
            print(f"   Score: {metrics['score']:.3f}")
            print(f"   Server startup: {startup_time:.1f}s")
            print(f"   Evaluation: {eval_time:.1f}s")
            print(f"   Total: {total_time:.1f}s")

            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_deepseek_v3_basic(self):
        """Test DeepSeek-V3 basic inference on MI35x with TP=8."""
        model = DEEPSEEK_V3_MODEL_PATH
        extra_args = [
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        metrics = self._run_model_eval(model, extra_args)
        threshold = MODEL_SCORE_THRESHOLDS.get(model, 0.85)
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} score {metrics['score']:.3f} below threshold {threshold}",
        )

    def test_deepseek_v3_mtp(self):
        """Test DeepSeek-V3 with MTP/EAGLE speculative decoding on MI35x with TP=8."""
        model = DEEPSEEK_V3_MODEL_PATH
        extra_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        metrics = self._run_model_eval(model, extra_args)
        threshold = MODEL_SCORE_THRESHOLDS.get(model, 0.85)
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} (MTP) score {metrics['score']:.3f} below threshold {threshold}",
        )

    @classmethod
    def tearDownClass(cls):
        """Generate summary report after all tests."""
        if cls.all_results:
            summary = "### MI35x 8-GPU Large Model Results\n\n"
            summary += "| Model | TP | Score | Threshold | Status |\n"
            summary += "| ----- | -- | ----- | --------- | ------ |\n"

            for result in cls.all_results:
                model = result["model"]
                score = result["score"]
                threshold = MODEL_SCORE_THRESHOLDS.get(model, 0.85)
                status = "✅ PASS" if score >= threshold else "❌ FAIL"
                summary += f"| {model} | 8 | {score:.3f} | {threshold:.2f} | {status} |\n"

            print(summary)
            if is_in_ci():
                write_github_step_summary(summary)


if __name__ == "__main__":
    unittest.main()
