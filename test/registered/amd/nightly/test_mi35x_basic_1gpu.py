"""MI35x Nightly Basic Tests - 1 GPU

This test module runs basic model tests on MI35x (gfx950) GPUs with 1 GPU configuration.
Tests basic model loading, inference, and accuracy using smaller models that fit in single GPU memory.

Runs on: linux-mi35x-gpu-1 runner
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Register for AMD MI35x CI - 1 GPU basic tests (~30 min)
register_amd_ci(est_time=1800, suite="nightly-1-gpu-mi35x", nightly=True)

# Model score thresholds for MI35x 1-GPU tests
MODEL_SCORE_THRESHOLDS = {
    "meta-llama/Llama-3.2-3B-Instruct": 0.55,
    "Qwen/Qwen2.5-7B-Instruct": 0.80,
    "Qwen/Qwen3-8B": 0.75,
}

# Models to test on 1 GPU
MI35X_1GPU_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


class TestMI35xBasic1GPU(CustomTestCase):
    """Basic model tests for MI35x with 1 GPU configuration."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _launch_server(self, model: str) -> "subprocess.Popen":
        """Launch server with MI35x-optimized settings."""
        other_args = [
            "--log-level-http",
            "warning",
            "--trust-remote-code",
        ]
        return popen_launch_server(
            model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _run_model_eval(self, model: str) -> dict:
        """Run mgsm_en evaluation for a model."""
        process = self._launch_server(model)
        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=model,
                eval_name="mgsm_en",
                num_examples=100,  # Use subset for faster testing
                num_threads=512,
            )
            metrics = run_eval(args)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_llama_3_2_3b(self):
        """Test Llama-3.2-3B-Instruct on MI35x."""
        model = "meta-llama/Llama-3.2-3B-Instruct"
        metrics = self._run_model_eval(model)
        threshold = MODEL_SCORE_THRESHOLDS[model]
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} score {metrics['score']:.3f} below threshold {threshold}",
        )

    def test_qwen2_5_7b(self):
        """Test Qwen2.5-7B-Instruct on MI35x."""
        model = "Qwen/Qwen2.5-7B-Instruct"
        metrics = self._run_model_eval(model)
        threshold = MODEL_SCORE_THRESHOLDS[model]
        self.assertGreaterEqual(
            metrics["score"],
            threshold,
            f"{model} score {metrics['score']:.3f} below threshold {threshold}",
        )


if __name__ == "__main__":
    unittest.main()
