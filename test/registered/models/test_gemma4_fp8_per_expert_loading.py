"""End-to-end test for compressed-tensors per-expert FP8 MoE checkpoint
loading on Gemma4 (e.g. RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic).

Regression coverage for the load_weights path that recognises
`experts.<id>.{gate,up,down}_proj.{weight,weight_scale}` keys and folds
them into SGLang's fused FusedMoE parameters. Without that path, all
routed-expert weights are silently skipped at load time and the model
emits only `<pad>` tokens at inference (GSM8K collapses to 0.0).
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Compressed-tensors per-expert FP8 MoE checkpoint that exercises the
# loader path (gated repo + ~27 GB download + 4 GPUs at TP=4).
register_cuda_ci(est_time=120, stage="stage-c", runner_config="4-gpu-h100")


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class TestGemma4FP8PerExpertLoading(CustomTestCase):
    """Three-stage check that catches the silent-skip failure mode:
    1. server health
    2. completion is not the all-`<pad>` garbage state
    3. GSM8K accuracy matches the BF16 baseline
    """

    model = "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--trust-remote-code",
                "--random-seed",
                "42",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health(self):
        r = requests.get(self.base_url + "/health")
        self.assertEqual(r.status_code, 200)

    def test_basic_generation_not_garbage(self):
        """Pre-fix the server starts but every routed expert is zero-init,
        which leads chat completions to deterministic `<pad>` spam."""
        r = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 7 + 5?"}],
                "temperature": 0,
                "max_tokens": 32,
            },
        )
        self.assertEqual(r.status_code, 200)
        text = r.json()["choices"][0]["message"]["content"]
        self.assertNotIn(
            "<pad>", text, f"Output looks like the pre-fix garbage state: {text!r}"
        )
        self.assertGreater(len(text.strip()), 0, "Empty completion")
        self.assertIn("12", text, f"Expected the answer to mention '12': {text!r}")

    def test_gsm8k_accuracy(self):
        """Pre-fix this scores exactly 0.00 (zero routed-expert weights);
        post-fix it matches the BF16 baseline (~0.95 on 20 samples)."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=20,
            num_threads=16,
        )
        metrics = run_eval(args)
        score = float(metrics["score"])
        print(f"Gemma4 FP8 per-expert GSM8K-20 score: {score:.3f}")
        # Threshold rules out the failure mode (0.00) while leaving ample
        # margin under the BF16 baseline (~0.95).
        self.assertGreaterEqual(
            score,
            0.80,
            f"Per-expert FP8 ckpt accuracy collapsed: {score} "
            "(pre-fix value is 0.00; BF16 baseline is ~0.95).",
        )


if __name__ == "__main__":
    unittest.main()
