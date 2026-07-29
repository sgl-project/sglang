import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, suite="nightly-4-gpu")

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestPrefetchCheckpointsMultiGPU(CustomTestCase):
    """Verify that --weight-loader-prefetch-checkpoints works with DP attention."""

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path="Qwen/Qwen1.5-MoE-A2.7B-Chat",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            disable_radix_cache=True,
            weight_loader_prefetch_checkpoints=True,
            cuda_graph_max_bs=1,
            max_total_tokens=256,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()

    def test_generate_with_prefetch(self):
        """Server launched with prefetch must produce valid output."""
        outputs = self.engine.generate(PROMPTS)
        self.assertEqual(len(outputs), len(PROMPTS))
        for i, output in enumerate(outputs):
            text = output["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Prompt {i} produced empty output")


if __name__ == "__main__":
    unittest.main()
