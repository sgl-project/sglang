import unittest

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestPrefetchCheckpointsMultiNPU(CustomTestCase):
    """Verify that --weight-loader-prefetch-checkpoints works with DP attention."""

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=QWEN3_32B_WEIGHTS_PATH,
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            disable_radix_cache=True,
            weight_loader_prefetch_checkpoints=True,
            attention_backend="ascend",
            disable_cuda_graph=True,
            device="npu",
            max_total_tokens=512,
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
