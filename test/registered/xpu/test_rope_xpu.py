"""RotaryEmbedding.forward_xpu must not raise for full-rotary head sizes sgl_kernel lacks.

sgl_kernel.rotary_embedding on XPU only takes head sizes 32/64/96/128/256; any other
full-rotary head size (e.g. 80) must fall back to forward_native instead of raising.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


class TestRotaryEmbeddingXPU(CustomTestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(0)

    def _check(self, head_size: int, is_neox: bool):
        from sglang.srt.layers.rotary_embedding import get_rope

        rope = get_rope(
            head_size,
            rotary_dim=head_size,
            max_position=4096,
            base=1e6,
            is_neox_style=is_neox,
        ).to("xpu")
        num_tokens, num_heads, num_kv_heads = 7, 4, 2
        positions = torch.randint(0, 4096, (num_tokens,), device="xpu")
        q = torch.randn(
            num_tokens, num_heads * head_size, dtype=torch.bfloat16, device="xpu"
        )
        k = torch.randn(
            num_tokens, num_kv_heads * head_size, dtype=torch.bfloat16, device="xpu"
        )
        q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone())
        q_out, k_out = rope.forward_xpu(positions, q.clone(), k.clone())
        torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=1e-2)

    def test_head_size_without_xpu_kernel(self):
        # Qwen3-32B's EAGLE3 draft (AngelSlim/Qwen3-32B_eagle3) uses head_dim 80.
        for is_neox in (True, False):
            self._check(80, is_neox)


if __name__ == "__main__":
    unittest.main()
