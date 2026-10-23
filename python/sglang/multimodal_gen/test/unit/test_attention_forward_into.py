"""``forward_into`` writes through a strided ``out`` exactly what ``forward``
returns, and declines (writing nothing) where it cannot."""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Blackwell FA4 required",
)
class TestFlashAttentionForwardInto(unittest.TestCase):
    def setUp(self):
        self._fa_ver = flash_attn.fa_ver
        flash_attn.set_fa_ver(4)

    def tearDown(self):
        flash_attn.set_fa_ver(self._fa_ver)

    def test_strided_out_matches_forward_bitwise(self):
        torch.manual_seed(0)
        impl = FlashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        q, k, v = (
            torch.randn(2, 1000, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        self.assertTrue(impl.forward_into(q, k, v, None, None))
        # one head group inside a token-major buffer of every head
        full = torch.zeros(2, 1000, 12, 128, device="cuda", dtype=torch.bfloat16)
        out = full[:, :, 4:8]
        self.assertTrue(impl.forward_into(q, k, v, out, None))
        self.assertTrue(torch.equal(out, impl.forward(q, k, v, None)))
        self.assertFalse(full[:, :, :4].any() or full[:, :, 8:].any())


class TestForwardIntoDefault(unittest.TestCase):
    def test_backends_without_it_decline(self):
        impl = SDPAImpl(num_heads=2, head_size=64, causal=False, softmax_scale=0.125)
        q = torch.zeros(1, 8, 2, 64)
        self.assertFalse(impl.forward_into(q, q, q, None, None))


if __name__ == "__main__":
    unittest.main()
