"""The ROCm AITer attention backend must accept grouped-query attention.

Cosmos3's cross-attention is GQA (32 query heads / 8 KV heads). The backend
used to reject any ``num_kv_heads != num_heads`` at construction time, which
made every GQA DiT unloadable on ROCm. ``aiter.flash_attn_func`` broadcasts KV
heads itself, so the only restriction left is the FP8 FMHA ASM kernel, which
stays MHA-only.
"""

import unittest

import torch

try:
    import aiter  # noqa: F401

    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import AITerImpl

    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


HEAD_DIM = 128


@unittest.skipUnless(AITER_AVAILABLE, "aiter is unavailable (non-ROCm platform)")
class TestAITerGroupedQueryAttention(unittest.TestCase):
    def _impl(self, num_heads: int, num_kv_heads: int | None) -> "AITerImpl":
        return AITerImpl(
            num_heads=num_heads,
            head_size=HEAD_DIM,
            softmax_scale=HEAD_DIM**-0.5,
            num_kv_heads=num_kv_heads,
        )

    def test_gqa_head_counts_are_accepted(self):
        # Cosmos3-Nano cross-attention.
        impl = self._impl(num_heads=32, num_kv_heads=8)
        self.assertEqual(impl.num_kv_heads, 8)
        self.assertTrue(impl.is_gqa)

    def test_mqa_head_counts_are_accepted(self):
        impl = self._impl(num_heads=32, num_kv_heads=1)
        self.assertTrue(impl.is_gqa)

    def test_num_kv_heads_defaults_to_num_heads(self):
        impl = self._impl(num_heads=32, num_kv_heads=None)
        self.assertEqual(impl.num_kv_heads, 32)
        self.assertFalse(impl.is_gqa)

    def test_indivisible_head_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            self._impl(num_heads=32, num_kv_heads=7)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU is unavailable")
    def test_gqa_forward_matches_sdpa(self):
        torch.manual_seed(0)
        batch, seq_len, num_heads, num_kv_heads = 1, 512, 32, 8
        shape_q = (batch, seq_len, num_heads, HEAD_DIM)
        shape_kv = (batch, seq_len, num_kv_heads, HEAD_DIM)
        q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)

        out = self._impl(num_heads=num_heads, num_kv_heads=num_kv_heads).forward(
            q, k, v
        )

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)
        self.assertEqual(out.shape, ref.shape)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
