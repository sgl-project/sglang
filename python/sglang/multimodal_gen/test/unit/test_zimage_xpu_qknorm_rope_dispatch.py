"""Z-Image's single-image XPU QK norm and RoPE dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits import zimage


class TestZImageXpuQKNormRopeDispatch(unittest.TestCase):
    def _forward(self, *, fuse=True, qk_norm=True, kv_heads=30, batch_size=1):
        seq_len, heads, head_dim = 4, 30, 128
        qkv = torch.zeros(
            batch_size,
            seq_len,
            (heads + 2 * kv_heads) * head_dim,
            dtype=torch.bfloat16,
        )
        attention = SimpleNamespace(
            qk_norm=qk_norm,
            enable_zimage_qk_fusion=fuse,
            use_fused_qkv=True,
            local_num_heads=heads,
            local_num_kv_heads=kv_heads,
            head_dim=head_dim,
            to_qkv=lambda _: (qkv, None),
            norm_q=object(),
            norm_k=object(),
            attn=lambda q, k, v, **kwargs: q,
            to_out=[lambda x: (x, None)],
        )
        cos = torch.ones(batch_size, seq_len, 64, dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        if batch_size == 1:
            cos, sin = cos[0], sin[0]

        with (
            patch.object(zimage, "_is_xpu", True, create=True),
            patch.object(zimage, "_is_cuda", False),
            patch.object(
                zimage,
                "apply_qk_norm_with_optional_rope",
                side_effect=lambda **kwargs: (kwargs["q"], kwargs["k"]),
            ) as norm_rope,
            patch.object(
                zimage, "_apply_rotary_emb", side_effect=lambda x, *args, **kwargs: x
            ) as rotary,
        ):
            output = zimage.ZImageAttention.forward(
                attention,
                torch.zeros(batch_size, seq_len, heads * head_dim),
                freqs_cis=(cos, sin),
            )
            return output, norm_rope.call_args, rotary.call_count

    def test_single_image_uses_cos_sin_cache(self):
        output, norm_rope, rotary_count = self._forward()
        self.assertEqual(output.shape, (1, 4, 3840))
        cache = norm_rope.kwargs["cos_sin_cache"]
        self.assertEqual(cache.shape, (4, 128))
        self.assertEqual(cache.dtype, torch.float32)
        self.assertTrue(norm_rope.kwargs["allow_inplace"])
        self.assertEqual(rotary_count, 0)

    def test_disabled_fusion_uses_separate_rope(self):
        _, norm_rope, rotary_count = self._forward(fuse=False)
        self.assertNotIn("cos_sin_cache", norm_rope.kwargs)
        self.assertEqual(rotary_count, 2)

    def test_different_q_k_head_counts_use_separate_rope(self):
        _, norm_rope, rotary_count = self._forward(kv_heads=15)
        self.assertNotIn("cos_sin_cache", norm_rope.kwargs)
        self.assertEqual(rotary_count, 2)

    def test_without_qk_norm_uses_separate_rope(self):
        _, norm_rope, rotary_count = self._forward(qk_norm=False)
        self.assertIsNone(norm_rope)
        self.assertEqual(rotary_count, 2)

    def test_batched_path_keeps_flat_cache_and_positions(self):
        _, norm_rope, rotary_count = self._forward(batch_size=2)
        self.assertEqual(norm_rope.kwargs["cos_sin_cache"].shape, (8, 128))
        torch.testing.assert_close(norm_rope.kwargs["positions"], torch.arange(8))
        self.assertEqual(rotary_count, 0)


if __name__ == "__main__":
    unittest.main()
