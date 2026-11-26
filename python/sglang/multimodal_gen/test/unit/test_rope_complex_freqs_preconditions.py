"""Guards against complex_freqs silently mixing RoPE styles or crashing on a
confusing broadcast error instead of failing at the actual precondition.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import apply_qk_norm_rope
from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding


def _complex_freqs(seq_len: int, half_dim: int) -> torch.Tensor:
    cos = torch.randn(seq_len, 1, half_dim)
    sin = torch.randn(seq_len, 1, half_dim)
    return torch.complex(cos, sin)


class TestRotaryEmbeddingComplexFreqsPreconditions(unittest.TestCase):
    def test_neox_style_with_only_complex_freqs_fails_instead_of_wrong_result(self):
        head_size = 64
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            is_neox_style=True,
            use_precomputed_cache=False,
        )
        seq_len = 5
        query = torch.randn(1, seq_len, 2, head_size, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        complex_freqs = _complex_freqs(seq_len, head_size // 2)

        with self.assertRaisesRegex(ValueError, "No valid inputs"):
            rope.forward_native(query=query, key=key, complex_freqs=complex_freqs)

    def test_partial_rotary_dim_with_only_complex_freqs_fails_instead_of_crashing(self):
        head_size, rotary_dim = 8, 4
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            is_neox_style=False,
            use_precomputed_cache=False,
        )
        seq_len = 3
        query = torch.randn(1, seq_len, 2, head_size, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        # Sized by rotary_dim, as a real caller's derived table would be;
        # this is the shape that used to crash inside view_as_complex.
        complex_freqs = _complex_freqs(seq_len, rotary_dim // 2)

        with self.assertRaisesRegex(ValueError, "No valid inputs"):
            rope.forward_native(query=query, key=key, complex_freqs=complex_freqs)


class TestApplyQkNormRopeRequiresCache(unittest.TestCase):
    def test_raises_without_cos_sin_cache(self):
        # apply_qk_norm_with_optional_rope only reaches apply_qk_norm_rope
        # when cos_sin_cache is not None; this pins down that apply_qk_norm_rope
        # itself still enforces that precondition for its other direct
        # callers (cosmos3video.py, ernie_image.py, zimage.py, ...), which
        # never go through the wrapper. Passing freqs_complex must not let a
        # caller substitute it for cos_sin_cache -- same raise either way.
        head_dim = 8
        seq_len = 3
        q = torch.randn(1, seq_len, 2, head_dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        freqs_complex = _complex_freqs(seq_len, head_dim // 2).squeeze(1)

        with self.assertRaisesRegex(
            ValueError, "cos_sin_cache must be a 2D torch.Tensor"
        ):
            apply_qk_norm_rope(
                q=q,
                k=k,
                q_norm=None,
                k_norm=None,
                head_dim=head_dim,
                cos_sin_cache=None,
                freqs_complex=freqs_complex,
            )


if __name__ == "__main__":
    unittest.main()
