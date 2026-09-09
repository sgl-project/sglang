"""Guards against complex_freqs silently mixing RoPE styles or crashing on a
confusing broadcast error instead of falling back to cos_sin_cache or failing.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm_with_optional_rope,
)
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

    def test_falls_back_to_cache_ignoring_unsupported_complex_freqs_neox(self):
        # Real callers always pass complex_freqs alongside a cos_sin_cache
        # derived from the same table; when the style can't use complex_freqs
        # (here: NeoX), it must be ignored rather than misapplied, so the
        # output must be identical to the cache-only call.
        head_size = 64
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            is_neox_style=True,
            use_precomputed_cache=False,
        )
        seq_len = 5
        torch.manual_seed(0)
        query = torch.randn(1, seq_len, 2, head_size, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        cos_sin_cache = torch.randn(seq_len, head_size)
        complex_freqs = _complex_freqs(seq_len, head_size // 2)

        q_with_complex, k_with_complex = rope.forward_native(
            query=query,
            key=key,
            complex_freqs=complex_freqs,
            cos_sin_cache=cos_sin_cache,
        )
        q_cache_only, k_cache_only = rope.forward_native(
            query=query, key=key, complex_freqs=None, cos_sin_cache=cos_sin_cache
        )
        self.assertTrue(torch.equal(q_with_complex, q_cache_only))
        self.assertTrue(torch.equal(k_with_complex, k_cache_only))

    def test_falls_back_to_cache_ignoring_unsupported_complex_freqs_partial_rotary_dim(
        self,
    ):
        # Same fallback contract as the NeoX case above, but for the other
        # way support_complex_style can be False: rotary_dim < head_size.
        # The cos_sin_cache branch derives its rotation width from the cache
        # tensor's own shape (not self.rotary_dim), so a cache sized to
        # rotary_dim=4 must still rotate only the first 4 of 8 dims and
        # ignore complex_freqs, identically to a call without it.
        head_size, rotary_dim = 8, 4
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            is_neox_style=False,
            use_precomputed_cache=False,
        )
        seq_len = 3
        torch.manual_seed(0)
        query = torch.randn(1, seq_len, 2, head_size, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        cos_sin_cache = torch.randn(seq_len, rotary_dim)
        complex_freqs = _complex_freqs(seq_len, rotary_dim // 2)

        q_with_complex, k_with_complex = rope.forward_native(
            query=query,
            key=key,
            complex_freqs=complex_freqs,
            cos_sin_cache=cos_sin_cache,
        )
        q_cache_only, k_cache_only = rope.forward_native(
            query=query, key=key, complex_freqs=None, cos_sin_cache=cos_sin_cache
        )
        self.assertTrue(torch.equal(q_with_complex, q_cache_only))
        self.assertTrue(torch.equal(k_with_complex, k_cache_only))
        # The tail past rotary_dim must be passed through untouched.
        self.assertTrue(
            torch.equal(q_cache_only[..., rotary_dim:], query[..., rotary_dim:])
        )

    def test_accepts_full_interleaved_rotation(self):
        head_size = 64
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            is_neox_style=False,
            use_precomputed_cache=False,
        )
        seq_len = 5
        query = torch.randn(1, seq_len, 2, head_size, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        complex_freqs = _complex_freqs(seq_len, head_size // 2)

        q_out, k_out = rope.forward_native(
            query=query, key=key, complex_freqs=complex_freqs
        )
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)


class TestApplyQkNormWithOptionalRopeRequiresCache(unittest.TestCase):
    def test_skips_rope_when_only_complex_freqs_given_without_cache(self):
        # cos_sin_cache is the sole trigger for the RoPE branch; freqs_complex
        # alone must not silently substitute for it (apply_qk_norm_rope
        # requires a real cos_sin_cache internally and would raise), so the
        # call must fall back to plain qk-norm with freqs_complex ignored,
        # matching a call that never passed freqs_complex at all.
        head_dim = 8
        seq_len = 3
        q = torch.randn(1, seq_len, 2, head_dim, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        q_norm = RMSNorm(head_dim, eps=1e-6).to(dtype=torch.bfloat16)
        k_norm = RMSNorm(head_dim, eps=1e-6).to(dtype=torch.bfloat16)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)
        freqs_complex = torch.complex(cos, sin)

        q_out, k_out = apply_qk_norm_with_optional_rope(
            q=q,
            k=k,
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            cos_sin_cache=None,
            freqs_complex=freqs_complex,
        )
        q_ref, k_ref = apply_qk_norm_with_optional_rope(
            q=q,
            k=k,
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            cos_sin_cache=None,
            freqs_complex=None,
        )
        self.assertTrue(torch.equal(q_out, q_ref))
        self.assertTrue(torch.equal(k_out, k_ref))


if __name__ == "__main__":
    unittest.main()
