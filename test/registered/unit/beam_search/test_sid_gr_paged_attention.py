"""SID arithmetic adapter: causal suffixes and exact paged ancestry gathering."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from examples.runtime.sid_gr.paged_attention import (
    SIDPagedAttentionBackend,
    register_paged_reference_backend,
)
from sglang.srt.beam_search.paged_kv import GRPagedForwardMetadata
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSIDPagedAttention(unittest.TestCase):
    def backend(self, tp=1):
        backend = SIDPagedAttentionBackend.__new__(SIDPagedAttentionBackend)
        backend.page_size = 4
        backend.tp_size = tp
        return backend

    def test_registration_requires_explicit_opt_in(self):
        original = ATTENTION_BACKENDS["gr_paged"]
        sentinel = object()
        try:
            for value in ("", "0", "1"):
                ATTENTION_BACKENDS["gr_paged"] = sentinel
                with patch.dict(os.environ, SGLANG_SID_GR_PAGED_REFERENCE=value):
                    register_paged_reference_backend()
                expected = SIDPagedAttentionBackend if value == "1" else sentinel
                self.assertIs(ATTENTION_BACKENDS["gr_paged"], expected)
        finally:
            ATTENTION_BACKENDS["gr_paged"] = original

    def test_cold_prefill_is_exact_sid_math_with_fragmented_pages(self):
        torch.manual_seed(314)
        q = torch.randn(7, 4, 8).bfloat16()
        k = torch.randn(7, 2, 8).bfloat16()
        v = torch.randn_like(k)
        slots = torch.tensor([12, 13, 14, 15, 4, 5, 6])
        key = torch.full((20, 2, 8), torch.nan, dtype=q.dtype)
        value = torch.full_like(key, torch.nan)
        key[slots], value[slots] = k, v
        meta = SimpleNamespace(
            context_pages=torch.tensor([3, 1]), context_len=7, prefix_len=0
        )
        actual = self.backend()._prefill_attention(q, key, value, meta, 8**-0.5)
        with sdpa_kernel(SDPBackend.MATH):
            expected = (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None],
                    k.transpose(0, 1)[None].repeat_interleave(2, 1),
                    v.transpose(0, 1)[None].repeat_interleave(2, 1),
                    is_causal=True,
                    scale=8**-0.5,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_cached_suffix_has_absolute_causal_positions(self):
        # Query at absolute position 3 sees four values, never the future 1000.
        q = torch.zeros(2, 4, 2)
        key = torch.full((32, 1, 2), torch.nan)
        value = torch.full_like(key, torch.nan)
        slots = torch.tensor([12, 13, 14, 15, 24])
        key[slots] = 0
        value[slots] = torch.tensor([1, 2, 3, 4, 1000.0])[:, None, None]
        meta = SimpleNamespace(
            context_pages=torch.tensor([3, 6]), context_len=5, prefix_len=3
        )
        actual = self.backend()._prefill_attention(q, key, value, meta, 2**-0.5)
        expected = torch.tensor([2.5, 202.0])[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_decode_gather_excludes_siblings_and_padding_and_retains_tp_geometry(self):
        # Two children share the first parent; sibling slot 21 and padding are
        # NaN. The backend must gather 20 twice and each child's current slot.
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                heads, kv_heads, dim = 16 // tp, 4 // tp, 2
                q = torch.zeros(2, heads, dim)
                key = torch.full((32, kv_heads, dim), torch.nan)
                value = torch.full_like(key, torch.nan)
                used = torch.tensor([12, 13, 14, 20, 24, 25])
                key[used] = used[:, None, None].float()
                value[used] = -key[used]
                meta = GRPagedForwardMetadata(
                    page_size=4,
                    context_len=3,
                    beam_width=2,
                    step=1,
                    context_locs=torch.tensor([12, 13, 14]),
                    context_pages=torch.tensor([3]),
                    beam_pages=torch.tensor([[5], [6]]),
                    topk_indices=torch.tensor([[0, 0], [0, 1]]),
                    out_cache_loc=torch.tensor([24, 25]),
                )
                with patch(
                    "examples.runtime.sid_gr.paged_attention.beam_decode_with_reference_splits",
                    side_effect=lambda query, *a, **kw: query,
                ) as kernel:
                    actual = self.backend(tp)._decode_attention(
                        q, key, value, meta, dim**-0.5
                    )
                torch.testing.assert_close(actual, q, atol=0, rtol=0)
                args = kernel.call_args.args
                torch.testing.assert_close(args[1], key[[12, 13, 14]][None])
                torch.testing.assert_close(args[2], value[[12, 13, 14]][None])
                torch.testing.assert_close(args[3], key[[20, 20, 24, 25]][None])
                torch.testing.assert_close(args[4], value[[20, 20, 24, 25]][None])
                self.assertEqual(args[5].shape, (1, 1, heads, 2, 2))
                self.assertEqual(args[5][0, 0, 0].tolist(), [[0, 1], [2, 3]])
                self.assertEqual(args[6], 2)
                self.assertEqual(kernel.call_args.kwargs, {"reference_heads": 16})
                self.assertEqual(args[3].shape[-2], kv_heads)


if __name__ == "__main__":
    unittest.main()
