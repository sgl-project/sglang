"""Ring + tail-pad dispatch: a2a within Ulysses, ring clamped to pad_start."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import _ring_merge_attention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class TestRingTailPadDispatch(unittest.TestCase):
    def _attn(self):
        obj = USPAttention.__new__(USPAttention)
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        obj.softmax_scale = 0.5
        obj.backend = AttentionBackendEnum.FA
        obj.causal = False
        obj.dropout_p = 0.0
        obj.attn_impl = SimpleNamespace(supports_ring_kv_chunk=lambda: True)
        return obj

    def test_tail_pad_meta_reaches_the_ring_kernel(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                self._check_tail_pad(batch_size)

    def _check_tail_pad(self, batch_size):
        obj = self._attn()
        q = torch.randn(batch_size, 4, 2, 8)
        meta = {"pad_start": 13, "pad_end": 16, "local_pad": 3}
        seen = {}

        def fake_ring(qc, kc, vc, *, attn_impl, real_seq_len, ring_ws):
            seen.update(
                shape=tuple(qc.shape),
                real=real_seq_len,
                ws=ring_ws,
                impl=attn_impl,
            )
            return torch.ones_like(qc)

        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=4),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=4),
            patch(
                f"{_LAYER}._usp_input_all_to_all_qkv",
                side_effect=lambda q, k, v: (q, k, v),
            ),
            patch(
                f"{_LAYER}._usp_output_all_to_all", side_effect=lambda t, head_dim: t
            ),
            patch(f"{_LAYER}._ring_attention_varlen", side_effect=fake_ring),
            patch(f"{_LAYER}.get_ring_parallel_rank", return_value=3),
        ):
            out = obj.forward(q, q, q, attn_mask_meta=meta)

        self.assertEqual(out.shape, q.shape)
        self.assertEqual(seen["real"], 13)
        self.assertEqual(seen["ws"], 4)
        self.assertIs(seen["impl"], obj.attn_impl)
        self.assertEqual(seen["shape"], (4, 2, 8) if batch_size == 1 else q.shape)
        # Last ring rank holds global rows [12, 16): row 13 onward is pad.
        self.assertTrue(torch.all(out[:, 1:] == 0))
        self.assertTrue(torch.all(out[:, :1] == 1))

    def test_batched_merge_matches_full_attention(self):
        torch.manual_seed(42)
        q = torch.randn(2, 3, 5, 8)
        k, v = torch.randn(2, 2, 3, 7, 8)
        logits = q @ k.transpose(-2, -1)
        expected = (logits.softmax(-1) @ v).transpose(1, 2)
        out = lse = None
        for start, end in ((0, 3), (3, 7)):
            scores = logits[..., start:end]
            partial = (scores.softmax(-1) @ v[..., start:end, :]).transpose(1, 2)
            out, lse = _ring_merge_attention(out, lse, partial, scores.logsumexp(-1))
        torch.testing.assert_close(out, expected)

    def test_explicit_mask_under_ring_still_refuses(self):
        obj = self._attn()
        q = torch.randn(1, 4, 2, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=4),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(NotImplementedError, "ring"):
                obj.forward(q, q, q, attn_mask=mask)

    def test_causal_tail_pad_refuses_local_only_causal_mask(self):
        obj = self._attn()
        obj.causal = True
        q = torch.randn(2, 4, 2, 8)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(NotImplementedError, "non-causal"):
                obj.forward(
                    q,
                    q,
                    q,
                    attn_mask_meta={"pad_start": 7, "pad_end": 8, "local_pad": 1},
                )


if __name__ == "__main__":
    unittest.main()
