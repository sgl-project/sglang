"""The static no-pack varlen path: fixed shapes, pad clamped by cu values."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class TestStaticNoPackVarlen(unittest.TestCase):
    def test_full_buffers_and_cu_values_reach_the_kernel(self):
        obj = USPAttention.__new__(USPAttention)
        obj.skip_sequence_parallel = False
        obj.softmax_scale = 0.5
        obj.backend = AttentionBackendEnum.FA

        seq, valid, heads, dim = 8, 5, 2, 4
        q = torch.randn(1, seq, heads, dim)
        cu = torch.tensor([0, valid], dtype=torch.int32)
        seen = {}

        def fake_varlen(**kw):
            seen.update(kw)
            return torch.zeros(seq, heads, dim)

        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
            patch(f"{_LAYER}.flash_attn_varlen_func", side_effect=fake_varlen),
        ):
            out = obj.forward(
                q, q, q, attn_mask_meta={"static_no_pack_cu": cu}
            )

        self.assertEqual(out.shape, (1, seq, heads, dim))
        self.assertEqual(seen["q"].shape, (seq, heads, dim))
        self.assertIs(seen["cu_seqlens_q"], cu)
        self.assertEqual(seen["max_seqlen_q"], seq)


if __name__ == "__main__":
    unittest.main()
