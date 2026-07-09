"""Regression test for USPAttention GQA replicated-prefix head sharding.

``USPAttention._forward_with_replicated_prefix`` keeps a replicated token prefix
(e.g. text) out of the Ulysses all-to-all and slices that prefix down to the local
head shard. For a GQA model (kv heads < q heads) the K/V prefix must be sliced by
the *KV* head shard, not the query head shard -- otherwise the per-rank query
offset overshoots the KV head dim, the prefix slice is empty/mismatched, and the
``cat`` with the all-to-all'd suffix raises. MHA (kv heads == q heads) is unaffected.

Single-process test: the Ulysses world size, rank, all-to-all helpers, and
all_gather are mocked so the per-rank slicing logic runs on CPU.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"
_SP = 2


def _fake_input_all_to_all(x, **_):
    # Ulysses input all-to-all: gather sequence (xSP), shard heads (/SP). Only the
    # resulting shape matters for this slicing test.
    h = x.shape[2]
    return x[:, :, : h // _SP, :].repeat_interleave(_SP, dim=1).contiguous()


def _fake_output_all_to_all(x, **_):
    # Inverse of the input all-to-all: shard sequence (/SP), gather heads (xSP).
    s = x.shape[1]
    return x[:, : s // _SP, :, :].repeat_interleave(_SP, dim=2).contiguous()


class _CaptureAttn:
    """Stand-in attn backend that records the q/k/v it receives."""

    def __init__(self):
        self.q = self.k = self.v = None

    def forward(self, q, k, v, _ctx):
        self.q, self.k, self.v = q, k, v
        return q.clone()


class TestUSPAttentionReplicatedPrefix(unittest.TestCase):
    def _run(self, q_heads, kv_heads, sp_rank, num_rep=3, suffix=4, head_dim=4):
        attn = _CaptureAttn()
        obj = USPAttention.__new__(USPAttention)  # bypass __init__/backend setup
        obj.attn_impl = attn

        seq = num_rep + suffix
        q = torch.randn(1, seq, q_heads, head_dim)
        k = torch.randn(1, seq, kv_heads, head_dim)
        v = torch.randn(1, seq, kv_heads, head_dim)

        sp_group = MagicMock()
        sp_group.ulysses_group = None

        def fake_all_gather(out_list, tensor, **_):
            for t in out_list:
                t.copy_(tensor)

        with (
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=_SP),
            patch(f"{_LAYER}.get_sp_parallel_rank", return_value=sp_rank),
            patch(
                f"{_LAYER}._usp_input_all_to_all", side_effect=_fake_input_all_to_all
            ),
            patch(
                f"{_LAYER}._usp_output_all_to_all",
                side_effect=_fake_output_all_to_all,
            ),
            patch(f"{_LAYER}.get_sp_group", return_value=sp_group),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            out = USPAttention._forward_with_replicated_prefix(
                obj, q, k, v, None, num_rep
            )
        return attn, out, q.shape

    def test_gqa_slices_kv_prefix_by_kv_heads(self):
        # GQA: 8 query heads, 2 kv heads. The old code sliced the K/V prefix by the
        # query head shard, producing an empty/mismatched prefix and a cat error.
        for sp_rank in range(_SP):
            with self.subTest(sp_rank=sp_rank):
                attn, out, q_shape = self._run(q_heads=8, kv_heads=2, sp_rank=sp_rank)
                # q keeps q_heads/SP, k/v keep kv_heads/SP -> GQA grouping preserved.
                self.assertEqual(attn.q.shape[2], 8 // _SP)
                self.assertEqual(attn.k.shape[2], 2 // _SP)
                self.assertEqual(attn.v.shape[2], 2 // _SP)
                # prefix + all-to-all'd suffix line up on the sequence axis.
                self.assertEqual(attn.k.shape[1], attn.q.shape[1])
                # output is restored to the input layout.
                self.assertEqual(tuple(out.shape), tuple(q_shape))

    def test_mha_prefix_unchanged(self):
        # MHA: q heads == kv heads, so the KV-shard slicing is identical to before.
        attn, out, q_shape = self._run(q_heads=8, kv_heads=8, sp_rank=1)
        self.assertEqual(attn.q.shape[2], 8 // _SP)
        self.assertEqual(attn.k.shape[2], 8 // _SP)
        self.assertEqual(tuple(out.shape), tuple(q_shape))


if __name__ == "__main__":
    unittest.main()
