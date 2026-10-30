# SPDX-License-Identifier: Apache-2.0
"""`_ipc_input_a2a_qkv` must decline cross-attention shapes.

It sizes one staging slot from `q` and reuses it for q, k and v, which only
holds when all three share a sequence length. Cross-attention with unequal
query and key/value lengths -- LTX-2's video-to-audio blocks, say -- has to fall
back to the general exchange, which handles them.
"""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers import usp


class TestIpcInputA2AQkvGuard(unittest.TestCase):
    def _call(self, q, k, v):
        # Pretend ulysses degree 2 so the guard, not the degree check, decides.
        with mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 2):
            return usp._ipc_input_a2a_qkv(q, k, v)

    def test_declines_when_kv_length_differs(self):
        q = torch.zeros(1, 1530, 8, 64)
        kv = torch.zeros(1, 43, 8, 64)
        self.assertIsNone(self._call(q, kv, kv))

    def test_declines_when_only_v_differs(self):
        q = torch.zeros(1, 128, 8, 64)
        k = torch.zeros(1, 128, 8, 64)
        v = torch.zeros(1, 64, 8, 64)
        self.assertIsNone(self._call(q, k, v))

    def test_self_attention_shapes_reach_the_ipc_path(self):
        # Without a real IPC group this returns None either way, so patch the
        # group lookup to prove the guard is not what rejected it.
        q = torch.zeros(1, 128, 8, 64)
        group = mock.MagicMock(return_value=None)
        with (
            mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 2),
            mock.patch.object(usp, "_ipc_ready_group", group),
        ):
            self.assertIsNone(usp._ipc_input_a2a_qkv(q, q.clone(), q.clone()))
        # Reached the group lookup, so the shape guard did not reject it.
        self.assertEqual(group.call_count, 1)

    def test_degree_other_than_two_declines(self):
        q = torch.zeros(1, 128, 8, 64)
        with mock.patch.object(usp, "get_ulysses_parallel_world_size", lambda: 4):
            self.assertIsNone(usp._ipc_input_a2a_qkv(q, q.clone(), q.clone()))


class TestReplicatedKvPrefixUsesIpcExchange(unittest.TestCase):
    """The replicated-KV-prefix Ulysses path exchanges q/k/v like the others: one
    IPC exchange when available, not three NCCL all-to-alls."""

    def test_split_path_takes_the_ipc_exchange(self):
        from sglang.multimodal_gen.runtime.layers.attention import layer

        heads, dim = 4, 8
        attn = layer.USPAttention.__new__(layer.USPAttention)
        attn.attn_impl = mock.MagicMock()
        attn.attn_impl.forward.side_effect = lambda q, k, v, _: q
        q, k_shard, v_shard = (torch.randn(1, 3, heads, dim) for _ in range(3))
        k_rep = v_rep = torch.randn(1, 2, heads, dim)
        gathered = tuple(torch.randn(1, 6, heads // 2, dim) for _ in range(3))
        ipc = mock.MagicMock(return_value=gathered)
        nccl = mock.MagicMock(side_effect=AssertionError("NCCL exchange"))
        with (
            mock.patch.object(layer, "_ipc_input_a2a_qkv", ipc),
            mock.patch.object(layer, "async_a2a_communicate", nccl),
            mock.patch.object(layer, "_usp_input_all_to_all", nccl),
            mock.patch.object(layer, "_usp_output_all_to_all", lambda x, head_dim: x),
            mock.patch.object(layer, "get_ulysses_parallel_rank", lambda: 0),
            mock.patch.object(layer, "get_ulysses_parallel_world_size", lambda: 2),
            mock.patch.object(layer, "get_ring_parallel_world_size", lambda: 1),
            mock.patch.object(
                layer.USPAttention, "_pipelined_ulysses", lambda *a, **k: None
            ),
        ):
            out = attn._forward_with_replicated_kv_prefix_split(
                q, k_rep, v_rep, k_shard, v_shard, None
            )
        ipc.assert_called_once_with(q, k_shard, v_shard)
        self.assertIs(out, gathered[0])


class TestIpcSkippedUnderCompile(unittest.TestCase):
    """The transport's custom-class handles cannot be traced: a compiled graph
    keeps the NCCL all-to-all instead of breaking at every exchange."""

    def test_ipc_group_is_unavailable_while_compiling(self):
        ready = mock.MagicMock(return_value=True)
        with (
            mock.patch.object(usp.torch.compiler, "is_compiling", lambda: True),
            mock.patch(
                "sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a.ipc_a2a_ready",
                ready,
            ),
        ):
            self.assertIsNone(usp._ipc_ready_group())
        ready.assert_not_called()


if __name__ == "__main__":
    unittest.main()
