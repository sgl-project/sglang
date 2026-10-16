"""Offline linear layouts retain immutable partition ownership during reload."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedParallelGroup,
    RowParallelLinear,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import rank_size
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinearLogicalGroups(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
        self.weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64

    def test_group_less_layouts_keep_frozen_ownership(self):
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        for selection, rank, size in (("tp", 3, 4), ("attn_tp", 1, 2)):
            for cls, axis in ((ColumnParallelLinear, 0), (RowParallelLinear, 1)):
                with self.subTest(group=selection, layer=cls.__name__):
                    kwargs = {"reduce_results": False} if axis == 1 else {}
                    layer = cls(8, 8, bias=False, parallel_group=selection, **kwargs)
                    owner = layer.tp_group
                    self.assertEqual(
                        (owner.rank_in_group, owner.world_size), (rank, size)
                    )
                    with self.assertRaises(AttributeError):
                        owner.rank_in_group = 0
                    with get_parallel().override(
                        tp_rank=0, attn_tp_rank=0, attn_dp_rank=0
                    ):
                        layer.weight.weight_loader(layer.weight, self.weight)
                    shard = self.weight.chunk(size, dim=axis)[rank]
                    torch.testing.assert_close(layer.weight, shard, rtol=0, atol=0)
                    x = self.x if axis == 0 else self.x[:, : 8 // size]
                    torch.testing.assert_close(layer(x)[0], F.linear(x, shard))
                    self.assertIs(layer.tp_group, owner)
                    self.assertFalse(hasattr(layer, "tp_rank"))
                    self.assertFalse(hasattr(layer, "tp_size"))
        qkv = QKVParallelLinear(
            8,
            2,
            4,
            2,
            bias=False,
            parallel_group="tp",
            kv_parallel_group=ReplicatedParallelGroup("tp", 2),
        )
        self.assertEqual(rank_size(qkv), (3, 4))
        self.assertEqual(rank_size(qkv, kv=True), (1, 2))


if __name__ == "__main__":
    unittest.main()
