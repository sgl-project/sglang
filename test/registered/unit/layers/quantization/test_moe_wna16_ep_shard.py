import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Method
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 16
INTERMEDIATE = 16
GROUP = 8
MOE_TP_SIZE = 2


def _layer(moe_tp_rank, *, local_experts=None):
    """A layer on this rank; ``local_experts`` lists the global ids it stores."""
    local_experts = [0] if local_experts is None else local_experts
    quant_config = SimpleNamespace(
        has_zp=True, linear_quant_method="gptq", weight_bits=4
    )
    return SimpleNamespace(
        quant_config=quant_config,
        moe_tp_size=MOE_TP_SIZE,
        moe_tp_rank=moe_tp_rank,
        intermediate_size_per_partition=INTERMEDIATE // MOE_TP_SIZE,
        group_size_div_factor=1,
        layer_id=0,
        num_local_experts=len(local_experts),
        _map_global_expert_id_to_local_expert_id=lambda e: (
            local_experts.index(e) if e in local_experts else -1
        ),
    )


def _load_qzeros(layer, weight, expert_id=0):
    """Load one expert's GPTQ int4 qzeros through the layer's weight loader."""
    per_partition = layer.intermediate_size_per_partition
    if weight == "w2":
        # down_proj: (intermediate // group, hidden // 8) int32 in the checkpoint.
        checkpoint_shape = (INTERMEDIATE // GROUP, HIDDEN // 8)
        param_shape = (1, HIDDEN // 2, per_partition // GROUP)
        name, shard_id = "experts.w2_qzeros", "w2"
    else:
        # gate_proj: (hidden // group, intermediate // 8) int32 in the checkpoint.
        checkpoint_shape = (HIDDEN // GROUP, INTERMEDIATE // 8)
        param_shape = (1, per_partition, HIDDEN // GROUP)
        name, shard_id = "experts.w13_qzeros", "w1"
    numel = checkpoint_shape[0] * checkpoint_shape[1]
    checkpoint = torch.arange(numel, dtype=torch.int32).view(checkpoint_shape)
    checkpoint = checkpoint * 0x01010101
    param = torch.nn.Parameter(
        torch.zeros((layer.num_local_experts,) + param_shape[1:], dtype=torch.uint8),
        requires_grad=False,
    )
    loader = MoeWNA16Method.get_weight_loader(layer, Mock())
    loader(param, checkpoint, name, shard_id, expert_id)
    return param.data.clone()


@contextmanager
def _topology(*, tp_size, ep_size, world_rank):
    with published_topology(
        tp_size=tp_size, ep_size=ep_size, ranks=dict(world_rank=world_rank)
    ):
        device_group = SimpleNamespace(world_size=tp_size, device=torch.device("cpu"))
        with get_parallel().override(tp_group=device_group):
            yield


class TestMoeWNA16QzerosShardUnderEP(CustomTestCase):
    WEIGHTS = ("w2", "w13")

    def setUp(self):
        # Without EP the TP rank and the MoE-TP rank coincide.
        with _topology(tp_size=MOE_TP_SIZE, ep_size=1, world_rank=1):
            self.expected = {w: _load_qzeros(_layer(1), w) for w in self.WEIGHTS}
            self.other_shard = {w: _load_qzeros(_layer(0), w) for w in self.WEIGHTS}

    def test_the_two_shards_differ(self):
        for weight in self.WEIGHTS:
            with self.subTest(weight):
                self.assertFalse(
                    torch.equal(self.expected[weight], self.other_shard[weight])
                )

    def test_ep_layout_loads_the_moe_tp_shard(self):
        # TP 4 with EP 2: TP rank 3 holds MoE-TP shard 1 of 2.
        with _topology(tp_size=4, ep_size=2, world_rank=3):
            parallel = get_parallel()
            self.assertEqual((parallel.tp_rank, parallel.moe_tp_rank), (3, 1))
            layer = _layer(parallel.moe_tp_rank)
            loaded = {w: _load_qzeros(layer, w) for w in self.WEIGHTS}
        for weight in self.WEIGHTS:
            with self.subTest(weight):
                self.assertTrue(torch.equal(loaded[weight], self.expected[weight]))

    def test_ep_rank_writes_only_its_own_experts(self):
        # EP 2 over 4 experts: this rank stores global experts 2 and 3.
        with _topology(tp_size=4, ep_size=2, world_rank=3):
            layer = _layer(1, local_experts=[2, 3])
            for weight in self.WEIGHTS:
                with self.subTest(weight):
                    stored = _load_qzeros(layer, weight, expert_id=3)
                    self.assertTrue(torch.equal(stored[1], self.expected[weight][0]))
                    self.assertFalse(stored[0].any())
                    self.assertFalse(_load_qzeros(layer, weight, expert_id=0).any())


if __name__ == "__main__":
    unittest.main()
