"""Muse Spark Hopper setup: one-host peer memory, BF16 expert loads after the repack."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.muse_spark import muse_spark_bf16_moe as bf16
from sglang.srt.layers.muse_spark import muse_spark_dynamic_guests as dg
from sglang.srt.layers.muse_spark import muse_spark_sp_experts as spe
from sglang.srt.layers.muse_spark.fused import glue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class _Group:  # hashable, as GroupCoordinator
    def __init__(self, name):
        self.world_size, self.rank_in_group, self.cpu_group = 8, 0, object()
        self.device_group = SimpleNamespace(group_name=name)


class TestMuseSparkSpOneHost(CustomTestCase):
    def setUp(self):
        dg.tp_group_on_one_host.cache_clear()
        spe._state.clear()

    def _prepare_guests(self, group, nnodes, same_node):
        layers = [
            SimpleNamespace(
                use_fused_w4a16=True,
                global_num_experts=256,
                num_local_experts=32,
                dynamic_guest_setup=None,
            )
        ]
        parallel = SimpleNamespace(tp_group=group, nnodes=nnodes)
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.distributed.parallel_state.in_the_same_node_as",
                return_value=same_node,
            ) as probe,
            patch.object(
                torch._C, "_accelerator_getAllocatorSettings", return_value=""
            ),
        ):
            dg.maybe_prepare_dynamic_guests(layers)
        return layers[0].dynamic_guest_setup, probe.call_count

    def test_tp8_over_two_hosts_keeps_dynamic_guests_off(self):
        """TP8 on two 4-GPU hosts: a peer's CUDA IPC handle would not open."""
        setup, probes = self._prepare_guests(
            _Group("tp:0"), 2, [True] * 4 + [False] * 4
        )
        self.assertIsNone(setup)
        self.assertEqual(probes, 1)

    def test_one_host_enables_dynamic_guests(self):
        setup, probes = self._prepare_guests(_Group("tp:1"), 1, None)
        self.assertIsInstance(setup, dg._Setup)
        self.assertEqual(probes, 0)  # one node: no probe collective
        setup, _ = self._prepare_guests(_Group("tp:2"), 2, [True] * 8)
        self.assertIsInstance(setup, dg._Setup)

    def test_sp_buffers_fall_back_to_nccl_across_hosts(self):
        """Symmetric memory rendezvous needs every rank on one host."""
        parallel = SimpleNamespace(nnodes=2)
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.distributed.parallel_state.in_the_same_node_as",
                return_value=[True] * 4 + [False] * 4,
            ),
            patch.object(spe, "SPExpertBuffers") as buffers,
        ):
            self.assertIsNone(spe.prepare(_Group("tp:3"), torch.device("cpu"), 4096))
        buffers.assert_not_called()


class TestMuseSparkBf16ExpertLoads(CustomTestCase):
    def test_loads_after_interleave_keep_the_interleaved_layout(self):
        """Once the Hopper Triton path interleaves gate / up rows, a later checkpoint
        load (weight update) must not write the old [gate; up] halves."""
        mod = bf16.MuseSpark1xBf16Experts(
            num_experts=4,
            num_local_experts=2,
            local_expert_offset=2,
            expert_hidden_size=8,
            intermediate_size=4,
            top_k=2,
        )
        mod.gate_up_proj.data.copy_(torch.randn(2, 8, 8))
        with (
            patch.object(bf16, "_trtllm_bf16_moe_supported", return_value=False),
            patch.object(bf16, "_triton_bf16_moe_supported", return_value=True),
            patch.object(glue, "hopper_kernels_enabled", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            mod.prepare_trtllm_weights(None)
        gate, up = torch.randn(4, 8), torch.randn(4, 8)
        param = mod.gate_up_proj
        mod.weight_loader(param, gate, "gate_up_proj", shard_id="w1", expert_id=3)
        mod.weight_loader(param, up, "gate_up_proj", shard_id="w3", expert_id=3)
        want = torch.stack([gate, up], 1).flatten(0, 1)
        self.assertTrue(torch.equal(param.data[1], want))
        stack = torch.randn(2, 8, 8)
        got = bf16.expert_stack_in_layout(param, stack)
        self.assertTrue(
            torch.equal(got[0], stack[0].view(2, 4, 8).transpose(0, 1).flatten(0, 1))
        )
        # a direct copy skips both loaders
        self.assertIsNotNone(mod._direct_weight_update_error)


if __name__ == "__main__":
    unittest.main()
