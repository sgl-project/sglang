"""Muse Spark Hopper setup: BF16 expert loads after the gate / up interleave."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.muse_spark import muse_spark_bf16_moe as bf16
from sglang.srt.layers.muse_spark.fused import glue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


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
