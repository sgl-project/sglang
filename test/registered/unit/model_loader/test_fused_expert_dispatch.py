"""CPU unit tests for FusedExpertDispatch expert fan-out."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch
from torch.nn import Parameter

from sglang.srt.model_loader.auto_loader import FusedExpertDispatch


class TestFusedExpertDispatch(unittest.TestCase):
    def test_fan_out_gate_up_splits_w1_w3(self):
        num_experts = 3
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append((shard_id, expert_id, shard_tensor.shape))

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader

        gate = torch.randn(num_experts, 4, 4)
        up = torch.randn(num_experts, 4, 4)
        fused = torch.cat([gate, up], dim=-2)

        params = {
            "model.layers.0.mlp.experts.w13_weight.weight": param,
        }
        dispatch = FusedExpertDispatch(num_experts=num_experts)
        ckpt_name = "model.layers.0.mlp.experts.gate_up_proj.weight"

        loaded = dispatch.try_load(ckpt_name, fused, params)
        self.assertEqual(loaded, "model.layers.0.mlp.experts.w13_weight.weight")
        self.assertEqual(len(calls), num_experts * 2)

    def test_fan_out_down_proj(self):
        num_experts = 2
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append((shard_id, expert_id))

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader

        fused = torch.randn(num_experts, 6, 3)
        params = {"layer.mlp.experts.w2_weight.weight": param}
        dispatch = FusedExpertDispatch(num_experts=num_experts)

        loaded = dispatch.try_load("layer.mlp.experts.down_proj.weight", fused, params)
        self.assertEqual(loaded, "layer.mlp.experts.w2_weight.weight")
        self.assertEqual(calls, [("w2", 0), ("w2", 1)])

    def test_static_fan_out_helper(self):
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append(expert_id)

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader
        tensor = torch.randn(4, 2, 2)

        FusedExpertDispatch.fan_out_to_experts(
            param, tensor, "experts.w13_weight", "w1", 4
        )
        self.assertEqual(calls, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
