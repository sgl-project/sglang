import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.model_loader.auto_loader import (
    MOE_EXPERT_STACKED_SKIP_SUBSTRS,
    STANDARD_GATE_UP_MAPPING,
    ExpertParamsDispatch,
    try_load_stacked_skip_moe_experts,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-b-test-cpu")


class _RecordingLoader:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, param, tensor, name, *, shard_id, expert_id):
        self.calls.append((name, shard_id, expert_id, tuple(tensor.shape)))


class TestExpertParamsDispatch(unittest.TestCase):
    def test_expert_routing_gate_proj(self):
        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=2,
        )
        dispatch = ExpertParamsDispatch.from_fused_moe_mapping(mapping)
        loader = _RecordingLoader()
        param = mock.Mock()
        param.weight_loader = loader
        params = {"model.layers.0.mlp.experts.w13_weight": param}
        tensor = torch.zeros(4, 4)
        ckpt_name = "model.layers.0.mlp.experts.1.gate_proj.weight"
        target = dispatch.try_load(ckpt_name, tensor, params)
        self.assertEqual(target, "model.layers.0.mlp.experts.w13_weight")
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(loader.calls[0][1], "w1")
        self.assertEqual(loader.calls[0][2], 1)

    def test_from_gate_up_down_mixtral_names(self):
        dispatch = ExpertParamsDispatch.from_gate_up_down(
            num_experts=1,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
        )
        self.assertTrue(
            any(
                "experts.0.w1." in weight_name
                for _, weight_name, _, _ in dispatch.mappings
            )
        )

    def test_stacked_skip_moe_experts_before_rename(self):
        gate_up = mock.Mock()
        gate_up.weight_loader = mock.Mock()
        params = {"mlp.experts.0.gate_up_proj.weight": gate_up}
        tensor = torch.zeros(2, 2)
        name = "mlp.experts.0.gate_proj.weight"
        result = try_load_stacked_skip_moe_experts(
            STANDARD_GATE_UP_MAPPING, name, tensor, params
        )
        self.assertIsNone(result)
        gate_up.weight_loader.assert_not_called()

    def test_stacked_applies_to_shared_expert_path(self):
        gate_up = mock.Mock()
        gate_up.weight_loader = mock.Mock()
        params = {"mlp.shared_expert.gate_up_proj.weight": gate_up}
        tensor = torch.zeros(2, 2)
        name = "mlp.shared_expert.gate_proj.weight"
        result = try_load_stacked_skip_moe_experts(
            STANDARD_GATE_UP_MAPPING, name, tensor, params
        )
        self.assertEqual(result, "mlp.shared_expert.gate_up_proj.weight")
        gate_up.weight_loader.assert_called_once()

    def test_moe_expert_skip_substrs_cover_experts_prefix(self):
        self.assertIn("mlp.experts", MOE_EXPERT_STACKED_SKIP_SUBSTRS)
        self.assertIn("experts.", MOE_EXPERT_STACKED_SKIP_SUBSTRS)


if __name__ == "__main__":
    unittest.main()
