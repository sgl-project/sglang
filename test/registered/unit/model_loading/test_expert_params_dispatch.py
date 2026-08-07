import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch import nn

sys.modules["sgl_kernel"] = MagicMock()
sys.modules["sgl_kernel.quantization"] = MagicMock()
sys.modules["sgl_kernel.scalar_type"] = MagicMock()

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE  # noqa: E402
from sglang.srt.model_loader.auto_loader import (  # noqa: E402
    MOE_EXPERT_STACKED_SKIP_SUBSTRS,
    STANDARD_GATE_UP_MAPPING,
    ExpertParamsDispatch,
    load_moe_sparse_block_weights,
    try_load_stacked_skip_moe_experts,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

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

    def test_missing_mapped_expert_target_is_rejected(self):
        dispatch = ExpertParamsDispatch.from_gate_up_down(num_experts=1)
        with self.assertRaisesRegex(ValueError, "experts.w13_weight"):
            dispatch.try_load(
                "experts.0.gate_proj.weight",
                torch.zeros(2, 2),
                {},
            )

    def test_first_existing_expert_target_wins(self):
        dispatch = ExpertParamsDispatch.from_fused_moe_mapping(
            [
                ("experts.missing_", "experts.0.gate_proj.", 0, "missing"),
                ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
                ("experts.w13_", "experts.0.gate_proj.", 0, "duplicate"),
            ]
        )
        loader = _RecordingLoader()
        param = mock.Mock()
        param.weight_loader = loader

        target = dispatch.try_load(
            "experts.0.gate_proj.weight",
            torch.zeros(2, 2),
            {"experts.w13_weight": param},
        )

        self.assertEqual(target, "experts.w13_weight")
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(loader.calls[0][1], "w1")

    def test_real_loader_type_error_propagates(self):
        dispatch = ExpertParamsDispatch.from_gate_up_down(num_experts=1)
        param = mock.Mock()
        param.weight_loader = mock.Mock(
            side_effect=TypeError("loader implementation failed")
        )
        with self.assertRaisesRegex(TypeError, "loader implementation failed"):
            dispatch.try_load(
                "experts.0.gate_proj.weight",
                torch.zeros(2, 2),
                {"experts.w13_weight": param},
            )
        param.weight_loader.assert_called_once()

    def test_moe_helper_rejects_unexpected_weight(self):
        with self.assertRaisesRegex(ValueError, "unexpected.weight"):
            load_moe_sparse_block_weights(
                nn.Module(),
                [("unexpected.weight", torch.ones(1))],
                expert_dispatch=ExpertParamsDispatch(),
            )

    def test_moe_helper_has_named_expert_bias_skip(self):
        dispatch = ExpertParamsDispatch.from_gate_up_down(num_experts=1)
        loaded = load_moe_sparse_block_weights(
            nn.Module(),
            [("experts.0.gate_proj.bias", torch.ones(1))],
            expert_dispatch=dispatch,
        )
        self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()
