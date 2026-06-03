"""Unit tests for DeepEP Waterfill and EPLB updater compatibility."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.utils import get_moe_weight_param_eplb_view
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.test_utils import CustomTestCase


class _FakeExpertParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = 5
        self._num_local_routed = 4
        self.weight = nn.Parameter(
            torch.arange(10, dtype=torch.float32).reshape(self.num_local_experts, 2)
        )
        self.correction_bias = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale._sglang_require_global_experts = True


class TestDeepEPWaterfillEPLB(CustomTestCase):
    def test_deepep_waterfill_eplb_view_exposes_only_routed_weights(self):
        experts = _FakeExpertParam()
        shared_before = experts.weight.data[-1].clone()

        view = get_moe_weight_param_eplb_view(
            "weight",
            experts.weight,
            experts.num_local_experts,
            experts._num_local_routed,
        )

        self.assertEqual(view.shape, (experts._num_local_routed, 2))
        view[0].copy_(view[1])
        self.assertTrue(torch.equal(experts.weight.data[0], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(experts.weight.data[-1], shared_before))

    def test_default_eplb_view_keeps_existing_weight_shape(self):
        experts = _FakeExpertParam()

        view = get_moe_weight_param_eplb_view(
            "weight",
            experts.weight,
            experts.num_local_experts,
        )

        self.assertEqual(view.shape, (experts.num_local_experts, 2))

    def test_eplb_view_skips_global_expert_metadata_param(self):
        experts = _FakeExpertParam()

        view = get_moe_weight_param_eplb_view(
            "global_scale",
            experts.global_scale,
            experts.num_local_experts,
            experts._num_local_routed,
        )

        self.assertIsNone(view)

    def test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=1, experts=experts)
        shared_before = experts.weight.data[-1].clone()

        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].shape, (experts._num_local_routed, 2))

        weights[0][-1].zero_()
        self.assertTrue(torch.equal(experts.weight.data[-2], torch.zeros(2)))
        self.assertTrue(torch.equal(experts.weight.data[-1], shared_before))


if __name__ == "__main__":
    unittest.main()
