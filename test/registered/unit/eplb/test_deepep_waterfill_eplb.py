"""Unit tests for DeepEP Waterfill and EPLB updater compatibility."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.test_utils import CustomTestCase


class _FakeExpertParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = 5
        self.weight = nn.Parameter(
            torch.arange(10, dtype=torch.float32).reshape(self.num_local_experts, 2)
        )
        self.correction_bias = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale._sglang_require_global_experts = True


class TestDeepEPWaterfillEPLB(CustomTestCase):
    def test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=1, experts=experts)
        shared_before = experts.weight.data[-1].clone()

        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(
            weights[0].shape,
            (experts.num_local_experts - moe.num_fused_shared_experts, 2),
        )

        weights[0][-1].zero_()
        self.assertTrue(torch.equal(experts.weight.data[-2], torch.zeros(2)))
        self.assertTrue(torch.equal(experts.weight.data[-1], shared_before))

    def test_deepseek_moe_get_moe_weights_keeps_full_shape_without_fusion(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=0, experts=experts)
        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].shape, (experts.num_local_experts, 2))


if __name__ == "__main__":
    unittest.main()
