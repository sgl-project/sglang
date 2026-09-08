# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.topk import biased_grouped_topk_impl  # noqa: E402
from sglang.srt.models.llada2 import LLaDA2MoeGate  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLLaDA2Router(CustomTestCase):
    def test_gate_preserves_router_dtype(self):
        config = SimpleNamespace(
            num_experts=4, hidden_size=8, moe_router_enable_expert_bias=True
        )
        for router_dtype in (torch.float32, torch.bfloat16):
            for hidden_dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(router_dtype=router_dtype, hidden_dtype=hidden_dtype):
                    gate = LLaDA2MoeGate(config, params_dtype=router_dtype)
                    with torch.no_grad():
                        gate.weight.copy_(torch.arange(32).reshape(4, 8) / 37)
                    hidden = (torch.arange(48).reshape(3, 16) / 19).to(hidden_dtype)[
                        :, ::2
                    ]
                    expected = F.linear(hidden.to(router_dtype), gate.weight)
                    logits = gate(hidden)
                    self.assertEqual(logits.dtype, router_dtype)
                    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_fp32_gate_keeps_near_tie_expert_selection(self):
        config = SimpleNamespace(
            num_experts=256, hidden_size=4, moe_router_enable_expert_bias=True
        )
        gate = LLaDA2MoeGate(config, params_dtype=torch.float32)
        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[:, 0].copy_(torch.linspace(0.999, 1.001, 256))
            gate.expert_bias.zero_()
        hidden = torch.ones((3, 4), dtype=torch.bfloat16)
        logits = gate(hidden)
        weights, indices = biased_grouped_topk_impl(
            hidden,
            logits.float(),
            gate.expert_bias,
            topk=8,
            renormalize=True,
            num_expert_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        expected_indices = torch.arange(248, 256, dtype=torch.int32).expand(3, -1)
        torch.testing.assert_close(
            indices.sort(dim=-1).values, expected_indices, rtol=0, atol=0
        )
        expected_scores = F.linear(hidden.float(), gate.weight).sigmoid()
        selected = expected_scores.gather(1, indices.long())
        expected_weights = selected / (selected.sum(-1, keepdim=True) + 1e-20) * 2.5
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
