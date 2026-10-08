# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.llada2 import (  # noqa: E402
    LLaDA2MoeGate,
    _prepare_llada2_language_weights,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLLaDA2LanguageModel(CustomTestCase):
    def test_fp32_router_logits_are_not_rounded_to_the_activation_dtype(self):
        """An fp32 router must score bf16 activations in fp32, as the reference does."""
        config = SimpleNamespace(
            num_experts=4, hidden_size=8, moe_router_enable_expert_bias=True
        )
        gate = LLaDA2MoeGate(config, params_dtype=torch.float32)
        with torch.no_grad():
            gate.weight.copy_(torch.arange(32).reshape(4, 8) / 37)
        hidden = (torch.arange(24).reshape(3, 8) / 19).to(torch.bfloat16)

        logits = gate(hidden)

        self.assertEqual(logits.dtype, torch.float32)
        expected = F.linear(hidden.float(), gate.weight)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_language_checkpoint_layout_is_normalized(self):
        """Prefixed names and fused expert tensors load as the native layout."""
        fused = torch.arange(24).reshape(2, 3, 4)
        lm_head = torch.randn(4, 3)

        expanded = list(
            _prepare_llada2_language_weights(
                [
                    ("model.language_model.layers.1.mlp.experts.gate_proj", fused),
                    ("model.lm_head.weight", lm_head),
                ],
                num_experts=2,
            )
        )

        self.assertEqual(
            [name for name, _ in expanded],
            [
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.1.gate_proj.weight",
                "lm_head.weight",
            ],
        )
        torch.testing.assert_close(expanded[0][1], fused[0])
        torch.testing.assert_close(expanded[1][1], fused[1])
        torch.testing.assert_close(expanded[2][1], lm_head)
        with self.assertRaisesRegex(ValueError, "expected first dimension 2"):
            list(
                _prepare_llada2_language_weights(
                    [("model.layers.1.mlp.experts.down_proj", torch.empty(3, 4, 5))],
                    num_experts=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
