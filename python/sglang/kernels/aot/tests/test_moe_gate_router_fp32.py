import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.models.deepseek_v2 import MoEGate


class TestMoEGateCPUFP32Router(unittest.TestCase):
    def _make_gate(self, weight_dtype: torch.dtype) -> MoEGate:
        config = types.SimpleNamespace(
            n_routed_experts=8,
            hidden_size=32,
            topk_method="greedy",
        )
        gate = MoEGate(config, quant_config=None)
        gate.weight = torch.nn.Parameter(
            torch.randn(
                (config.n_routed_experts, config.hidden_size), dtype=weight_dtype
            )
        )
        return gate

    def test_cpu_amx_router_path_returns_fp32_logits_for_router_weights(self):
        hidden_states = torch.randn((5, 32), dtype=torch.bfloat16)

        for weight_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(weight_dtype=weight_dtype):
                gate = self._make_gate(weight_dtype)
                expected = F.linear(hidden_states.float(), gate.weight.float(), None)

                with patch("sglang.srt.models.deepseek_v2._is_cpu", True), patch(
                    "sglang.srt.models.deepseek_v2._is_cpu_amx_available", True
                ), patch(
                    "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
                    side_effect=AssertionError(
                        "router correctness path should bypass packed AMX linear"
                    ),
                ):
                    logits = gate(hidden_states)

                self.assertEqual(logits.dtype, torch.float32)
                torch.testing.assert_close(logits, expected)


if __name__ == "__main__":
    unittest.main()
