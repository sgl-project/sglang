import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmRunnerCore,
    _apply_gemm1_alpha_activation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepGemmRunnerActivation(CustomTestCase):
    @patch(
        "sglang.srt.layers.moe.moe_runner.deep_gemm.envs."
        "SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get",
        return_value=True,
    )
    def test_gemm1_alpha_disables_incompatible_fused_activation(self, _get_flag):
        runner = DeepGemmRunnerCore(
            MoeRunnerConfig(
                activation="silu",
                is_gated=True,
                gemm1_alpha=1.702,
                gemm1_clamp_limit=7.0,
            )
        )

        self.assertTrue(runner.use_sigmoid_alpha_swiglu)
        self.assertFalse(runner.use_fused_contiguous_activation)

    def test_gemm1_alpha_rejects_standard_swiglu_limit(self):
        config = MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            gemm1_alpha=1.702,
            swiglu_limit=7.0,
        )

        with self.assertRaisesRegex(
            AssertionError, "Only one activation variant can be configured"
        ):
            DeepGemmRunnerCore(config)

    def test_non_interleaved_gemm1_alpha_activation(self):
        gate = torch.tensor([[8.0, -2.0]])
        up = torch.tensor([[9.0, -9.0]])
        gateup = torch.cat([gate, up], dim=-1)

        actual = _apply_gemm1_alpha_activation(
            gateup,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
        clamped_gate = gate.clamp(max=7.0)
        clamped_up = up.clamp(min=-7.0, max=7.0)
        expected = (
            clamped_gate * torch.sigmoid(clamped_gate * 1.702) * (clamped_up + 1.0)
        )

        torch.testing.assert_close(actual, expected)

    def test_interleaved_gemm1_alpha_activation(self):
        gateup = torch.tensor([[8.0, 9.0, -2.0, -9.0]])

        actual = _apply_gemm1_alpha_activation(
            gateup,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=True,
        )
        gate = gateup[..., ::2].clamp(max=7.0)
        up = gateup[..., 1::2].clamp(min=-7.0, max=7.0)
        expected = gate * torch.sigmoid(gate * 1.702) * (up + 1.0)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
