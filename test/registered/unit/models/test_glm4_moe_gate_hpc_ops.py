"""Unit tests for GLM MoE gate GEMM dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.glm4_moe import Glm4MoeGate  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _glm_config(hidden_size, n_routed_experts):
    return SimpleNamespace(
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
    )


class TestGlm4MoeGateHpcOps(CustomTestCase):
    def _assert_uses_fp32_gate(
        self, gate, hidden_states, *, is_cuda=True, is_hip=False
    ):
        expected = torch.randn(
            (hidden_states.shape[0], gate.weight.shape[0]), dtype=torch.float32
        )

        with patch(
            "sglang.srt.models.glm4_moe.linear_bf16_fp32",
            side_effect=AssertionError("unexpected hpc_ops dispatch"),
        ), patch("sglang.srt.models.glm4_moe._is_cuda", is_cuda), patch(
            "sglang.srt.models.glm4_moe._is_hip", is_hip
        ), patch(
            "sglang.srt.models.glm4_moe.F.linear", return_value=expected
        ) as mock_linear:
            out = gate(hidden_states)

        self.assertIs(out, expected)
        mock_linear.assert_called_once()
        self.assertEqual(mock_linear.call_args.args[0].dtype, torch.float32)
        self.assertIs(mock_linear.call_args.args[1], gate._weight_fp32)

    def test_glm5_shape_dispatches_large_prefill_to_hpc_ops(self):
        gate = Glm4MoeGate(_glm_config(hidden_size=6144, n_routed_experts=256))
        hidden_states = torch.randn((2048, 6144), dtype=torch.bfloat16)
        expected = torch.randn((2048, 256), dtype=torch.float32)

        with patch(
            "sglang.srt.models.glm4_moe.linear_bf16_fp32",
            return_value=expected,
        ) as mock_linear, patch("sglang.srt.models.glm4_moe._is_cuda", True):
            out = gate(hidden_states)

        self.assertIs(out, expected)
        mock_linear.assert_called_once()
        args, kwargs = mock_linear.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], gate._weight_fp32)
        self.assertEqual(kwargs["hpc_ops_min_m"], 2048)

    def test_glm5_shape_uses_fp32_gate_for_small_batches(self):
        gate = Glm4MoeGate(_glm_config(hidden_size=6144, n_routed_experts=256))
        hidden_states = torch.randn((1024, 6144), dtype=torch.bfloat16)

        self._assert_uses_fp32_gate(gate, hidden_states)

    def test_glm5_shape_uses_fp32_gate_on_non_cuda(self):
        gate = Glm4MoeGate(_glm_config(hidden_size=6144, n_routed_experts=256))
        hidden_states = torch.randn((2048, 6144), dtype=torch.bfloat16)

        self._assert_uses_fp32_gate(gate, hidden_states, is_cuda=False)

    def test_glm5_shape_uses_fp32_gate_on_hip(self):
        gate = Glm4MoeGate(_glm_config(hidden_size=6144, n_routed_experts=256))
        hidden_states = torch.randn((2048, 6144), dtype=torch.bfloat16)

        self._assert_uses_fp32_gate(gate, hidden_states, is_hip=True)

    def test_unbenchmarked_shape_uses_fp32_gate(self):
        gate = Glm4MoeGate(_glm_config(hidden_size=4096, n_routed_experts=256))
        hidden_states = torch.randn((2048, 4096), dtype=torch.bfloat16)

        self._assert_uses_fp32_gate(gate, hidden_states)


if __name__ == "__main__":
    unittest.main()
