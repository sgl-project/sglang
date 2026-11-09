"""Unit tests for MiMo-V2 router GEMM dispatch to the HPC-Ops bf16xfp32 kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.models.mimo_v2 import MoEGate  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _mimo_config(hidden_size, n_routed_experts, **kwargs):
    return SimpleNamespace(
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        topk_method="noaux_tc",
        **kwargs,
    )


class TestMiMoV2RouterHpcGemm(CustomTestCase):
    def _make_gate(self, config, algo):
        with envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.override(algo):
            return MoEGate(config, quant_config=None)

    def test_benchmarked_shape_dispatches_with_min_m(self):
        gate = self._make_gate(_mimo_config(4096, 256), "hpc")
        hidden_states = torch.randn((4, 4096), dtype=torch.bfloat16)
        expected = torch.randn((4, 256), dtype=torch.float32)

        with patch(
            "sglang.srt.models.mimo_v2.linear_bf16_fp32", return_value=expected
        ) as mock_linear:
            out = gate(hidden_states)

        self.assertIs(out, expected)
        mock_linear.assert_called_once()
        args, kwargs = mock_linear.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], gate.weight)
        self.assertEqual(kwargs["hpc_kernel_min_m"], 8)

    def _assert_uses_default_path(self, gate, hidden_size, n_routed_experts):
        with torch.no_grad():
            gate.weight.copy_(torch.randn_like(gate.weight))
        hidden_states = torch.randn((4, hidden_size), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.mimo_v2.linear_bf16_fp32",
            side_effect=AssertionError("unexpected hpc kernel dispatch"),
        ):
            out = gate(hidden_states)

        self.assertEqual(out.shape, (4, n_routed_experts))
        self.assertEqual(out.dtype, torch.float32)

    def test_default_algo_keeps_original_path(self):
        gate = self._make_gate(_mimo_config(4096, 256), "cublas")
        self.assertIsNone(gate.hpc_kernel_min_m)
        self._assert_uses_default_path(gate, 4096, 256)

    def test_unbenchmarked_shape_keeps_original_path(self):
        gate = self._make_gate(_mimo_config(2048, 128), "hpc")
        self.assertIsNone(gate.hpc_kernel_min_m)
        self._assert_uses_default_path(gate, 2048, 128)

    def test_bf16_router_keeps_original_path(self):
        gate = self._make_gate(
            _mimo_config(4096, 256, moe_router_dtype="bfloat16"), "hpc"
        )
        self.assertIsNone(gate.hpc_kernel_min_m)
        self._assert_uses_default_path(gate, 4096, 256)


if __name__ == "__main__":
    unittest.main()
