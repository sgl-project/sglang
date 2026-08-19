# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.expert_pack import KimiGGMLExpertPackStore
from sglang.srt.layers.quantization.expert_pack import (
    ExpertPackMoEMethod,
    _clamped_swiglu,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestExpertPackQuantization(unittest.TestCase):
    def test_deepseek_v4_clamped_swiglu(self) -> None:
        gate = torch.tensor([[-20.0, 20.0, 2.0]])
        up = torch.tensor([[-20.0, 20.0, 3.0]])
        actual = _clamped_swiglu(gate, up, 10.0)
        expected = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        torch.testing.assert_close(actual, expected)

    def test_deepseek_v4_clamped_swiglu_uses_current_kernel_namespace(self) -> None:
        gate = SimpleNamespace(is_cuda=True)
        up = object()
        gate_up = object()
        output = object()

        with (
            patch.object(torch, "cat", return_value=gate_up),
            patch.object(torch, "empty_like", return_value=output),
            patch("sglang.kernels.ops.attention.dsv4.silu_and_mul_clamp") as kernel,
        ):
            actual = _clamped_swiglu(gate, up, 10.0)

        self.assertIs(actual, output)
        kernel.assert_called_once_with(gate_up, output, 10.0)

    def test_runner_leaves_routed_scale_to_model_and_captures_limit(self) -> None:
        method = ExpertPackMoEMethod(None, "model.layers.0.mlp.experts")
        config = SimpleNamespace(
            activation="silu", swiglu_limit=10.0, routed_scaling_factor=1.5
        )
        method.create_moe_runner(None, config)
        self.assertEqual(method.swiglu_limit, 10.0)
        self.assertFalse(hasattr(method, "routed_scaling_factor"))

    def test_kimi_runner_requires_exact_situ_constants(self) -> None:
        store = object.__new__(KimiGGMLExpertPackStore)
        method = ExpertPackMoEMethod(store, "model.layers.1.mlp.experts")
        valid = SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            swiglu_limit=None,
        )
        method.create_moe_runner(None, valid)
        self.assertEqual(method.situ_beta, 4.0)
        self.assertEqual(method.situ_linear_beta, 25.0)

        for invalid in (
            SimpleNamespace(
                activation="silu",
                gemm1_alpha=4.0,
                gemm1_clamp_limit=25.0,
                swiglu_limit=None,
            ),
            SimpleNamespace(
                activation="situ",
                gemm1_alpha=4.0,
                gemm1_clamp_limit=24.0,
                swiglu_limit=None,
            ),
        ):
            with self.assertRaises(ValueError):
                ExpertPackMoEMethod(
                    store, "model.layers.1.mlp.experts"
                ).create_moe_runner(None, invalid)


if __name__ == "__main__":
    unittest.main()
