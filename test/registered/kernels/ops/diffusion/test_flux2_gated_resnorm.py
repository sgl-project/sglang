"""Blackwell coverage for the FLUX.2 gated residual normalization fast path."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
from sglang.kernels.ops.diffusion import residual_gate_add
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@unittest.skipUnless(_is_blackwell(), "FLUX.2 gated residual fusion requires SM100+")
class TestFlux2GatedResnorm(CustomTestCase):
    def test_gated_residual_norm_modulate_is_bit_exact(self):
        torch.manual_seed(0)
        hidden = 6144
        shape = (1, 64, hidden)
        residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        params = torch.randn(1, 1, 3 * hidden, device="cuda").bfloat16()
        gate, scale, shift = params.chunk(3, dim=-1)
        norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6, device="cuda")

        expected_residual = residual_gate_add(residual, update, gate)
        expected = norm(expected_residual) * (1 + scale) + shift
        actual, actual_residual = flux2._flux2_gated_resnorm(
            norm, residual, update, gate, scale, shift
        )

        self.assertIsInstance(
            flux2._defer_gated_residual(residual, update, gate), tuple
        )
        self.assertTrue(torch.equal(actual_residual, expected_residual))
        self.assertTrue(torch.equal(actual, expected))

    def test_gated_residual_defer_rejects_unsupported_inputs(self):
        torch.manual_seed(0)
        for batch, dtype in ((1, torch.float16), (2, torch.bfloat16)):
            residual = torch.randn(batch, 17, 6144, device="cuda", dtype=dtype)
            update = torch.randn_like(residual)
            gate = torch.randn(batch, 1, 6144, device="cuda", dtype=dtype)
            expected = residual_gate_add(residual, update, gate)
            actual = flux2._defer_gated_residual(residual, update, gate)

            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(torch.equal(actual, expected))

        residual = torch.randn(1, 17, 6144, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        gate = torch.randn(1, 1, 6144, device="cuda", dtype=torch.bfloat16)
        with patch("torch.compiler.is_compiling", return_value=True):
            actual = flux2._defer_gated_residual(residual, update, gate)

        self.assertIsInstance(actual, torch.Tensor)
        self.assertTrue(torch.equal(actual, residual_gate_add(residual, update, gate)))


if __name__ == "__main__":
    unittest.main()
