# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small-amd")

HIDDEN = 1024
SEQ = 64
EPS = 1e-6


def _reference(x, weight, bias, scale, shift, residual=None, gate=None):
    """FP32 LayerNorm + modulation, the contract the ROCm kernel documents."""
    h = x.float()
    residual_out = None
    if residual is not None:
        h = residual.float() + h * gate.float()
        residual_out = h.to(x.dtype)
    mean = h.mean(-1, keepdim=True)
    var = ((h - mean) ** 2).mean(-1, keepdim=True)
    normed = (h - mean) * torch.rsqrt(var + EPS) * weight.float() + bias.float()
    out = normed * (1.0 + scale.float()) + shift.float()
    return out.to(x.dtype), residual_out


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "needs a ROCm GPU"
)
class TestRocmNormScaleShiftTriton(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.diffusion.norm.rocm_norm_scale_shift_triton import (
            rocm_norm_scale_shift,
        )

        self.kernel = rocm_norm_scale_shift
        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        self.x = torch.randn(1, SEQ, HIDDEN, device=dev, dtype=dt)
        self.residual = torch.randn(1, SEQ, HIDDEN, device=dev, dtype=dt)
        self.weight = torch.randn(HIDDEN, device=dev, dtype=dt)
        self.bias = torch.randn(HIDDEN, device=dev, dtype=dt)
        self.scale = torch.randn(1, HIDDEN, device=dev, dtype=dt)
        self.shift = torch.randn(1, HIDDEN, device=dev, dtype=dt)
        self.gate = torch.randn(1, HIDDEN, device=dev, dtype=dt)

    def test_norm_scale_shift_matches_fp32_reference(self):
        out, residual_out = self.kernel(
            self.x, self.weight, self.bias, self.scale, self.shift, EPS
        )
        ref, _ = _reference(self.x, self.weight, self.bias, self.scale, self.shift)
        self.assertIsNone(residual_out)
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    def test_gated_residual_norm_matches_fp32_reference(self):
        out, residual_out = self.kernel(
            self.x,
            self.weight,
            self.bias,
            self.scale,
            self.shift,
            EPS,
            residual=self.residual,
            gate=self.gate,
        )
        ref, ref_res = _reference(
            self.x,
            self.weight,
            self.bias,
            self.scale,
            self.shift,
            residual=self.residual,
            gate=self.gate,
        )
        torch.testing.assert_close(residual_out, ref_res, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    def test_unsupported_dtype_falls_back_to_none(self):
        out = self.kernel(
            self.x.float(), self.weight, self.bias, self.scale, self.shift, EPS
        )
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
