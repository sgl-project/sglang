"""Unit tests for norm/scale-shift fallback math."""

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")
register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd")

import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.test.test_utils import CustomTestCase


class TestNormScaleShiftFallbacks(CustomTestCase):
    def test_layernorm_scaleshift_fallback_matches_reference(self):
        batch_size, seq_len, hidden_size = 2, 17, 320
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        shift = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32) * 0.05
        scale = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32) * 0.05

        mod = LayerNormScaleShift(hidden_size, elementwise_affine=False, eps=1e-6)
        out = mod(x, shift, scale)

        ref = F.layer_norm(x, (hidden_size,), None, None, 1e-6) * (1 + scale) + shift
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_scale_residual_layernorm_scaleshift_fallback_matches_reference(self):
        batch_size, seq_len, hidden_size = 2, 19, 320
        residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32) * 0.02
        gate = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32) * 0.05
        shift = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32) * 0.05
        scale = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32) * 0.05

        mod = ScaleResidualLayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        out_norm, out_residual = mod(residual, x, gate, shift, scale)

        ref_residual = residual + x * gate
        ref_norm = (
            F.layer_norm(ref_residual, (hidden_size,), None, None, 1e-6) * (1 + scale)
            + shift
        )

        torch.testing.assert_close(out_residual, ref_residual, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_norm, ref_norm, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(out_norm).all().item())

    def test_muladd_fallback_matches_reference_for_3d_and_4d_modulation(self):
        op = MulAdd()

        # 3D modulation case
        a = torch.randn(2, 23, 64, dtype=torch.float32)
        b = torch.randn(2, 1, 64, dtype=torch.float32) * 0.05
        c = torch.randn(2, 23, 64, dtype=torch.float32)
        out = op(a, b, c)
        ref = c + a * b
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

        # 4D frame-wise modulation case
        a = torch.randn(2, 24, 64, dtype=torch.float32)
        b = torch.randn(2, 6, 1, 64, dtype=torch.float32) * 0.05
        c = torch.randn(2, 24, 64, dtype=torch.float32)
        out = op(a, b, c)
        ref = c + (a.unflatten(dim=1, sizes=(6, 4)) * b).flatten(1, 2)
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
