"""Correctness tests for the PyTorch diffusion fallbacks used on MPS."""

import unittest

import torch

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    norm_infer_native,
    rms_norm_fn_native,
    triton_one_pass_rms_norm_native,
)
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestDiffusionTorchFallback(unittest.TestCase):
    @property
    def device(self):
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def test_norm_infer_matches_reference(self):
        dtypes = (
            (torch.float32, torch.float16, torch.bfloat16)
            if self.device.type == "mps"
            else (torch.float32,)
        )
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = torch.randn(4, 32, device=self.device, dtype=dtype)
                weight = torch.randn(32, device=self.device, dtype=dtype)
                bias = torch.randn(32, device=self.device, dtype=dtype)

                rms = norm_infer_native(x, weight, None, 1e-5, is_rms_norm=True)
                x_fp32 = x.float()
                rms_ref = (
                    x_fp32
                    * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5)
                    * weight.float()
                ).to(dtype)

                layer = norm_infer_native(x, weight, bias, 1e-5)
                layer_ref = torch.nn.functional.layer_norm(x, (32,), weight, bias, 1e-5)

                tolerance = 2e-2 if dtype != torch.float32 else 2e-5
                torch.testing.assert_close(
                    rms.cpu(), rms_ref.cpu(), rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(layer.cpu(), layer_ref.cpu())

                out = torch.empty_like(x)
                returned = norm_infer_native(x, weight, bias, 1e-5, out=out)
                self.assertIs(returned, out)
                torch.testing.assert_close(out.cpu(), layer_ref.cpu())

    def test_one_pass_rms_norm_matches_reference(self):
        x = torch.randn(8, 128, device=self.device, dtype=torch.float32)
        weight = torch.randn(128, device=self.device, dtype=torch.float32)
        result = triton_one_pass_rms_norm_native(x, weight, 1e-6)
        reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
        torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=2e-5, atol=2e-5)

    def test_rms_norm_fn_preserves_residual_contract(self):
        x = torch.randn(4, 32, device=self.device, dtype=torch.float32)
        residual = torch.randn_like(x)
        weight = torch.randn(32, device=self.device)
        bias = torch.randn(32, device=self.device)

        result, residual_out = rms_norm_fn_native(
            x,
            weight,
            bias,
            residual=residual,
            residual_in_fp32=True,
            zero_centered_weight=True,
        )

        combined = x.float() + residual.float()
        reference = combined * torch.rsqrt(
            combined.square().mean(-1, keepdim=True) + 1e-6
        )
        reference = reference * (weight.float() + 1.0) + bias.float()
        torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(residual_out.cpu(), combined.cpu())


if __name__ == "__main__":
    unittest.main()
