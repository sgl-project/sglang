"""Correctness tests for the PyTorch diffusion fallbacks used on MPS."""

import unittest

import torch

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
    fuse_scale_shift_kernel_native,
    norm_infer_native,
    rms_norm_fn_native,
    triton_one_pass_rms_norm_native,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=1, suite="base-b-test-cpu-arm64")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestDiffusionTorchFallback(unittest.TestCase):
    @property
    def device(self):
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def test_norm_infer_matches_reference(self):
        dtypes = (
            (torch.float32, torch.float16, torch.bfloat16)
            if self.device.type == "mps"
            else (torch.float32, torch.bfloat16)
        )
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = torch.randn(4, 32, device=self.device, dtype=dtype)
                weight = torch.randn(32, device=self.device, dtype=dtype)
                bias = torch.randn(32, device=self.device, dtype=dtype)

                rms = norm_infer_native(x, weight, bias, 1e-5, is_rms_norm=True)
                x_fp32 = x.float()
                rms_ref = (
                    x_fp32
                    * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5)
                    * weight.float()
                    + bias.float()
                ).to(dtype)

                layer = norm_infer_native(x, weight, bias, 1e-5)
                mean = x_fp32.mean(-1, keepdim=True)
                layer_ref = (
                    (x_fp32 - mean)
                    * torch.rsqrt(
                        (x_fp32 - mean).square().mean(-1, keepdim=True) + 1e-5
                    )
                    * weight.float()
                    + bias.float()
                ).to(dtype)

                tolerance = 0 if dtype != torch.float32 else 2e-5
                torch.testing.assert_close(
                    rms.cpu(), rms_ref.cpu(), rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(layer.cpu(), layer_ref.cpu())

                out = torch.empty_like(x)
                returned = norm_infer_native(x, weight, bias, 1e-5, out=out)
                self.assertIs(returned, out)
                torch.testing.assert_close(out.cpu(), layer_ref.cpu())

    def test_norm_infer_preserves_input_dtype_with_fp32_parameters(self):
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(4, 32, device=self.device, dtype=dtype)
                weight = torch.randn(32, device=self.device, dtype=torch.float32)
                bias = torch.randn(32, device=self.device, dtype=torch.float32)

                result = norm_infer_native(x, weight, bias, 1e-5, is_rms_norm=True)
                x_fp32 = x.float()
                reference = (
                    x_fp32
                    * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5)
                    * weight
                    + bias
                ).to(dtype)

                self.assertEqual(result.dtype, dtype)
                torch.testing.assert_close(
                    result.cpu(), reference.cpu(), rtol=0, atol=0
                )

    def test_scale_shift_matches_broadcast_reference(self):
        x = torch.randn(2, 6, 8, device=self.device)
        scale = torch.randn(2, 8, device=self.device)
        shift = torch.randn(2, 8, device=self.device)

        result = fuse_scale_shift_kernel_native(x, scale, shift, scale_constant=0.5)
        reference = x * (0.5 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        torch.testing.assert_close(result.cpu(), reference.cpu())

        frame_scale = torch.randn(2, 3, 1, 8, device=self.device)
        frame_shift = torch.randn(2, 3, 1, 8, device=self.device)
        result = fuse_scale_shift_kernel_native(x, frame_scale, frame_shift)
        expanded_scale = (
            frame_scale.squeeze(2).unsqueeze(2).expand(-1, -1, 2, -1).reshape_as(x)
        )
        expanded_shift = (
            frame_shift.squeeze(2).unsqueeze(2).expand(-1, -1, 2, -1).reshape_as(x)
        )
        reference = x * (1.0 + expanded_scale) + expanded_shift
        torch.testing.assert_close(result.cpu(), reference.cpu())

    def test_rotary_embedding_matches_reference(self):
        x = torch.randn(4, 3, 8, device=self.device)
        cos = torch.randn(4, 4, device=self.device)
        sin = torch.randn(4, 4, device=self.device)

        result = apply_rotary_embedding_native(x, cos, sin)
        cos_expanded = cos.unsqueeze(-2)
        sin_expanded = sin.unsqueeze(-2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        reference = torch.stack(
            (
                x1 * cos_expanded - x2 * sin_expanded,
                x2 * cos_expanded + x1 * sin_expanded,
            ),
            dim=-1,
        ).flatten(-2)
        torch.testing.assert_close(result.cpu(), reference.cpu())

        full_cos = torch.repeat_interleave(cos, 2, dim=-1)
        full_sin = torch.repeat_interleave(sin, 2, dim=-1)
        interleaved = apply_rotary_embedding_native(
            x, full_cos, full_sin, interleaved=True
        )
        torch.testing.assert_close(interleaved.cpu(), reference.cpu())

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
