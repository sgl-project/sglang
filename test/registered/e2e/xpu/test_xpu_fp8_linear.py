"""
SGLang XPU backend integration tests for FP8 scaled_mm linear paths.

Usage:
    python3 -m unittest test.registered.e2e.xpu.test_xpu_fp8_linear
    pytest test/registered/e2e/xpu/test_xpu_fp8_linear.py
"""

import unittest
from typing import List

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    dispatch_w8a8_block_fp8_linear,
    per_token_group_quant_fp8,
    torch_w8a8_block_fp8_linear,
    use_rowwise_torch_scaled_mm,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


def baseline_block_fp8_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Reference for block-FP8 linear dequantization and matmul on XPU."""
    block_n, block_k = block_size
    input_2d = input.view(-1, input.shape[-1])
    q_input, activation_scale = per_token_group_quant_fp8(input_2d, block_k)

    # Dequantize A: q_input [M, K], activation_scale [M, K // block_k]
    M, K = q_input.shape
    scale_a_expanded = (
        activation_scale.unsqueeze(-1).expand(M, K // block_k, block_k).reshape(M, K)
    )
    a_dequant = q_input.float() * scale_a_expanded.float()

    # Dequantize B: weight [N, K], weight_scale [N // block_n, K // block_k]
    N = weight.shape[0]
    scale_b_expanded = (
        weight_scale.unsqueeze(1)
        .unsqueeze(-1)
        .expand(N // block_n, block_n, K // block_k, block_k)
        .reshape(N, K)
    )
    b_dequant = weight.float() * scale_b_expanded.float()

    output = torch.matmul(a_dequant, b_dequant.t())
    if bias is not None:
        output = output + bias.float()
    return output.to(dtype=input.dtype).view(*input.shape[:-1], N)


class TestXPUFP8Linear(CustomTestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU is not available")
        torch.manual_seed(42)
        self.device = "xpu"

    def test_w8a8_block_fp8_dispatch_on_xpu(self):
        """Verify that dispatch_w8a8_block_fp8_linear cleanly routes to torch_w8a8_block_fp8_linear on XPU."""
        dispatched_fn = dispatch_w8a8_block_fp8_linear()
        self.assertIs(dispatched_fn, torch_w8a8_block_fp8_linear)

    def test_torch_w8a8_block_fp8_linear_shapes_and_dims(self):
        """Test small sizes with varied M, 2D/3D shapes, and bias to keep memory minimal."""
        K, N = 256, 256
        block_size = [128, 128]

        # Weights: [N, K], weight_scale: [N // 128, K // 128]
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(
            N // 128, K // 128, dtype=torch.float32, device=self.device
        )
        bias = torch.randn(N, dtype=torch.bfloat16, device=self.device)

        # 1. 2D inputs with different M (decode M=1, small batch M=8, unaligned M=17)
        for M in [1, 8, 17]:
            x_2d = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
            out_no_bias = torch_w8a8_block_fp8_linear(
                x_2d, weight, block_size, weight_scale
            )
            self.assertEqual(out_no_bias.shape, (M, N))
            self.assertEqual(out_no_bias.dtype, torch.bfloat16)

            out_bias = torch_w8a8_block_fp8_linear(
                x_2d, weight, block_size, weight_scale, bias=bias
            )
            self.assertEqual(out_bias.shape, (M, N))

        # 2. 3D input: [Batch, SeqLen, Hidden]
        x_3d = torch.randn(2, 4, K, dtype=torch.bfloat16, device=self.device)
        out_3d = torch_w8a8_block_fp8_linear(x_3d, weight, block_size, weight_scale)
        self.assertEqual(out_3d.shape, (2, 4, N))

    def test_torch_w8a8_block_fp8_linear_prequantized(self):
        """Test pre-quantized input branch (input_scale is not None)."""
        M, K, N = 16, 256, 256
        block_size = [128, 128]
        block_k = block_size[1]

        x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        q_input, input_scale = per_token_group_quant_fp8(x_bf16, block_k)

        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(
            N // 128, K // 128, dtype=torch.float32, device=self.device
        )

        out = torch_w8a8_block_fp8_linear(
            q_input, weight, block_size, weight_scale, input_scale=input_scale
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_torch_w8a8_block_fp8_linear_non_contiguous_views(self):
        """Test that pre-quantized non-contiguous slice views are handled safely without crash."""
        M, K, N = 16, 256, 256
        block_size = [128, 128]

        x = torch.randn(M * 2, K, dtype=torch.bfloat16, device=self.device)
        qx, scale = per_token_group_quant_fp8(x, block_size[1])
        qx_strided = qx[::2, :]
        scale_strided = scale[::2, :]
        self.assertFalse(qx_strided.is_contiguous())
        self.assertFalse(scale_strided.is_contiguous())

        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(
            N // 128, K // 128, dtype=torch.float32, device=self.device
        )

        out_q_strided = torch_w8a8_block_fp8_linear(
            qx_strided,
            weight,
            block_size,
            weight_scale,
            input_scale=scale_strided.contiguous(),
        )
        self.assertEqual(out_q_strided.shape, (M, N))

        out_scale_strided = torch_w8a8_block_fp8_linear(
            qx_strided.contiguous(),
            weight,
            block_size,
            weight_scale,
            input_scale=scale_strided,
        )
        torch.testing.assert_close(out_scale_strided, out_q_strided)

        scale_transpose_contiguous = scale_strided.t().contiguous().t()
        self.assertFalse(scale_transpose_contiguous.is_contiguous())
        self.assertTrue(scale_transpose_contiguous.t().is_contiguous())
        out_scale_transpose_contiguous = torch_w8a8_block_fp8_linear(
            qx_strided.contiguous(),
            weight,
            block_size,
            weight_scale,
            input_scale=scale_transpose_contiguous,
        )
        torch.testing.assert_close(out_scale_transpose_contiguous, out_q_strided)

        weight_noncontiguous = torch.empty(
            N * 2, K, dtype=torch.float8_e4m3fn, device=self.device
        )
        weight_noncontiguous[::2] = weight
        weight_noncontiguous = weight_noncontiguous[::2]
        self.assertFalse(weight_noncontiguous.is_contiguous())
        self.assertEqual(weight_noncontiguous.stride(-1), 1)
        weight_scale_noncontiguous = weight_scale.t().contiguous().t()
        out_weight_views = torch_w8a8_block_fp8_linear(
            qx_strided.contiguous(),
            weight_noncontiguous,
            block_size,
            weight_scale_noncontiguous,
            input_scale=scale_strided.contiguous(),
        )
        self.assertEqual(out_weight_views.shape, (M, N))
        torch.testing.assert_close(out_weight_views, out_q_strided)

        weight_last_dim_strided = torch.empty(
            N, K, 2, dtype=torch.float8_e4m3fn, device=self.device
        )
        weight_last_dim_strided[..., 0] = weight
        weight_last_dim_strided = weight_last_dim_strided[..., 0]
        self.assertFalse(weight_last_dim_strided.is_contiguous())
        self.assertNotEqual(weight_last_dim_strided.stride(-1), 1)
        out_weight_strided = torch_w8a8_block_fp8_linear(
            qx_strided.contiguous(),
            weight_last_dim_strided,
            block_size,
            weight_scale,
            input_scale=scale_strided.contiguous(),
        )
        torch.testing.assert_close(out_weight_strided, out_q_strided)

    def test_torch_w8a8_block_fp8_linear_rejects_invalid_block_size(self):
        x = torch.randn(8, 256, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(256, 256, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(2, 2, dtype=torch.float32, device=self.device)
        for block_size in ([], [128], [128, 128, 128], [64, 64]):
            with self.subTest(block_size=block_size), self.assertRaises(ValueError):
                torch_w8a8_block_fp8_linear(x, weight, block_size, weight_scale)

    def test_torch_w8a8_block_fp8_linear_non_square_weight_scale(self):
        """Keep the public v2 weight-scale orientation correct when N != K."""
        M, K, N = 8, 256, 384
        block_size = [128, 128]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(
            N // 128, K // 128, dtype=torch.float32, device=self.device
        )
        out = torch_w8a8_block_fp8_linear(x, weight, block_size, weight_scale)
        self.assertEqual(out.shape, (M, N))

    def test_torch_w8a8_block_fp8_linear_numerical_accuracy(self):
        """Compare torch_w8a8_block_fp8_linear with dequantized baseline on XPU."""
        M, K, N = 16, 256, 256
        block_size = [128, 128]

        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
        )
        bias = torch.randn(N, dtype=torch.bfloat16, device=self.device)

        out_torch = torch_w8a8_block_fp8_linear(
            x, weight, block_size, weight_scale, bias=bias
        )
        out_ref = baseline_block_fp8_matmul(
            x, weight, block_size, weight_scale, bias=bias
        )

        torch.testing.assert_close(out_torch, out_ref, rtol=0.05, atol=0.2)

    def test_rowwise_torch_scaled_mm_xpu(self):
        """Verify rowwise torch scaled_mm path on XPU."""
        self.assertTrue(use_rowwise_torch_scaled_mm())

        M, K, N = 64, 256, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(K, N, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.ones(N, 1, dtype=torch.float32, device=self.device)

        out = apply_fp8_linear(
            input=x,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=None,
            use_per_token_if_dynamic=True,
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
