"""
SGLang XPU backend integration tests for FP8 scaled_mm linear paths.

Usage:
    python3 -m unittest test.registered.e2e.xpu.test_xpu_fp8_linear
    pytest test/registered/e2e/xpu/test_xpu_fp8_linear.py
"""

import unittest
from contextlib import contextmanager
from typing import List
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    dispatch_w8a8_block_fp8_linear,
    dispatch_w8a8_mxfp8_linear,
    get_fp8_gemm_runner_backend,
    mxfp8_group_quantize,
    prepare_xpu_block_scale_for_scaled_mm,
    torch_w8a8_block_fp8_linear,
    xpu_mxfp8_scaled_mm_linear,
)
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


def reference_block_fp8_matmul(
    q_input: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Reference using the exact quantized operands passed to scaled_mm."""
    block_n, block_k = block_size

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
    return output


@contextmanager
def fp8_gemm_backend_from_cli(backend: str):
    """Parse --fp8-gemm-backend through the CLI, publish it, and initialize from it."""
    server_args = prepare_server_args(
        ["--model-path", "dummy", "--fp8-gemm-backend", backend]
    )
    reset_context()
    publish(server_args, role="test", ranks=SpawnRanks(world_rank=0, dp_rank=None))
    try:
        with patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None):
            fp8_utils.initialize_fp8_gemm_config()
            yield
    finally:
        reset_context()


class TestXPUFP8Linear(CustomTestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU is not available")
        self.device = "xpu"

    def test_w8a8_block_fp8_dispatch_on_xpu(self):
        """Verify that dispatch_w8a8_block_fp8_linear cleanly routes to torch_w8a8_block_fp8_linear on XPU."""
        dispatched_fn = dispatch_w8a8_block_fp8_linear()
        self.assertIs(dispatched_fn, torch_w8a8_block_fp8_linear)

    def test_default_auto_backend_resolves_to_xpu_scaled_mm(self):
        """The default --fp8-gemm-backend=auto is pinned to xpu_scaled_mm on XPU."""
        with fp8_gemm_backend_from_cli("auto"):
            self.assertTrue(get_fp8_gemm_runner_backend().is_xpu_scaled_mm())

    def test_explicit_xpu_scaled_mm_backend_serves_block_fp8(self):
        """--fp8-gemm-backend=xpu_scaled_mm parses, initializes, and runs block FP8."""
        with fp8_gemm_backend_from_cli("xpu_scaled_mm"):
            self.assertTrue(get_fp8_gemm_runner_backend().is_xpu_scaled_mm())
            method = Fp8LinearMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
                )
            )
        self.assertIs(method.w8a8_block_fp8_linear, torch_w8a8_block_fp8_linear)

        M, K, N = 16, 256, 384
        block_size = [128, 128]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
        )
        out = method.w8a8_block_fp8_linear(x, weight, block_size, weight_scale)
        q_input, input_scale = per_token_group_quant_fp8(x, block_size[1])
        ref = reference_block_fp8_matmul(
            q_input, input_scale, weight, block_size, weight_scale
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)

    def test_w8a8_mxfp8_dispatch_on_xpu(self):
        dispatched_fn = dispatch_w8a8_mxfp8_linear()
        self.assertIs(dispatched_fn, xpu_mxfp8_scaled_mm_linear)

        dispatched_block_fn = dispatch_w8a8_block_fp8_linear(weight_block_size=[1, 32])
        self.assertIs(dispatched_block_fn, torch_w8a8_block_fp8_linear)

    def test_torch_w8a8_mxfp8_linear_prequantized(self):
        M, K, N = 8, 256, 384
        block_k = 32
        q_input = torch.randn(M, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        input_scale = torch.full(
            (M, K // block_k), 120, dtype=torch.uint8, device=self.device
        )
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // block_k), 121, dtype=torch.uint8, device=self.device
        )

        output = torch_w8a8_block_fp8_linear(
            q_input,
            weight,
            [1, 32],
            weight_scale,
            input_scale=input_scale,
        )
        input_scale_f = input_scale.view(torch.float8_e8m0fnu).float()
        weight_scale_f = weight_scale.view(torch.float8_e8m0fnu).float()
        input_scale_f = input_scale_f.unsqueeze(-1).expand(M, K // block_k, block_k)
        weight_scale_f = weight_scale_f.unsqueeze(-1).expand(N, K // block_k, block_k)
        reference = (
            (q_input.float() * input_scale_f.reshape(M, K))
            @ (weight.float() * weight_scale_f.reshape(N, K)).t()
        ).to(torch.bfloat16)
        self.assertEqual(output.shape, (M, N))
        torch.testing.assert_close(output, reference, rtol=0.05, atol=0.1)

    def test_torch_w8a8_mxfp8_linear_dynamic(self):
        M, K, N = 8, 256, 384
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // 32), 121, dtype=torch.uint8, device=self.device
        )

        output = torch_w8a8_block_fp8_linear(
            input_tensor, weight, [1, 32], weight_scale
        )
        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_fp8_linear_method_mxfp8_e2e(self):
        """Verify end-to-end forward through Fp8LinearMethod with use_mxfp8=True."""
        M, K, N = 8, 256, 384
        config = Fp8Config(is_checkpoint_fp8_serialized=True, use_mxfp8=True)
        method = Fp8LinearMethod(config)
        self.assertTrue(method.use_mxfp8)

        layer = torch.nn.Module()
        layer.logical_widths = [N]
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N
        layer.orig_dtype = torch.bfloat16
        layer.weight = torch.nn.Parameter(
            torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.full((N, K // 32), 127, dtype=torch.uint8, device=self.device),
            requires_grad=False,
        )
        layer.weight_scale_inv.format_ue8m0 = True
        layer.input_scale = None
        layer.deepgemm_mxfp8_input = False

        method.process_weights_after_loading(layer)

        # Dynamic activation quant
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        out_dyn = method.apply(layer, x)
        self.assertEqual(out_dyn.shape, (M, N))
        self.assertEqual(out_dyn.dtype, torch.bfloat16)

        # Pre-quantized activation
        qx = torch.randn(M, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        sx = torch.full((M, K // 32), 127, dtype=torch.uint8, device=self.device)
        out_pre = method.apply(layer, (qx, sx))
        self.assertEqual(out_pre.shape, (M, N))
        self.assertEqual(out_pre.dtype, torch.bfloat16)

    def test_xpu_mxfp8_scaled_mm_linear_3d_prequantized(self):
        """Verify xpu_mxfp8_scaled_mm_linear correctly handles 3D pre-quantized activation scale."""
        B, S, K, N = 2, 4, 256, 384
        block_k = 32
        qx = torch.randn(B, S, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        sx = torch.full(
            (B, S, K // block_k), 120, dtype=torch.uint8, device=self.device
        )
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // block_k), 121, dtype=torch.uint8, device=self.device
        )
        output = xpu_mxfp8_scaled_mm_linear(qx, weight, weight_scale, input_scale=sx)
        self.assertEqual(output.shape, (B, S, N))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_mxfp8_group_quantize_non_contiguous_assert(self):
        """Test that mxfp8_group_quantize asserts on non-contiguous input."""
        x = torch.randn(8, 64, dtype=torch.bfloat16, device=self.device)[::2, :]
        self.assertFalse(x.is_contiguous())
        with self.assertRaises(AssertionError):
            mxfp8_group_quantize(x)

    def test_mxfp8_group_quantize_accuracy(self):
        """Test mxfp8_group_quantize matches reference UE8M0 quantization on XPU."""
        x = torch.randn(8, 64, dtype=torch.bfloat16, device=self.device)
        q, s = mxfp8_group_quantize(x)
        self.assertEqual(q.shape, (8, 64))
        self.assertEqual(s.shape, (8, 2))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)

        # Compare with mathematical formula
        blocks = x.float().reshape(8, -1, 32)
        amax = blocks.abs().amax(-1).clamp(min=1e-10)
        exp = torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max))
        ref_scale = (exp + 127).to(torch.uint8)
        self.assertTrue(torch.equal(s, ref_scale))

    def test_torch_w8a8_mxfp8_linear_rejects_float32_weight_scale(self):
        """Verify TypeError when float32 weight_scale is passed to [1, 32] linear."""
        M, K, N = 8, 256, 384
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        float_weight_scale = torch.ones(
            (N, K // 32), dtype=torch.float32, device=self.device
        )
        with self.assertRaises(TypeError):
            torch_w8a8_block_fp8_linear(x, weight, [1, 32], float_weight_scale)

    def test_xpu_mxfp8_scaled_mm_linear_validations(self):
        """Verify input validation errors in xpu_mxfp8_scaled_mm_linear."""
        M, K, N = 8, 256, 384
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // 32), 127, dtype=torch.uint8, device=self.device
        )

        # 1. Reject float32 weight_scale
        with self.assertRaises(TypeError):
            xpu_mxfp8_scaled_mm_linear(
                torch.randn(M, K, dtype=torch.bfloat16, device=self.device),
                weight,
                torch.ones(N, K // 32, dtype=torch.float32, device=self.device),
            )

        # 2. Reject non-e4m3fn weight
        with self.assertRaises(TypeError):
            xpu_mxfp8_scaled_mm_linear(
                torch.randn(M, K, dtype=torch.bfloat16, device=self.device),
                weight.to(torch.bfloat16),
                weight_scale,
            )

        # 3. Reject wrong weight_scale shape
        with self.assertRaises(ValueError):
            xpu_mxfp8_scaled_mm_linear(
                torch.randn(M, K, dtype=torch.bfloat16, device=self.device),
                weight,
                torch.full((N, 4), 127, dtype=torch.uint8, device=self.device),
            )

        # 4. Reject non-e4m3fn pre-quantized input
        with self.assertRaises(TypeError):
            xpu_mxfp8_scaled_mm_linear(
                torch.randn(M, K, dtype=torch.bfloat16, device=self.device),
                weight,
                weight_scale,
                input_scale=torch.full(
                    (M, K // 32), 127, dtype=torch.uint8, device=self.device
                ),
            )

        # 5. Reject wrong input_scale shape in pre-quantized mode
        qx = torch.randn(M, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        with self.assertRaises(ValueError):
            xpu_mxfp8_scaled_mm_linear(
                qx,
                weight,
                weight_scale,
                input_scale=torch.full(
                    (M, 4), 127, dtype=torch.uint8, device=self.device
                ),
            )

    def test_xpu_mxfp8_scaled_mm_linear_output_dtype(self):
        """Verify custom output_dtype support and bias alignment in xpu_mxfp8_scaled_mm_linear."""
        M, K, N = 8, 256, 384
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // 32), 127, dtype=torch.uint8, device=self.device
        )
        bias_fp32 = torch.randn(N, dtype=torch.float32, device=self.device)
        out_fp16 = xpu_mxfp8_scaled_mm_linear(
            x, weight, weight_scale, bias=bias_fp32, output_dtype=torch.float16
        )
        self.assertEqual(out_fp16.dtype, torch.float16)

    def test_xpu_mxfp8_scaled_mm_linear_non_contiguous_weight(self):
        """Verify non-contiguous sliced weight is handled properly."""
        M, K, N = 8, 256, 384
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        # Sliced along dim 1: non-contiguous
        weight_large = torch.randn(
            N, K * 2, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        weight_sliced = weight_large[:, :K]
        self.assertFalse(weight_sliced.is_contiguous())
        weight_scale = torch.full(
            (N, K // 32), 127, dtype=torch.uint8, device=self.device
        )
        out = xpu_mxfp8_scaled_mm_linear(x, weight_sliced, weight_scale)
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_prepare_xpu_block_scale_for_mxfp8(self):
        """Verify prepare_xpu_block_scale_for_scaled_mm handles [1, 32] contiguous layout."""
        N, K_blocks = 64, 8
        scale_2d = torch.full(
            (N * 2, K_blocks), 127, dtype=torch.uint8, device=self.device
        )
        # Sliced along dim 0 with step 2 is non-contiguous
        non_contiguous_scale = scale_2d[::2, :]
        self.assertFalse(non_contiguous_scale.is_contiguous())

        reordered = prepare_xpu_block_scale_for_scaled_mm(
            non_contiguous_scale,
            [1, 32],
            torch_w8a8_block_fp8_linear,
        )
        self.assertTrue(reordered.is_contiguous())
        self.assertEqual(reordered.shape, (N, K_blocks))

    def test_mxfp8_group_quantize_all_zeros(self):
        """Verify mxfp8_group_quantize works on all-zero inputs."""
        zero_x = torch.zeros(8, 64, dtype=torch.bfloat16, device=self.device)
        qz, sz = mxfp8_group_quantize(zero_x)
        self.assertEqual(qz.shape, (8, 64))
        self.assertEqual(sz.shape, (8, 2))
        self.assertTrue(torch.all(qz.view(torch.uint8) == 0))

    def test_mxfp8_float32_input_support(self):
        """Verify mxfp8_group_quantize and xpu_mxfp8_scaled_mm_linear support float32 directly."""
        M, K, N = 8, 256, 384
        x_fp32 = torch.randn(M, K, dtype=torch.float32, device=self.device)
        q, s = mxfp8_group_quantize(x_fp32)
        self.assertEqual(q.shape, (M, K))
        self.assertEqual(s.shape, (M, K // 32))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)

        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full(
            (N, K // 32), 127, dtype=torch.uint8, device=self.device
        )
        out = xpu_mxfp8_scaled_mm_linear(x_fp32, weight, weight_scale)
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.float32)

    def test_torch_w8a8_block_fp8_linear_shapes_and_dims(self):
        """Test small sizes with varied M, 2D/3D shapes, and bias to keep memory minimal."""
        K, N = 256, 256
        block_size = [128, 128]

        # Weights: [N, K], weight_scale: [N // 128, K // 128]
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
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
            torch.testing.assert_close(
                out_bias, out_no_bias + bias, atol=0.05, rtol=0.01
            )

        # 2. 3D input: [Batch, SeqLen, Hidden]
        x_3d = torch.randn(2, 4, K, dtype=torch.bfloat16, device=self.device)
        out_3d = torch_w8a8_block_fp8_linear(x_3d, weight, block_size, weight_scale)
        self.assertEqual(out_3d.shape, (2, 4, N))

        x_3d_noncontiguous = x_3d.transpose(0, 1)
        out_3d_noncontiguous = torch_w8a8_block_fp8_linear(
            x_3d_noncontiguous, weight, block_size, weight_scale
        )
        self.assertEqual(out_3d_noncontiguous.shape, (4, 2, N))

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
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
        )

        out = torch_w8a8_block_fp8_linear(
            q_input, weight, block_size, weight_scale, input_scale=input_scale
        )
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

        scale_a = (
            input_scale.unsqueeze(-1).expand(M, K // block_k, block_k).reshape(M, K)
        )
        scale_b = (
            weight_scale.unsqueeze(1)
            .unsqueeze(-1)
            .expand(N // 128, 128, K // 128, 128)
            .reshape(N, K)
        )
        ref = torch.matmul(
            q_input.float() * scale_a.float(),
            (weight.float() * scale_b.float()).t(),
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)

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
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
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
        weight_scale = (
            torch.rand(N // 128, K // 128, dtype=torch.float32, device=self.device)
            + 0.1
        )
        out = torch_w8a8_block_fp8_linear(x, weight, block_size, weight_scale)
        q_input, input_scale = per_token_group_quant_fp8(x, block_size[1])
        ref = reference_block_fp8_matmul(
            q_input, input_scale, weight, block_size, weight_scale
        ).to(torch.bfloat16)
        self.assertEqual(out.shape, (M, N))
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)

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
        weight_scale_post_load = torch.empty_strided(
            weight_scale.shape,
            (1, weight_scale.shape[0]),
            dtype=weight_scale.dtype,
            device=weight_scale.device,
        )
        weight_scale_post_load.copy_(weight_scale)
        out_post_load = torch_w8a8_block_fp8_linear(
            x, weight, block_size, weight_scale_post_load, bias=bias
        )
        torch.testing.assert_close(out_post_load, out_torch, rtol=0, atol=0)
        q_input, input_scale = per_token_group_quant_fp8(x, block_size[1])
        out_ref = reference_block_fp8_matmul(
            q_input, input_scale, weight, block_size, weight_scale, bias=bias
        ).to(torch.bfloat16)

        torch.testing.assert_close(out_torch, out_ref, rtol=0.05, atol=0.1)

    def test_torch_w8a8_block_fp8_linear_1x128_recipe(self):
        """Validate the alternate [1, 128] weight-scale recipe with N != K."""
        M, K, N = 8, 256, 384
        block_size = [1, 128]
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(N, K // 128, dtype=torch.float32, device=self.device) + 0.1
        )
        bias = torch.randn(N, dtype=torch.bfloat16, device=self.device)

        out = torch_w8a8_block_fp8_linear(
            x, weight, block_size, weight_scale, bias=bias
        )
        weight_scale_post_load = torch.empty_strided(
            weight_scale.shape,
            (1, weight_scale.shape[0]),
            dtype=weight_scale.dtype,
            device=weight_scale.device,
        )
        weight_scale_post_load.copy_(weight_scale)
        out_post_load = torch_w8a8_block_fp8_linear(
            x, weight, block_size, weight_scale_post_load, bias=bias
        )
        torch.testing.assert_close(out_post_load, out, rtol=0, atol=0)
        q_input, input_scale = per_token_group_quant_fp8(x, block_size[1])
        ref = reference_block_fp8_matmul(
            q_input, input_scale, weight, block_size, weight_scale, bias
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)

    def test_channelwise_xpu_rowwise_scaled_mm_gate(self):
        """Use fused rowwise scaled_mm without changing accuracy."""
        method = Fp8LinearMethod(
            Fp8Config(is_checkpoint_fp8_serialized=False, weight_block_size=None)
        )
        K, N = 256, 384
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=self.device) / 3

        layer = torch.nn.Module()
        layer.logical_widths = [N]
        layer.weight = torch.nn.Parameter(weight_bf16, requires_grad=False)
        layer.orig_dtype = torch.bfloat16
        layer.input_scale = None
        self.assertFalse(method.cutlass_fp8_supported)
        method.process_weights_after_loading(layer)
        self.assertTrue(method.use_per_token_if_dynamic)
        weight = layer.weight
        weight_scale = layer.weight_scale
        self.assertEqual(tuple(weight_scale.shape), (N, 1))
        self.assertTrue(weight_scale.t().is_contiguous())
        original_scaled_mm = torch._scaled_mm

        for m in (1, 8):
            with self.subTest(m=m):
                x = torch.randn(m, K, dtype=torch.bfloat16, device=self.device) / 3
                q_input, input_scale = scaled_fp8_quant(
                    x, use_per_token_if_dynamic=True
                )
                reference = (q_input.float() * input_scale.float()) @ (
                    weight.float() * weight_scale.t()
                )
                calls = []

                def traced_scaled_mm(*args, **kwargs):
                    calls.append(
                        (
                            tuple(kwargs["scale_a"].shape),
                            tuple(kwargs["scale_b"].shape),
                        )
                    )
                    return original_scaled_mm(*args, **kwargs)

                with patch.object(torch, "_scaled_mm", side_effect=traced_scaled_mm):
                    output = apply_fp8_linear(
                        input=x,
                        weight=weight,
                        weight_scale=weight_scale,
                        use_per_token_if_dynamic=method.use_per_token_if_dynamic,
                        compressed_tensor_quant=True,
                    )

                torch.testing.assert_close(
                    output, reference.to(torch.bfloat16), rtol=0.01, atol=0.1
                )
                expected_scale_shapes = ((m, 1), (1, N))
                self.assertEqual(calls, [expected_scale_shapes])

    def test_static_input_scale_with_channelwise_weight_scale(self):
        """Static activation_scheme + per-channel weight_scale must not crash.

        process_weights_after_loading forces XPU through the channelwise
        weight_scale branch unconditionally, but a static checkpoint gives a
        0-dim per-tensor input_scale (not a per-token one). That combination
        used to hit `_apply_fallback_scaled_mm`'s `torch.narrow` on a 0-dim
        tensor.
        """
        method = Fp8LinearMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=None)
        )
        method.quant_config.activation_scheme = "static"
        N, K = 384, 256
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=self.device) / 3
        weight_scale = (
            weight_bf16.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            / torch.finfo(torch.float8_e4m3fn).max
        )

        layer = torch.nn.Module()
        layer.logical_widths = [N]
        layer.weight = torch.nn.Parameter(
            weight_bf16.to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.orig_dtype = torch.bfloat16
        layer.input_scale = torch.nn.Parameter(
            torch.tensor(0.05, device=self.device), requires_grad=False
        )
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.input_scale.shape, torch.Size([]))

        x = torch.randn(4, K, dtype=torch.bfloat16, device=self.device) / 3
        output = method.apply(layer, x)

        q_input = (
            (x.float() / layer.input_scale.float())
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        # layer.weight is stored transposed as (K, N) after loading.
        reference = (q_input.float() * layer.input_scale.float()) @ (
            layer.weight.float() * layer.weight_scale.float().t()
        )
        torch.testing.assert_close(
            output, reference.to(torch.bfloat16), rtol=0.05, atol=0.2
        )

    def test_prequantized_vector_scale_with_channelwise_weight_scale(self):
        """A one-element pre-quantized input scale is treated as scalar."""
        M, K, N = 4, 256, 384
        q_input = torch.randn(M, K, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight = torch.randn(K, N, dtype=torch.bfloat16, device=self.device).to(
            torch.float8_e4m3fn
        )
        weight_scale = torch.full((N, 1), 0.5, device=self.device)
        input_scale = torch.tensor([0.25], device=self.device)

        output = apply_fp8_linear(
            input=q_input,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
            pre_quant_output_dtype=torch.bfloat16,
        )
        reference = (q_input.float() * input_scale.float()) @ (
            weight.float() * weight_scale.t()
        )
        torch.testing.assert_close(
            output, reference.to(torch.bfloat16), rtol=0.01, atol=0.1
        )


if __name__ == "__main__":
    unittest.main()
