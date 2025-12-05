import itertools
import unittest

# TODO: use interface in cpu.py
import torch
import torch.nn as nn
from utils import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
)

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


class TestGemm(CustomTestCase):
    M = [1, 101]
    N = [16, 32 * 13]
    K = [32 * 16]
    has_bias = [False, True]

    M_int8 = [2, 128]
    N_int8 = [32 * 12]
    K_int8 = [32 * 17]

    M_fp8 = [1, 11]
    N_fp8 = [128, 224]
    K_fp8 = [512, 576]

    def _bf16_gemm(self, M, N, K, has_bias):

        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)

        ref = torch.matmul(mat1.float(), mat2.float().t())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias.bfloat16())

        ref = ref.bfloat16()

        out = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, mat2, bias if has_bias else None, False
        )

        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
        out2 = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, packed_mat2, bias if has_bias else None, True
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out2, atol=atol, rtol=rtol)

    def test_bf16_gemm(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._bf16_gemm(*params)

    def _bf16_gemm_with_small_oc(self, M, N, K, has_bias, use_post_sigmul):
        use_post_sigmul = use_post_sigmul and N == 1
        mat_mul = (
            None if not use_post_sigmul else torch.randn(M, 2 * K, dtype=torch.bfloat16)
        )
        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)

        ref = torch.nn.functional.linear(mat1, mat2)
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias)
        if use_post_sigmul:
            ref = torch.nn.functional.sigmoid(ref) * mat_mul
            out = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,
                mat_mul if use_post_sigmul else None,
            )
        else:
            out = torch.ops.sgl_kernel.weight_packed_linear(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,
            )
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_bf16_gemm_with_small_oc(self):
        for params in itertools.product(
            [1, 8, 32, 1024], [12, 1], self.K, self.has_bias, [False, True]
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                use_post_sigmul=params[4],
            ):
                self._bf16_gemm_with_small_oc(*params)

    def _int8_gemm(self, M, N, K, has_bias):
        dtype = torch.bfloat16
        A = torch.randn((M, K), dtype=dtype) / 10
        Aq, As = per_token_quant_int8(A)

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = torch.rand(N) * factor_for_scale

        bias = torch.randn(N) if has_bias else None
        ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

        atol = rtol = precision[ref_out.dtype]

        Aq2, As2 = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
        out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            Aq2, Bq, As2, Bs, bias if has_bias else None, torch.bfloat16, False
        )
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        # test the fused version
        fused_out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A, Bq, Bs, bias if has_bias else None, torch.bfloat16, False
        )
        torch.testing.assert_close(ref_out, fused_out, atol=atol, rtol=rtol)

    def test_int8_gemm(self):
        for params in itertools.product(
            self.M_int8,
            self.N_int8,
            self.K_int8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._int8_gemm(*params)

    def _fp8_gemm(self, M, N, K, has_bias):
        prepack = True
        chunk = False
        scale_block_size_N = 64
        scale_block_size_K = 128
        assert scale_block_size_N <= N
        assert scale_block_size_K <= K
        A_dtype = torch.bfloat16

        model = Mod(K, N, has_bias).eval()
        if chunk:
            data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
        else:
            data = torch.randn(M, K, dtype=A_dtype)

        weight = model.linear.weight  # (N, K)

        if has_bias:
            bias = model.linear.bias

        fp8_weight, scales, dq_weight = convert_weight(
            weight, [scale_block_size_N, scale_block_size_K], A_dtype
        )

        if has_bias:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
        else:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T)

        if prepack:
            fp8_weight = torch.ops.sgl_kernel.convert_weight_packed(fp8_weight)

        opt = torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            data,
            fp8_weight,
            scales,
            [scale_block_size_N, scale_block_size_K],
            bias if has_bias else None,
            data.dtype,
            prepack,
        )
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, opt, atol=atol, rtol=rtol)

    def test_fp8_gemm(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._fp8_gemm(*params)


if __name__ == "__main__":
    unittest.main()
