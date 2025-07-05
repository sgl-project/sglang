import itertools
import math
import unittest

# TODO: use interface in cpu.py
import sgl_kernel
import torch

kernel = torch.ops.sgl_kernel

torch.manual_seed(1234)

from utils import (
    BLOCK_K,
    BLOCK_N,
    factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_w8a8_per_column_fused_moe,
)

from sglang.test.test_utils import CustomTestCase


def fused_moe(a, w1, w2, score, topk, renormalize, prepack):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = kernel.grouped_topk_cpu(
        a, score, topk, renormalize, G, topk_group, 0, None, None
    )

    packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
    packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2

    inplace = True
    return kernel.fused_experts_cpu(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        inplace,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        prepack,
    )


class TestFusedExperts(CustomTestCase):
    M = [2, 114]
    N = [32]
    K = [32]
    E = [4]
    topk = [2]
    renormalize = [False, True]

    M_int8 = [1, 39]
    N_int8 = [128]
    K_int8 = [256]
    E_int8 = [8]
    topk_int8 = [3]

    M_fp8 = [2, 121]
    N_fp8 = [512]
    K_fp8 = [256]
    E_fp8 = [8]
    topk_fp8 = [4]

    def _bf16_moe(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)

        torch_output = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)
        fused_output = fused_moe(a, w1, w2, score, topk, renormalize, prepack)

        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    def test_bf16_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
            self.renormalize,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                e=params[3],
                topk=params[4],
                renormalize=params[5],
            ):
                self._bf16_moe(*params)

    def _int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        prepack = True

        # Initialize int8 quantization parameters
        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        # Input tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

        # Generate int8 weights
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        # Generate scale for each column (per-column quantization)
        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * int8_factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * int8_factor_for_scale

        # Calculate routing
        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = True
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        out = kernel.fused_experts_cpu(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids.to(torch.int32),
            inplace,
            True,
            False,
            w1_s,
            w2_s,
            None,
            None,
            None,
            prepack,
        )

        atol = rtol = precision[ref_out.dtype]
        # Increase the tolerance for large input shapes
        if M > 35:
            atol = rtol = 0.02
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_int8_moe(self):
        for params in itertools.product(
            self.M_int8,
            self.N_int8,
            self.K_int8,
            self.E_int8,
            self.topk_int8,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
            ):
                self._int8_moe(*params)

    def _fp8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(E, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(E, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(E, 2 * N // BLOCK_N, K // BLOCK_K) * factor_for_scale
        w2s = torch.randn(E, K // BLOCK_N, N // BLOCK_K) * factor_for_scale

        w1_scaled = scaled_weight(w1, w1s)
        w2_scaled = scaled_weight(w2, w2s)

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        ref_out = native_fp8_fused_moe(
            a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk
        )
        out = kernel.fused_experts_cpu(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids.to(torch.int32),
            False,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    def test_fp8_moe(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.E_fp8,
            self.topk_fp8,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
            ):
                self._fp8_moe(*params)


if __name__ == "__main__":
    unittest.main()
