import itertools
import math
import unittest

# TODO: use interface in cpu.py
import sgl_kernel
import torch
import torch.nn as nn
from utils import (
    BLOCK_K,
    BLOCK_N,
    SiluAndMul,
    factor_for_scale,
    fp8_max,
    fp8_min,
    per_token_quant_int8,
    precision,
    scaled_weight,
    torch_naive_moe,
    torch_w8a8_per_column_moe,
)

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestSharedExpert(CustomTestCase):
    M = [2, 121]
    N = [32, 32 * 4]
    K = [32, 32 * 2]
    routed_scaling_factor = [16]

    M_fp8 = [2, 12]
    N_fp8 = [512]
    K_fp8 = [256]

    def _bf16_shared_expert(self, m, n, k, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = torch.randn(m, k, dtype=dtype) / k

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        # bfloat16
        ref = torch_naive_moe(
            hidden_states.float(),
            w1.float(),
            w2.float(),
            fused_output.float(),
            routed_scaling_factor,
        ).to(dtype=dtype)
        res = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, res, atol=atol, rtol=rtol)

    def test_bf16_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
            ):
                self._bf16_shared_expert(*params)

    def _int8_shared_expert(self, m, n, k, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = torch.randn(m, k, dtype=dtype) / k

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        w1_q, w1_s = per_token_quant_int8(w1)
        w2_q, w2_s = per_token_quant_int8(w2)
        ref2 = torch_w8a8_per_column_moe(
            hidden_states2.float(),
            w1_q,
            w2_q,
            w1_s,
            w2_s,
            fused_output.float(),
            routed_scaling_factor,
        ).to(dtype=dtype)
        res2 = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1_q,
            w2_q,
            fused_output,
            routed_scaling_factor,
            True,
            True,
            False,
            w1_s,
            w2_s,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref2.dtype]
        torch.testing.assert_close(ref2, res2, atol=atol, rtol=rtol)

    def test_int8_shared_expert(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                routed_scaling_factor=params[3],
            ):
                self._int8_shared_expert(*params)

    def _fp8_shared_expert(self, M, N, K, routed_scaling_factor):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(1, 2 * N, K)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(1, K, N)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(1, 2 * N // BLOCK_N, K // BLOCK_K) * factor_for_scale
        w2s = torch.randn(1, K // BLOCK_N, N // BLOCK_K) * factor_for_scale

        w1_scaled = scaled_weight(w1, w1s).view(2 * N, K)
        w2_scaled = scaled_weight(w2, w2s).view(K, N)

        # change back to 2D
        w1, w2 = w1.squeeze(0), w2.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
        w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

        fused_out = torch.randn(M, K, dtype=dtype) / math.sqrt(K)
        a2 = a.clone()

        # ref
        ic0 = torch.matmul(a.float(), w1_scaled.transpose(0, 1))
        ic1 = SiluAndMul(ic0)
        shared_out = torch.matmul(ic1, w2_scaled.transpose(0, 1))
        ref_out = shared_out + fused_out.float() * routed_scaling_factor
        ref_out = ref_out.to(dtype=dtype)

        w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)  # [2N, K]
        w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)  # [K, N]
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            a2,
            w1,
            w2,
            fused_out,
            routed_scaling_factor,
            True,
            False,
            True,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,
        )

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_fp8_shared_expert(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.routed_scaling_factor,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                routed_scaling_factor=params[3],
            ):
                self._fp8_shared_expert(*params)


if __name__ == "__main__":
    unittest.main()
