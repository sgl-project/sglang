import math
import unittest

import torch

kernel = torch.ops.sgl_kernel

from utils import (
    BLOCK_K,
    BLOCK_N,
    MXFP4QuantizeUtil,
    SiluAndMul,
    factor_for_scale,
    fp8_max,
    fp8_min,
    parametrize,
    per_token_quant_int8,
    precision,
    scaled_weight,
    torch_naive_clamped_silu_moe,
    torch_naive_moe,
    torch_w8a8_per_column_moe,
)

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class TestSharedExpert(CustomTestCase):
    @parametrize(
        m=[2, 121],
        n=[32, 32 * 4],
        k=[32, 32 * 2],
        routed_scaling_factor=[16],
        apply_scaling_factor=[True, False],
    )
    def test_bf16_shared_expert(
        self, m, n, k, routed_scaling_factor, apply_scaling_factor
    ):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = (
            torch.randn(m, k, dtype=dtype) / k if apply_scaling_factor else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        # bfloat16
        ref = torch_naive_moe(
            hidden_states,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            output_dtype=dtype,
        )
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
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
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(
        m=[2, 121],
        n=[32, 32 * 4],
        k=[32, 32 * 2],
        routed_scaling_factor=[16],
        apply_scaling_factor=[True, False],
    )
    def test_int8_shared_expert(
        self, m, n, k, routed_scaling_factor, apply_scaling_factor
    ):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / k
        w1 = torch.randn(2 * n, k, dtype=dtype)
        w2 = torch.randn(k, n, dtype=dtype)
        fused_output = (
            torch.randn(m, k, dtype=dtype) / k if apply_scaling_factor else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None

        # fused moe mutates content in hs
        hidden_states2 = hidden_states.clone()

        w1_q, w1_s = per_token_quant_int8(w1)
        w2_q, w2_s = per_token_quant_int8(w2)
        ref = torch_w8a8_per_column_moe(
            hidden_states,
            w1_q,
            w2_q,
            w1_s,
            w2_s,
            fused_output,
            routed_scaling_factor,
        )
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1_q,
            w2_q,
            fused_output,
            routed_scaling_factor,
            True,
            True,
            False,
            False,
            w1_s,
            w2_s,
            None,
            None,
            None,
            False,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(
        m=[2, 12],
        n=[512],
        k=[256],
        routed_scaling_factor=[16],
        apply_scaling_factor=[True, False],
    )
    def test_fp8_shared_expert(
        self, m, n, k, routed_scaling_factor, apply_scaling_factor
    ):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / math.sqrt(k)

        w1_fp32 = torch.randn(1, 2 * n, k)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(1, k, n)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(1, 2 * n // BLOCK_N, k // BLOCK_K) * factor_for_scale
        w2s = torch.randn(1, k // BLOCK_N, n // BLOCK_K) * factor_for_scale

        w1_scaled = scaled_weight(w1, w1s).view(2 * n, k)
        w2_scaled = scaled_weight(w2, w2s).view(k, n)

        # change back to 2D
        w1, w2 = w1.squeeze(0), w2.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
        w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

        fused_output = (
            torch.randn(m, k, dtype=dtype) / math.sqrt(k)
            if apply_scaling_factor
            else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None
        hidden_states2 = hidden_states.clone()

        # ref with bfloat16
        ref = torch_naive_moe(
            hidden_states,
            w1_scaled,
            w2_scaled,
            fused_output,
            routed_scaling_factor,
            output_dtype=dtype,
        )

        w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)  # [2N, K]
        w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)  # [K, N]
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            True,
            False,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            None,
            True,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(
        m=[2, 12],
        n=[512],
        k=[256],
        routed_scaling_factor=[16],
        apply_scaling_factor=[True, False],
        limit=[None, 10.0],
    )
    def test_fp8_shared_expert_dsv4(
        self, m, n, k, routed_scaling_factor, apply_scaling_factor, limit
    ):
        dtype = torch.bfloat16

        hidden_states = torch.randn(m, k, dtype=dtype) / math.sqrt(k)

        w1_fp32 = torch.randn(1, 2 * n, k)
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = torch.randn(1, k, n)
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w1s = torch.randn(1, 2 * n // BLOCK_N, k // BLOCK_K) * factor_for_scale
        w2s = torch.randn(1, k // BLOCK_N, n // BLOCK_K) * factor_for_scale

        w1_scaled = scaled_weight(w1, w1s).view(2 * n, k)
        w2_scaled = scaled_weight(w2, w2s).view(k, n)

        # change back to 2D
        w1, w2 = w1.squeeze(0), w2.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
        w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

        fused_output = (
            torch.randn(m, k, dtype=dtype) / math.sqrt(k)
            if apply_scaling_factor
            else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None
        hidden_states2 = hidden_states.clone()

        # ref with bfloat16
        if limit is None:
            ref = torch_naive_moe(
                hidden_states,
                w1_scaled,
                w2_scaled,
                fused_output,
                routed_scaling_factor,
                output_dtype=dtype,
            )
        else:
            ref = torch_naive_clamped_silu_moe(
                hidden_states,
                w1_scaled,
                w2_scaled,
                fused_output,
                routed_scaling_factor,
                limit,
                output_dtype=dtype,
            )

        w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)  # [2N, K]
        w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)  # [K, N]
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states2,
            w1,
            w2,
            fused_output,
            routed_scaling_factor,
            True,
            False,
            True,
            False,
            w1s,
            w2s,
            [BLOCK_N, BLOCK_K],
            None,
            limit,
            True,
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(
        M=[2, 12],
        N=[512],
        K=[256],
        routed_scaling_factor=[16],
        apply_scaling_factor=[True, False],
    )
    def test_mxfp4_shared_expert(
        self, M, N, K, routed_scaling_factor, apply_scaling_factor
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        dtype = torch.bfloat16
        prepack = True

        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1_fp32 = torch.randn(1, 2 * N, K) / 10
        w1q, w1s = MXFP4QuantizeUtil.quantize(w1_fp32)
        w1s = w1s.reshape(1, 2 * N, K // 32)
        w1dq = MXFP4QuantizeUtil.dequantize(w1q, torch.float32, w1s)

        w2_fp32 = torch.randn(1, K, N) / 10
        w2q, w2s = MXFP4QuantizeUtil.quantize(w2_fp32)
        w2s = w2s.reshape(1, K, N // 32)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q, torch.float32, w2s)

        fused_out = (
            torch.randn(M, K, dtype=dtype) / math.sqrt(K)
            if apply_scaling_factor
            else None
        )
        routed_scaling_factor = routed_scaling_factor if apply_scaling_factor else None
        a2 = a.clone()

        # ref
        ic0 = torch.matmul(a.float(), w1dq.squeeze(0).transpose(0, 1))
        ic1 = SiluAndMul(ic0)
        shared_out = torch.matmul(ic1, w2dq.squeeze(0).transpose(0, 1))
        ref_out = shared_out + (
            fused_out.float() * routed_scaling_factor if apply_scaling_factor else 0
        )
        ref_out = ref_out.to(dtype=dtype)

        # change back to 2D
        w1q, w2q = w1q.squeeze(0), w2q.squeeze(0)
        w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)

        w1 = kernel.convert_weight_packed(w1q)
        w2 = kernel.convert_weight_packed(w2q)
        w1s = kernel.convert_scale_packed(w1s)
        w2s = kernel.convert_scale_packed(w2s)
        out = torch.ops.sgl_kernel.shared_expert_cpu(
            a2,
            w1,
            w2,
            fused_out,
            routed_scaling_factor,
            True,
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

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
