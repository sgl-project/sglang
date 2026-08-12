import math

import pytest

# TODO: use interface in cpu.py
import torch

from sglang.srt.layers.amx_utils import CPUQuantMethod

kernel = torch.ops.sgl_kernel

torch.manual_seed(1183)

dtype = torch.bfloat16
prepack = True
alpha = 1.702
limit = 7.0

from utils import (
    BLOCK_K,
    BLOCK_N,
    MXFP4QuantizeUtil,
    factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_naive_fused_moe_gptoss,
    torch_w8a8_per_column_fused_moe,
    unpack_and_dequant_awq,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")


def run_fused_experts(
    a,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    quant=CPUQuantMethod.UNQUANT,
    w1_scale=None,
    w2_scale=None,
    w1_zp=None,
    w2_zp=None,
    block_size=None,
    w1_bias=None,
    w2_bias=None,
    alpha=None,
    limit=None,
    is_vnni=True,
    inplace=False,
):
    return kernel.fused_experts_cpu(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids.to(torch.int32),
        inplace,
        quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        block_size,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
    )


def make_routing(m, e, topk, dtype, renormalize=False, score=None, return_score=False):
    if score is None:
        score = torch.randn((m, e), dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    if return_score:
        return topk_weight, topk_ids, score
    return topk_weight, topk_ids


def make_bf16_weights(e, out_dim, in_dim, with_bias=False):
    weight = torch.randn((e, out_dim, in_dim), dtype=dtype) / 10

    if not with_bias:
        return weight

    bias = torch.randn((e, out_dim), dtype=torch.float32) / 10
    return weight, bias


def make_int8_weights(e, out_dim, in_dim, int8_max=127, int8_min=-128):
    weight_fp32 = (torch.rand((e, out_dim, in_dim), dtype=torch.float32) - 0.5) * 2
    weight = (weight_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
    return weight


def make_fp8_weights(e, out_dim, in_dim):
    weight_fp32 = torch.randn(e, out_dim, in_dim)
    weight = (
        (weight_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    )

    weight_s = (
        torch.rand(e, math.ceil(out_dim / BLOCK_N), math.ceil(in_dim / BLOCK_K))
        * factor_for_scale
    )

    weight_scaled = scaled_weight(weight, weight_s)
    return weight, weight_s, weight_scaled


def make_mxfp4_weights(e, out_dim, in_dim, dtype, with_bias=False):
    weight_bf16 = torch.randn((e, out_dim, in_dim), dtype=dtype) / 10
    weight_q, weight_s = MXFP4QuantizeUtil.quantize(weight_bf16)
    weight_s = weight_s.reshape(e, out_dim, in_dim // 32)
    weight_dq = MXFP4QuantizeUtil.dequantize(weight_q, dtype, weight_s)

    weight_packed = kernel.convert_weight_packed(weight_q)
    weight_s_packed = kernel.convert_scale_packed(weight_s)

    if not with_bias:
        return weight_dq, weight_packed, weight_s_packed

    bias = torch.randn((e, out_dim), dtype=torch.float32) / 10
    return weight_dq, bias, weight_packed, weight_s_packed


class TestFusedExperts:

    @pytest.mark.parametrize("m", [2, 114])
    @pytest.mark.parametrize("n", [32])
    @pytest.mark.parametrize("k", [32])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False, True])
    def test_bf16_moe(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = make_bf16_weights(e, 2 * n, k)
        w2 = make_bf16_weights(e, k, n)
        topk_weights, topk_ids, score = make_routing(
            m,
            e,
            topk,
            dtype=dtype,
            renormalize=renormalize,
            return_score=True,
        )
        torch_output = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)

        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        fused_output = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weights,
            topk_ids,
            quant=CPUQuantMethod.UNQUANT,
            is_vnni=prepack,
            inplace=True,
        )

        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_bf16_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1, w1_b = make_bf16_weights(e, 2 * n, k, with_bias=True)
        w2, w2_b = make_bf16_weights(e, k, n, with_bias=True)
        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )
        torch_output = torch_naive_fused_moe_gptoss(
            a,
            w1,
            w2,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )
        packed_w1 = kernel.convert_weight_packed(w1)
        packed_w2 = kernel.convert_weight_packed(w2)
        fused_output = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.UNQUANT,
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [1, 39])
    @pytest.mark.parametrize("N", [128])
    @pytest.mark.parametrize("K", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [3])
    def test_int8_moe(self, M, N, K, E, topk):
        # Initialize int8 quantization parameters
        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        # Input tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

        # Generate int8 weights
        w1 = make_int8_weights(E, 2 * N, K, int8_max=int8_max, int8_min=int8_min)
        w2 = make_int8_weights(E, K, N, int8_max=int8_max, int8_min=int8_min)

        # Generate scale for each column (per-column quantization)
        w1_s = torch.rand(E, 2 * N) * int8_factor_for_scale
        w2_s = torch.rand(E, K) * int8_factor_for_scale

        # Calculate routing
        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = True
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        out = run_fused_experts(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT8_W8A8,
            w1_scale=w1_s,
            w2_scale=w2_s,
            is_vnni=prepack,
            inplace=inplace,
        )

        atol = rtol = precision[ref_out.dtype]
        # Increase the tolerance for large input shapes
        if M > 35:
            atol = rtol = 0.02
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [2, 121])
    @pytest.mark.parametrize("N", [352, 512])
    @pytest.mark.parametrize("K", [256, 320])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    def test_fp8_moe(self, M, N, K, E, topk):
        a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

        w1, w1s, w1_scaled = make_fp8_weights(E, 2 * N, K)
        w2, w2s, w2_scaled = make_fp8_weights(E, K, N)

        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        ref_out = native_fp8_fused_moe(
            a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk
        )
        out = run_fused_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.FP8_W8A16,
            w1_scale=w1s,
            w2_scale=w2s,
            block_size=[BLOCK_N, BLOCK_K],
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_fp8_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10

        w1, w1s, w1_scaled = make_fp8_weights(e, 2 * n, k)
        w2, w2s, w2_scaled = make_fp8_weights(e, k, n)
        w1_b = torch.randn((e, 2 * n), dtype=torch.float32) / 10

        w2_b = torch.randn((e, k), dtype=torch.float32) / 10

        w1_scaled = w1_scaled.to(dtype)
        w2_scaled = w2_scaled.to(dtype)

        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )

        ref_out = torch_naive_fused_moe_gptoss(
            a,
            w1_scaled,
            w2_scaled,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )

        w1 = kernel.convert_weight_packed(w1)
        w2 = kernel.convert_weight_packed(w2)

        out = run_fused_experts(
            a,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.FP8_W8A16,
            w1_scale=w1s,
            w2_scale=w2s,
            block_size=[BLOCK_N, BLOCK_K],
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [2, 121])
    @pytest.mark.parametrize("N", [352, 512])
    @pytest.mark.parametrize("K", [256, 320])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    def test_mxfp4_moe(self, M, N, K, E, topk):
        a = torch.randn(M, K, dtype=dtype) / 10

        w1dq, w1_packed, w1s_packed = make_mxfp4_weights(E, 2 * N, K, dtype=dtype)
        w2dq, w2_packed, w2s_packed = make_mxfp4_weights(E, K, N, dtype=dtype)

        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype)

        ref_out = native_fp8_fused_moe(
            a, w1dq.float(), w2dq.float(), topk_weight, topk_ids, topk
        )
        out = run_fused_experts(
            a,
            w1_packed,
            w2_packed,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.MXFP4,
            w1_scale=w1s_packed,
            w2_scale=w2s_packed,
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [1, 32])
    @pytest.mark.parametrize("n", [128, 64])
    @pytest.mark.parametrize("k", [128, 64])
    @pytest.mark.parametrize("e", [4])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("renormalize", [False])
    def test_mxfp4_moe_bias(self, m, n, k, e, topk, renormalize):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1dq, w1_b, w1_packed, w1s_packed = make_mxfp4_weights(
            e, 2 * n, k, dtype=dtype, with_bias=True
        )
        w2dq, w2_b, w2_packed, w2s_packed = make_mxfp4_weights(
            e, k, n, dtype=dtype, with_bias=True
        )
        topk_weight, topk_ids = make_routing(
            m, e, topk, dtype=dtype, renormalize=renormalize
        )
        torch_output = torch_naive_fused_moe_gptoss(
            a,
            w1dq,
            w2dq,
            w1_b,
            w2_b,
            topk_weight,
            topk_ids,
            renormalize,
            alpha,
            limit,
            e,
        )

        fused_output = run_fused_experts(
            a,
            w1_packed,
            w2_packed,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.MXFP4,
            w1_scale=w1s_packed,
            w2_scale=w2s_packed,
            w1_bias=w1_b,
            w2_bias=w2_b,
            alpha=alpha,
            limit=limit,
            is_vnni=True,
            inplace=False,
        )
        atol = rtol = precision[torch_output.dtype]
        torch.testing.assert_close(torch_output, fused_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("M", [1, 6])
    @pytest.mark.parametrize("N", [512])
    @pytest.mark.parametrize("K", [256])
    @pytest.mark.parametrize("E", [8])
    @pytest.mark.parametrize("topk", [4])
    def test_int4_moe(self, M, N, K, E, topk, group_size=128):
        a = torch.rand(M, K, dtype=dtype) / math.sqrt(K)

        awq_w13_weight = torch.randint(-127, 128, (E, K, 2 * N // 8)).to(torch.int)
        awq_w13_zero = torch.randint(0, 10, (E, K // group_size, 2 * N // 8)).to(
            torch.int
        )
        awq_w13_scales = (
            torch.rand(E, int(K // group_size), 2 * N) * factor_for_scale
        ).to(torch.bfloat16)

        awq_w2_weight = torch.randint(-127, 128, (E, N, K // 8)).to(torch.int)
        awq_w2_zero = torch.randint(0, 10, (E, N // group_size, K // 8)).to(torch.int)
        awq_w2_scales = (torch.rand(E, int(N // group_size), K) * factor_for_scale).to(
            torch.bfloat16
        )
        bf16_w13_weight = []
        bf16_w2_weight = []
        for i in range(E):
            bf16_w13_weight_i, _ = unpack_and_dequant_awq(
                awq_w13_weight[i], awq_w13_zero[i], awq_w13_scales[i], 4, 128
            )
            bf16_w2_weight_i, _ = unpack_and_dequant_awq(
                awq_w2_weight[i], awq_w2_zero[i], awq_w2_scales[i], 4, 128
            )
            bf16_w13_weight.append(bf16_w13_weight_i)
            bf16_w2_weight.append(bf16_w2_weight_i)
        bf16_w13_weight = torch.stack(bf16_w13_weight).detach()
        bf16_w2_weight = torch.stack(bf16_w2_weight).detach()

        score = torch.rand((M, E), dtype=dtype)

        ref_out = torch_naive_fused_moe(
            a, bf16_w13_weight, bf16_w2_weight, score, topk, False
        )
        topk_weight, topk_ids = make_routing(M, E, topk, dtype=dtype, score=score)
        awq_w13_weight_pack, awq_w13_zero_pack, awq_w13_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w13_weight, awq_w13_zero, awq_w13_scales, 0
            )
        )
        awq_w2_weight_pack, awq_w2_zero_pack, awq_w2_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w2_weight, awq_w2_zero, awq_w2_scales, 0
            )
        )

        out = run_fused_experts(
            a,
            awq_w13_weight_pack,
            awq_w2_weight_pack,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT4_W4A8,
            w1_scale=awq_w13_scales_pack,
            w2_scale=awq_w2_scales_pack,
            w1_zp=awq_w13_zero_pack,
            w2_zp=awq_w2_zero_pack,
            is_vnni=True,
            inplace=False,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
