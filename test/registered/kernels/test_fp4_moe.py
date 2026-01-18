# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import pytest
import torch
from flashinfer import scaled_fp4_grouped_quantize
from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
from sgl_kernel import scaled_fp4_quant, silu_and_mul

from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.registered.kernels.moe.moe_utils import dequantize_nvfp4_to_dtype

register_cuda_ci(est_time=300, suite="nightly-4-gpu-b200", nightly=True)

if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1024),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


# Reference implementation of torch_moe
def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def flashinfer_cutedsl_grouped_gemm_nt_masked(
    hidden_states: torch.Tensor,  # 3d
    input_global_scale: torch.Tensor,  # (l,)
    weights: torch.Tensor,
    w_global_scale: torch.Tensor,  # (l,)
    masked_m: torch.Tensor,
):
    from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

    # hidden_states: [l, m, k]
    # weights: [l, n, k]
    aq, aq_sf = scaled_fp4_grouped_quantize(
        hidden_states,
        masked_m.to(hidden_states.device),
        input_global_scale,
    )
    num_experts, n, k = weights.shape
    bq, bq_sf = scaled_fp4_grouped_quantize(
        weights,
        torch.ones(num_experts, device=weights.device, dtype=torch.int32) * n,
        w_global_scale,
    )

    out = torch.zeros(
        (num_experts, max(masked_m), n), dtype=weights.dtype, device=aq.device
    )
    out = out.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"
    alpha = 1.0 / (input_global_scale * w_global_scale).to(out.dtype).view(
        1, 1, num_experts
    )

    def get_cute_dtype(input: torch.Tensor) -> str:
        if input.dtype == torch.bfloat16:
            return "bfloat16"
        elif input.dtype == torch.float16:
            return "float16"
        elif input.dtype == torch.float32:
            return "float32"
        else:
            raise ValueError(f"Unsupported cute dtype {input.dtype}")

    grouped_gemm_nt_masked(
        (aq, aq_sf),
        (bq, bq_sf),
        out,
        masked_m.to(aq.device),
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=alpha,
        alpha_dtype=get_cute_dtype(alpha),
    )

    return out


def check_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    moe_impl: Callable,
    flip_w13: bool,
):
    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )

    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = scaled_fp4_quant(
            w1[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = scaled_fp4_quant(
            w2[expert], w2_gs[expert]
        )

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    topk_weights, topk_ids, _ = topk_output

    a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
    a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
    test_output = moe_impl(
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_q=w1_q,
        w2_q=w2_q,
        a1_gs=a1_gs,
        w1_blockscale=w1_blockscale,
        w1_alphas=(1 / w1_gs),
        a2_gs=a2_gs,
        w2_blockscale=w2_blockscale,
        w2_alphas=(1 / w2_gs),
    )

    # Reference check:
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
    ).to(torch.float32)
    a_fp4, a_scale_interleaved = scaled_fp4_quant(a, a_global_scale)
    _, m_k = a_fp4.shape
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a_global_scale,
        dtype=a.dtype,
        device=a.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    if flip_w13:
        dim = -2
        size = w1_d.size(dim)
        assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
        half = size // 2
        # Reorder weight
        w1, w3 = w1_d.split(half, dim=dim)
        w1_d = torch.cat([w3, w1], dim=dim).contiguous()

    torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk, None)

    torch.testing.assert_close(torch_output, test_output, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_cutlass_fp4_moe_no_graph(
    m: int, n: int, k: int, e: int, topk: int, dtype: torch.dtype
):
    def cutlass_moe_impl(
        a,
        topk_weights,
        topk_ids,
        w1_q,
        w2_q,
        a1_gs,
        w1_blockscale,
        w1_alphas,
        a2_gs,
        w2_blockscale,
        w2_alphas,
    ):
        params = CutlassMoEParams(
            CutlassMoEType.BlockscaledFP4,
            device=a.device,
            num_experts=e,
            intermediate_size_per_partition=n,  # n
            hidden_size=k,
        )  # k
        return cutlass_moe_fp4(
            a=a,
            a1_gscale=a1_gs,
            w1_fp4=w1_q,
            w1_blockscale=w1_blockscale,
            w1_alphas=w1_alphas,
            a2_gscale=a2_gs,
            w2_fp4=w2_q,
            w2_blockscale=w2_blockscale,
            w2_alphas=w2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            params=params,
            apply_router_weight_on_input=False,
        )

    check_moe(m, n, k, e, topk, dtype, cutlass_moe_impl, flip_w13=False)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int, n: int, k: int, e: int, topk: int, dtype: torch.dtype
):
    def flashinfer_moe_impl(
        a,
        topk_weights,
        topk_ids,
        w1_q,
        w2_q,
        a1_gs,
        w1_blockscale,
        w1_alphas,
        a2_gs,
        w2_blockscale,
        w2_alphas,
    ):
        return flashinfer_cutlass_fused_moe(
            a,
            topk_ids.to(torch.int),
            topk_weights,
            w1_q.view(torch.long),
            w2_q.view(torch.long),
            a.dtype,
            quant_scales=[
                a1_gs,
                w1_blockscale.view(torch.int32),
                w1_alphas,
                a2_gs,
                w2_blockscale.view(torch.int32),
                w2_alphas,
            ],
        )[0]

    check_moe(m, n, k, e, topk, dtype, flashinfer_moe_impl, flip_w13=True)


if __name__ == "__main__":
    test_cutlass_fp4_moe_no_graph(224, 1024, 1024, 256, 8, torch.half)
    test_flashinfer_fp4_moe_no_graph(224, 1024, 1024, 256, 8, torch.half)
