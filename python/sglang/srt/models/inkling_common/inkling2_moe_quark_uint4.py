"""Forward of the UINT4-AWQ (W4A16) Quark experts with the rounding points of the BF16
expert path: the route weight is folded into the FP32 SwiGLU before its BF16 rounding,
and each token sums its BF16 down rows in FP32 in top-k order."""

import torch

from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as marlin_sm90
from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
from sglang.kernels.ops.moe import inkling2_quark_gemm_hip as quark_gemm
from sglang.kernels.ops.moe import inkling2_swiglu_bf16_hip as swiglu_hip


def forward_experts(experts, x, weights, ids, combine):
    top_k = ids.shape[1]
    num_local = experts.num_local_experts
    start = experts.ep_rank * num_local
    flat, gammas = ids.flatten(), weights.flatten()
    rows = flat.numel()
    # both GEMMs take the same block height: the config tables key on rows
    block_m = quark_gemm.default_config_quark(
        quark_gemm.UINT4_AWQ,
        rows,
        experts.num_global_experts,
        scale_dtype=experts.w13_scale.dtype,
    ).bm
    route = experts_hip.route(
        flat, start, num_local, experts.num_global_experts, gammas, block_m=block_m
    )
    acc = quark_gemm.expert_gemm_quark(
        x.contiguous(),
        experts.w13_weight,
        experts.w13_scale,
        experts.w13_weight_zero_point,
        route,
        experts.output_sizes["w13"],
        top_k,
        torch.bfloat16,
        rows,
        fmt=quark_gemm.UINT4_AWQ,
        num_experts=experts.num_global_experts,
    )
    hidden = swiglu_hip.swiglu_bf16(
        acc, flat, gammas, start, num_local, (experts.gate_cap, experts.up_cap)
    )
    down = quark_gemm.expert_gemm_quark(
        hidden,
        experts.w2_weight,
        experts.w2_scale,
        experts.w2_weight_zero_point,
        route,
        experts.output_sizes["w2"],
        1,
        torch.bfloat16,
        rows,
        fmt=quark_gemm.UINT4_AWQ,
        num_experts=experts.num_global_experts,
    )
    sums = marlin_sm90.source_partial(
        down, flat, (None, None, None), start, num_local, top_k, round_bf16=True
    )
    return experts.finish_forward(sums, combine, x.dtype)
