"""ROCm forward of Inkling2's NVFP4 / MXFP8 experts (Inkling2Experts.uses_hip_path)
and its fused decode routing."""

import torch

from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90, marlin_sm90
from sglang.kernels.ops.inkling2.comm import all_reduce
from sglang.kernels.ops.inkling2.linear import linear_k_group_partials, uses_k_groups
from sglang.kernels.ops.inkling2.route_hip import (
    ROUTE_FUSED_MAX_TOKENS,
    ROUTE_QUANT_MAX_IDS,
    route_metadata_quantize,
    route_topk_metadata,
)
from sglang.srt.models.inkling_common.inkling2_moe import (
    BF16_COMBINE_MIN_TOKENS,
    RoutedOutput,
)
from sglang.srt.runtime_context import get_parallel


def mfma_reorders(shape):
    """(weight, scale) reorders that load quantized experts of this shape in
    experts_gluon's MFMA order, or (None, None) where Gluon does not take them."""
    if not (experts_hip.MFMA_ORDER and experts_hip.experts_gluon.fits(*shape[1:])):
        return None, None
    gluon = experts_hip.experts_gluon
    return gluon.mfma_order_weights, gluon.mfma_order_scales


def prepare_experts(experts):
    """After loading: the identity src2dst for the combined decode rows (as on
    Hopper), kept across reloads, since captured graphs hold its address."""
    if experts.w13_weight.is_cuda and getattr(experts, "_identity", None) is None:
        experts._identity = torch.arange(
            1 << 17, dtype=torch.int32, device=experts.w13_weight.device
        )


def moe_forward(moe, x, combine):
    """Inkling2MoE.forward on this path: gate_and_route's route and quantized x go
    straight to forward_experts."""
    weights, ids, route, quantized = gate_and_route(moe.gate, moe.experts, x)
    if not combine and get_parallel().moe_tp_size == 1:
        return forward_experts(moe.experts, x, weights, ids, False, route, quantized)
    out = forward_experts(moe.experts, x, weights, ids, True, route, quantized)
    if get_parallel().moe_tp_size > 1:
        out = all_reduce(out, get_parallel().moe_tp_group)
    return out


def forward_experts(experts, x, weights, ids, combine, route=None, quantized=None):
    """ROCm NVFP4 / MXFP8 experts in source (token, top-k) order, with the Hopper
    path's numerics: fake-quantized activations against BF16-dequantized weights.
    quantized is x's (fake-quantized rows, amax) when already computed.
    combine=False returns the all-reduced sums as Hopper's _forward_source does:
    rows in token order (identity src2dst, top_k 1) for the conv epilogue, which
    rounds them to BF16 and, for FP32 sums, keeps its own math in FP32."""
    tokens, top_k = ids.shape
    num_local = experts.w13_weight.shape[0]
    start = experts.ep_rank * num_local
    flat = ids.flatten()
    rows = flat.numel()
    gammas = weights.flatten()
    nvfp4 = experts.w13_weight.dtype == torch.uint8
    # both GEMMs take the same block height: default_config keys on rows
    block_m = experts_hip.default_config(
        rows, experts.num_global_experts, nvfp4, experts.in_mfma_order("w13")
    ).bm
    caps = (experts.gate_cap, experts.up_cap)
    act_dtype = experts_hip.activation_dtype(experts.w13_weight.dtype)
    if route is None and quantized is None and rows <= ROUTE_QUANT_MAX_IDS:
        # small batches: the routing metadata and x's fake quantization in one launch
        meta, quantized = route_metadata_quantize(
            flat,
            gammas,
            start,
            num_local,
            experts.num_global_experts,
            block_m,
            x.contiguous(),
            "nvfp4" if nvfp4 else "mxfp8",
            act_dtype,
        )
        _, dst2src, _, counts, offsets, _, schedule, _ = meta
        route = experts_hip.Route(dst2src, counts, offsets, block_m, schedule)
    if route is None:
        route = experts_hip.route(
            flat, start, num_local, experts.num_global_experts, gammas, block_m=block_m
        )
    if quantized is not None:
        xq, x_amax = quantized
    elif nvfp4:
        xq, x_amax = experts_sm90.fake_quantize(x.contiguous(), act_dtype)
    else:
        xq = experts_sm90.fake_quantize_mxfp8(x.contiguous())
    # the SwiGLU and combine kernels sum the GEMMs' split-K partials in split order
    acc = experts_hip.expert_gemm(
        xq,
        experts.w13_weight,
        experts.w13_scale,
        route,
        experts.output_sizes["w13"],
        top_k,
        torch.float32 if nvfp4 else torch.bfloat16,
        rows,
        num_experts=experts.num_global_experts,
        mfma_order=experts.in_mfma_order("w13"),
    )
    if nvfp4:
        hidden, hidden_amax = marlin_sm90.swiglu_fake_quantize(
            acc,
            flat,
            gammas,
            (x_amax, experts.w13_amax, experts.w13_outer_scale_denominator),
            start,
            num_local,
            top_k,
            caps,
            act_dtype,
        )
        scales = (hidden_amax, experts.w2_amax, experts.w2_outer_scale_denominator)
    else:
        hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(
            acc, flat, gammas, start, num_local, caps
        )
        scales = (None, None, None)
    down = experts_hip.expert_gemm(
        hidden,
        experts.w2_weight,
        experts.w2_scale,
        route,
        experts.output_sizes["w2"],
        1,
        torch.float32 if nvfp4 else torch.bfloat16,
        rows,
        num_experts=experts.num_global_experts,
        mfma_order=experts.in_mfma_order("w2"),
    )
    sums = marlin_sm90.source_partial(
        down, flat, scales, start, num_local, top_k, round_bf16=not nvfp4
    )
    if experts.ep_size > 1:
        # Like Hopper: BF16 partial sums above BF16_COMBINE_MIN_TOKENS halve the
        # prefill traffic; the reduction itself accumulates in FP32.
        if tokens > BF16_COMBINE_MIN_TOKENS:
            sums = sums.to(torch.bfloat16)
        sums = get_parallel().moe_ep_group.all_reduce(sums)
    if not combine:
        return RoutedOutput(sums, experts._identity[:tokens], 1)
    return sums.to(x.dtype)


def gate_and_route(gate, experts, x):
    """The gate's (weights, ids) and, for decode or MTP verify with up to
    ROUTE_FUSED_MAX_TOKENS tokens, the experts' route and fake-quantized x from the
    same launch (route_topk_metadata, bitwise the separate kernels); otherwise
    (weights, ids, None, None)."""
    if not (
        1 <= x.shape[0] <= ROUTE_FUSED_MAX_TOKENS
        and gate.topk == 8
        and uses_k_groups(x, gate.weight)
    ):
        return *gate(x), None, None
    num_local = experts.w13_weight.shape[0]
    block_m = experts_hip.default_config(8 * x.shape[0], experts.num_global_experts).bm
    x = x.contiguous()
    partials = linear_k_group_partials(x, gate.weight)
    nvfp4 = experts.w13_weight.dtype == torch.uint8
    weights, ids, meta, quantized = route_topk_metadata(
        partials,
        gate.bias,
        gate.route_scale,
        gate._one,
        experts.ep_rank * num_local,
        num_local,
        block_m,
        x,
        "nvfp4" if nvfp4 else "mxfp8",
        experts_hip.activation_dtype(experts.w13_weight.dtype),
    )
    _, dst2src, _, counts, offsets, _, schedule, _ = meta
    return (
        weights,
        ids,
        experts_hip.Route(dst2src, counts, offsets, block_m, schedule),
        quantized,
    )
