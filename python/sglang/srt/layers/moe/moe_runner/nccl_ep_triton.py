"""Explicit NCCL EP LL compatibility path using existing Triton FP8 GEMMs.

The LL receive layout contains fixed-capacity expert slots. Only valid slots
are dequantized; routing weights remain owned by NCCL combine. This path adds
dequantization/requantization and capacity-sized scratch, not an optimized
masked GEMM implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_expert_slots(
    received,
    scales,
    counts,
    output,
    ids,
    weights,
    H: tl.constexpr,
    CAPACITY: tl.constexpr,
    X_E: tl.constexpr,
    X_M: tl.constexpr,
    X_H: tl.constexpr,
    S_E: tl.constexpr,
    S_M: tl.constexpr,
    S_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    expert = slot // CAPACITY
    row = slot % CAPACITY
    valid = row < tl.load(counts + expert)
    h = tl.arange(0, BLOCK)
    x = tl.load(
        received + expert * X_E + row * X_M + h * X_H,
        mask=valid & (h < H),
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scales + expert * S_E + row * S_M + (h // 128) * S_H,
        mask=valid & (h < H),
        other=0.0,
    )
    tl.store(output + slot * H + h, x * scale, mask=h < H)
    tl.store(ids + slot, tl.where(valid, expert, -1))
    tl.store(weights + slot, 1.0)


def prepare_expert_slots(received, scales, counts):
    """Convert LL slots without reading invalid data or uninitialized scales."""
    experts, capacity, hidden = received.shape
    if (
        received.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float32
        or hidden % 128
        or scales.shape != (experts, capacity, hidden // 128)
        or counts.shape != (experts,)
        or not counts.is_contiguous()
    ):
        raise ValueError(
            "NCCL EP Triton requires FP8 slots and float32 group-128 scales"
        )
    output = torch.empty(
        (experts * capacity, hidden), dtype=torch.bfloat16, device=received.device
    )
    ids = torch.empty(
        (experts * capacity, 1), dtype=torch.int32, device=received.device
    )
    weights = torch.empty_like(ids, dtype=torch.float32)
    _prepare_expert_slots[(experts * capacity,)](
        received,
        scales,
        counts,
        output,
        ids,
        weights,
        hidden,
        capacity,
        *received.stride(),
        *scales.stride(),
        BLOCK=triton.next_power_of_2(hidden),
    )
    return output, ids, weights


def run_nccl_ep_triton(dispatch_output, quant_info, config):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    if (
        not quant_info.use_fp8_w8a8
        or quant_info.use_mxfp8
        or quant_info.block_shape != [128, 128]
        or config.activation != "silu"
        or not config.is_gated
        or config.num_fused_shared_experts
        or config.apply_router_weight_on_input
    ):
        raise ValueError(
            "NCCL EP Triton supports block-128 FP8 gated SiLU without fused shared "
            "experts or router weights on input"
        )
    received, scales, original_ids, original_weights, counts, _ = dispatch_output
    hidden, ids, weights = prepare_expert_slots(received, scales, counts)
    output = fused_experts_impl(
        hidden,
        quant_info.w13_weight,
        quant_info.w2_weight,
        weights,
        ids,
        b1=quant_info.b13,
        b2=quant_info.b2,
        inplace=False,
        activation="silu",
        is_gated=True,
        use_fp8_w8a8=True,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        block_shape=[128, 128],
        no_combine=True,
        filter_expert=True,
        apply_router_weight_on_input=False,
        gemm1_alpha=config.gemm1_alpha,
        gemm1_limit=config.gemm1_clamp_limit,
        swiglu_limit=config.swiglu_limit,
        gate_up_interleaved=config.gate_up_interleaved,
    )
    return DeepEPLLCombineInput(
        hidden_states=output.view(received.shape),
        topk_ids=original_ids,
        topk_weights=original_weights,
    )
