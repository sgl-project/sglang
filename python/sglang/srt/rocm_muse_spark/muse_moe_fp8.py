# ruff: noqa: C408
"""Muse Spark routed experts in Quark PTPC-FP8 on ROCm, composed from existing kernels.

Weights are FP8 e4m3 with one static fp32 scale per output channel; activations
are quantized to FP8 per token at run time. Muse applies a per-expert RMSNorm
before the weighted top-k sum, so both GEMMs keep one output row per routed slot
and the existing Muse finalize does the norm and the combine:

1. slot sort by *local* expert (Muse ``moe_align_block_size``; non-local slots are
   dropped or labelled -1 and never computed),
2. GEMM1 ``w13`` [E, 2I, K] via the public ``invoke_fused_moe_kernel``
   (per-token activation quant inside, per-channel weight scales, no routing
   weight, one output row per slot),
3. ``silu_and_mul``,
4. GEMM2 ``w2`` [E, H, I], same flags with ``top_k=1`` (A row = slot),
5. ``finalize_norm_weighted_sum_kernel`` (masks non-local slots, applies the
   post-expert norm per slot, then the weighted sum).

Every step is shape-static and host-sync free, so the path captures in HIP graphs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.moe import fused_moe_triton_kernels
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.rocm_muse_spark.muse_moe_triton import (
    finalize_norm_weighted_sum_kernel,
    moe_align_block_size,
)

# MI350X (gfx950) tile sweep 2026-11-02 at EP8 shapes (32 local experts,
# K = I = H = 4096, top-8), graph replay. Buckets by routed rows = M * top_k; one
# BLOCK_SIZE_M per bucket because both GEMMs share the slot alignment, "down"
# overrides the w2 tile. Tiles change only the accumulation schedule.
_ROW_BOUNDS = (64, 256, 512, 4096)
_CONFIGS = (
    dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=2,
    ),
    dict(
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=4,
        num_warps=4,
        num_stages=2,
    ),
    dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=256,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2,
        down=dict(BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=8, num_warps=8),
    ),
    dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=256,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=2,
        down=dict(BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=8, num_warps=8),
    ),
    dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=2,
    ),
)


def pick_configs(rows: int) -> tuple[dict, dict]:
    """(GEMM1 config, GEMM2 config) for ``rows`` routed slots."""
    index = next((i for i, b in enumerate(_ROW_BOUNDS) if rows <= b), len(_ROW_BOUNDS))
    up = {k: v for k, v in _CONFIGS[index].items() if k != "down"}
    down = {**up, **_CONFIGS[index].get("down", {})}
    return up, down


def check_moe_fp8_supported() -> None:
    """The serialized weights are OCP e4m3fn (gfx950); e4m3fnuz GPUs would need a
    conversion this path does not do. Without aiter the generic wrapper subtracts
    the MoE padding from K (``SGLANG_MOE_PADDING``); Muse expert weights are
    unpadded, so K would shrink."""
    if is_fp8_fnuz():
        raise ValueError("Quark PTPC-FP8 Muse experts need OCP FP8 e4m3fn (gfx950)")
    wrapper = fused_moe_triton_kernels
    padded = 0 if wrapper._use_aiter else wrapper.padding_size
    if padded:
        raise ValueError(
            "Quark PTPC-FP8 Muse experts are not K-padded; unset SGLANG_MOE_PADDING "
            "or run with SGLANG_USE_AITER=1"
        )


def moe_fp8_gemm(a, w, w_scale, out, scores, ids, align, top_k: int, config: dict):
    """One slot-sorted PTPC GEMM: ``out[slot] = q(a[slot // top_k]) @ (w[e] * w_scale[e]).T`` for local slots."""
    sorted_ids, expert_ids, num_post = align
    invoke_fused_moe_kernel(
        a,
        w,
        None,
        out,
        None,
        w_scale,
        None,
        scores,
        ids,
        sorted_ids,
        expert_ids,
        num_post,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        c_sorted=False,
    )


def muse_moe_fp8(
    hidden_states: torch.Tensor,  # [M, K] bf16, pooled + pre-expert-normed
    topk_scores: torch.Tensor,  # [M, top_k] routing weights
    topk_ids: torch.Tensor,  # [M, top_k] global expert ids
    w13: torch.Tensor,  # [E_local, 2I, K] float8_e4m3fn
    w13_scale: torch.Tensor,  # [E_local, 2I] fp32
    w2: torch.Tensor,  # [E_local, H, I] float8_e4m3fn
    w2_scale: torch.Tensor,  # [E_local, H] fp32
    local_expert_start: int,
    post_norm_weight: torch.Tensor | None,
    eps: float,
    post_norm_gain: float = 0.0,
    has_norm: bool = True,
) -> torch.Tensor:
    """This rank's per-token sum over its local experts' normed outputs, [M, H]."""
    M, K = hidden_states.shape
    top_k = topk_ids.shape[1]
    E, two_i, kw = w13.shape
    inter_size = two_i // 2
    _, H, iw = w2.shape
    assert hidden_states.dtype == torch.bfloat16, hidden_states.dtype
    assert w13.dtype == w2.dtype == torch.float8_e4m3fn, (w13.dtype, w2.dtype)
    assert kw == K and iw == inter_size, (tuple(w13.shape), tuple(w2.shape), K)
    assert w13_scale.shape == (E, two_i) and w2_scale.shape == (E, H)
    assert w13_scale.dtype == w2_scale.dtype == torch.float32
    assert topk_scores.shape == topk_ids.shape and topk_ids.shape[0] == M
    if M == 0:
        return hidden_states.new_empty((0, H))
    hidden_states = hidden_states.contiguous()
    num_slots = M * top_k
    up, down = pick_configs(num_slots)
    ids32 = topk_ids.to(torch.int32).contiguous()
    scores32 = topk_scores.to(torch.float32).contiguous()
    align = moe_align_block_size(ids32, E, local_expert_start, up["BLOCK_SIZE_M"])
    dev = hidden_states.device

    gate_up = torch.empty((num_slots, two_i), dtype=torch.bfloat16, device=dev)
    moe_fp8_gemm(
        hidden_states, w13, w13_scale, gate_up, scores32, ids32, align, top_k, up
    )
    inter = silu_and_mul(gate_up)
    y = torch.empty((num_slots, H), dtype=torch.bfloat16, device=dev)
    moe_fp8_gemm(inter, w2, w2_scale, y, scores32, ids32, align, 1, down)

    return finalize_slots(
        y,
        scores32,
        ids32,
        local_expert_start,
        E,
        post_norm_weight,
        eps,
        post_norm_gain,
        has_norm,
        hidden_states.dtype,
    )


def finalize_slots(
    y: torch.Tensor,  # [M * top_k, H] per-slot expert outputs; rows of non-local slots are never read
    scores32: torch.Tensor,  # [M, top_k] fp32
    ids32: torch.Tensor,  # [M, top_k] int32 global expert ids
    local_expert_start: int,
    num_local: int,
    post_norm_weight: torch.Tensor | None,
    eps: float,
    post_norm_gain: float,
    has_norm: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Muse finalize of this rank's slots: per-slot post-expert norm, then the score-weighted sum, [M, H]."""
    M, top_k = ids32.shape
    H = y.shape[1]
    out = torch.empty((M, H), dtype=dtype, device=y.device)
    has_scale = has_norm and post_norm_weight is not None
    finalize_norm_weighted_sum_kernel[(M,)](
        y,
        post_norm_weight if has_scale else out,  # unused pointer without a scale
        scores32,
        ids32,
        out,
        H,
        local_expert_start,
        num_local,
        eps,
        float(post_norm_gain),
        0,
        TOP_K=top_k,
        HAS_NORM=has_norm,
        HAS_SCALE=has_scale,
        BLOCK_H=triton.next_power_of_2(H),
        SPLIT_K=1,
        num_warps=4,
    )
    return out
