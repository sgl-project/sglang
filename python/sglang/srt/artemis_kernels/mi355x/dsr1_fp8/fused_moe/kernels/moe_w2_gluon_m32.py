"""Bit-exact no-atomic MI355X DSR1 TP8 FP8 W2 for concurrency 32.

The selected native-M16 producer writes each expert result to its unique
``contribution[token, route_slot, hidden]`` location. A fixed M8xN128 reducer
then adds slots 0 through 8 in order, rounding to BF16 after every add. Under
the DSR1 contract of eight unique routed experts plus shared expert 256, every
contribution element is overwritten and the workspace needs no clear launch.
"""

from __future__ import annotations

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

M = gl.constexpr(32)
H = gl.constexpr(7168)
I = gl.constexpr(256)
E = gl.constexpr(257)
TOPK = gl.constexpr(9)
M16 = gl.constexpr(16)
BLOCK_N = gl.constexpr(128)
BLOCK_K = gl.constexpr(128)
OUTPUT_SPLITS = gl.constexpr(28)
NUM_WARPS = gl.constexpr(4)
REDUCE_M = gl.constexpr(8)
REDUCE_N = gl.constexpr(128)

@gluon.jit
def _load_preshuffled_w2(
    w2_ptr,
    expert_id,
    output_tile,
    k_block,
    packed_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
):
    packed_k = gl.arange(0, BLOCK_K * 16, layout=gl.SliceLayout(0, packed_layout))
    packed_n = gl.arange(
        0,
        BLOCK_N // 16,
        layout=gl.SliceLayout(1, packed_layout),
    )
    offsets = (
        expert_id * (H * I)
        + (output_tile * (BLOCK_N // 16) + packed_n[:, None]) * (I * 16)
        + k_block * (BLOCK_K * 16)
        + packed_k[None, :]
    )
    packed = gl.amd.cdna4.buffer_load(w2_ptr, offsets, cache=".cg")
    logical = (
        packed.reshape(1, BLOCK_N // 16, BLOCK_K // 32, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_N, BLOCK_K)
        .trans(1, 0)
    )
    return gl.convert_layout(logical, dot_b_layout, assert_trivial=True)


@gluon.jit
def _load_w2_pair(
    w2_ptr,
    w2_scale_ptr,
    expert_id,
    output_tile,
    packed_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
):
    weights0 = _load_preshuffled_w2(
        w2_ptr,
        expert_id,
        output_tile,
        0,
        packed_layout,
        dot_b_layout,
    )
    weights1 = _load_preshuffled_w2(
        w2_ptr,
        expert_id,
        output_tile,
        1,
        packed_layout,
        dot_b_layout,
    )
    scale_base = expert_id * 56 * 2 + output_tile * 2
    scale0 = gl.load(w2_scale_ptr + scale_base)
    scale1 = gl.load(w2_scale_ptr + scale_base + 1)
    return weights0, weights1, scale0, scale1


@gluon.jit
def _load_m16_chunk(
    intermediate_ptr,
    intermediate_scale_ptr,
    route_tokens_ptr,
    route_weights_ptr,
    expert_id,
    route_count,
    row_base,
    dot_a_layout: gl.constexpr,
    row_layout: gl.constexpr,
):
    a_rows = gl.arange(0, M16, layout=gl.SliceLayout(1, dot_a_layout))
    a_cols = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a_layout))
    logical_a_rows = row_base + a_rows
    valid_a = logical_a_rows[:, None] < route_count
    workspace_base = expert_id * 2 * M * BLOCK_K
    offsets = logical_a_rows[:, None] * BLOCK_K + a_cols[None, :]
    inter0 = gl.amd.cdna4.buffer_load(
        intermediate_ptr,
        workspace_base + offsets,
        mask=valid_a,
        other=0.0,
    )
    inter1 = gl.amd.cdna4.buffer_load(
        intermediate_ptr,
        workspace_base + M * BLOCK_K + offsets,
        mask=valid_a,
        other=0.0,
    )
    rows = gl.arange(0, M16, layout=row_layout)
    logical_rows = row_base + rows
    valid_rows = logical_rows < route_count
    route_base = expert_id * M
    tokens = gl.amd.cdna4.buffer_load(
        route_tokens_ptr,
        route_base + logical_rows,
        mask=valid_rows,
        other=M,
    )
    route_weights = gl.amd.cdna4.buffer_load(
        route_weights_ptr,
        route_base + logical_rows,
        mask=valid_rows,
        other=0.0,
    )
    scale_base = expert_id * 2 * M
    scale0 = gl.amd.cdna4.buffer_load(
        intermediate_scale_ptr,
        scale_base + logical_rows,
        mask=valid_rows,
        other=0.0,
    )
    scale1 = gl.amd.cdna4.buffer_load(
        intermediate_scale_ptr,
        scale_base + M + logical_rows,
        mask=valid_rows,
        other=0.0,
    )
    return inter0, inter1, scale0, scale1, route_weights, tokens, valid_rows


@gluon.jit
def _load_route_slots_m32_exact(
    route_slots_ptr,
    expert_id,
    route_count,
    row_base,
    row_layout: gl.constexpr,
):
    rows = gl.arange(0, M16, layout=row_layout)
    logical_rows = row_base + rows
    return gl.amd.cdna4.buffer_load(
        route_slots_ptr,
        expert_id * M + logical_rows,
        mask=logical_rows < route_count,
        other=TOPK,
    )


@gluon.jit
def _store_contribution_m32_exact(
    contribution_ptr,
    weights,
    inputs,
    slots,
    output_tile,
    output_n,
    zero,
):
    weights0, weights1, weight_scale0, weight_scale1 = weights
    (
        inter0,
        inter1,
        inter_scale0,
        inter_scale1,
        route_weights,
        tokens,
        valid_rows,
    ) = inputs
    down0 = gl.amd.cdna4.mfma_scaled(
        inter0,
        None,
        "e4m3",
        weights0,
        None,
        "e4m3",
        zero,
    )
    down1 = gl.amd.cdna4.mfma_scaled(
        inter1,
        None,
        "e4m3",
        weights1,
        None,
        "e4m3",
        zero,
    )
    down = down0 * inter_scale0[:, None] * weight_scale0
    down += down1 * inter_scale1[:, None] * weight_scale1
    down *= route_weights[:, None]
    matched = valid_rows & (slots >= 0) & (slots < TOPK)
    offsets = (
        (tokens[:, None] * TOPK + slots[:, None]) * H + output_tile * BLOCK_N + output_n[None, :]
    )
    gl.amd.cdna4.buffer_store(
        down.to(contribution_ptr.type.element_ty),
        contribution_ptr,
        offsets,
        mask=matched[:, None],
    )


@gluon.jit
def _store_two_contributions_m32_exact(
    contribution_ptr,
    weight0,
    weight1,
    inputs,
    slots,
    output_base,
    output_n,
    zero,
):
    _store_contribution_m32_exact(
        contribution_ptr,
        weight0,
        inputs,
        slots,
        output_base,
        output_n,
        zero,
    )
    _store_contribution_m32_exact(
        contribution_ptr,
        weight1,
        inputs,
        slots,
        output_base + 1,
        output_n,
        zero,
    )


@gluon.jit
def _moe_w2_m32_exact_contribution_kernel(
    contribution_ptr,
    route_slots_ptr,
    w2_ptr,
    w2_scale_ptr,
    intermediate_ptr,
    intermediate_scale_ptr,
    route_tokens_ptr,
    route_weights_ptr,
    route_counts_ptr,
):
    pid = gl.program_id(0)
    expert_index = pid // OUTPUT_SPLITS
    raw_split = pid % OUTPUT_SPLITS
    output_split = (raw_split * 9) % OUTPUT_SPLITS
    expert_id = gl.where(expert_index == 0, E - 1, expert_index - 1)
    route_count = gl.load(route_counts_ptr + expert_id)
    if route_count == 0:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 16)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 16)
    row_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    packed_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 1024], [4, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [0, 512]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N // 16, BLOCK_K * 16],
    )
    output_base = output_split * 2
    output_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    zero = gl.zeros((M16, BLOCK_N), gl.float32, layout=mfma_layout)
    weight0 = _load_w2_pair(
        w2_ptr,
        w2_scale_ptr,
        expert_id,
        output_base,
        packed_layout,
        dot_b_layout,
    )
    weight1 = _load_w2_pair(
        w2_ptr,
        w2_scale_ptr,
        expert_id,
        output_base + 1,
        packed_layout,
        dot_b_layout,
    )
    inputs0 = _load_m16_chunk(
        intermediate_ptr,
        intermediate_scale_ptr,
        route_tokens_ptr,
        route_weights_ptr,
        expert_id,
        route_count,
        0,
        dot_a_layout,
        row_layout,
    )
    slots0 = _load_route_slots_m32_exact(
        route_slots_ptr,
        expert_id,
        route_count,
        0,
        row_layout=row_layout,
    )
    _store_two_contributions_m32_exact(
        contribution_ptr,
        weight0,
        weight1,
        inputs0,
        slots0,
        output_base,
        output_n,
        zero,
    )
    if route_count > M16:
        inputs1 = _load_m16_chunk(
            intermediate_ptr,
            intermediate_scale_ptr,
            route_tokens_ptr,
            route_weights_ptr,
            expert_id,
            route_count,
            M16,
            dot_a_layout,
            row_layout,
        )
        slots1 = _load_route_slots_m32_exact(
            route_slots_ptr,
            expert_id,
            route_count,
            M16,
            row_layout=row_layout,
        )
        _store_two_contributions_m32_exact(
            contribution_ptr,
            weight0,
            weight1,
            inputs1,
            slots1,
            output_base,
            output_n,
            zero,
        )


@gluon.jit
def _moe_w2_m32_exact_reduce_kernel(out_ptr, contribution_ptr):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    rows = gl.arange(0, REDUCE_M, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, REDUCE_N, layout=gl.SliceLayout(0, layout))
    pid = gl.program_id(0)
    token_base = (pid // (H // REDUCE_N)) * REDUCE_M
    output_base = (pid % (H // REDUCE_N)) * REDUCE_N
    tokens = token_base + rows
    output_cols = output_base + cols

    accumulator = gl.zeros((REDUCE_M, REDUCE_N), gl.float32, layout=layout)
    for route_slot in gl.static_range(TOPK):
        offsets = (tokens[:, None] * TOPK + route_slot) * H + output_cols[None, :]
        contribution = gl.amd.cdna4.buffer_load(contribution_ptr, offsets)
        accumulator = (accumulator + contribution.to(gl.float32)).to(gl.bfloat16).to(gl.float32)

    output_offsets = tokens[:, None] * H + output_cols[None, :]
    gl.amd.cdna4.buffer_store(
        accumulator.to(out_ptr.type.element_ty),
        out_ptr,
        output_offsets,
    )


def moe_w2_gluon_m32(
    out: torch.Tensor,
    contribution: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    intermediate: torch.Tensor,
    intermediate_scale: torch.Tensor,
    route_tokens: torch.Tensor,
    route_weights: torch.Tensor,
    route_counts: torch.Tensor,
    route_slots: torch.Tensor,
) -> torch.Tensor:
    """Run direct contribution stores and fixed slot-order BF16 reduction."""
    _moe_w2_m32_exact_contribution_kernel[(int(E) * int(OUTPUT_SPLITS),)](
        contribution,
        route_slots,
        w2,
        w2_scale,
        intermediate,
        intermediate_scale,
        route_tokens,
        route_weights,
        route_counts,
        num_warps=4,
        waves_per_eu=2,
    )
    _moe_w2_m32_exact_reduce_kernel[
        ((int(M) // int(REDUCE_M)) * (int(H) // int(REDUCE_N)),)
    ](
        out,
        contribution,
        num_warps=4,
        waves_per_eu=3,
    )
    return out


__all__ = ["moe_w2_gluon_m32"]
