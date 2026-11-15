"""Fixed-shape expert-major/row-major conversion for the FlashInfer RDMA dispatcher.

Only valid received rows enter the expert compute path. The compact buffer's
capacity is EP * per-rank token capacity * top-k, independent of expert count.
All counts and maps stay on device so the conversions can be graph captured.
"""

import triton
import triton.language as tl


@triton.jit
def _pack(
    recv,
    recv_sf,
    counts,
    packed,
    packed_sf,
    ids,
    weights,
    slot_map,
    EXPERTS: tl.constexpr,
    EXPERT_OFFSET: tl.constexpr,
    GLOBAL_EXPERTS: tl.constexpr,
    CAP: tl.constexpr,
    WIDTH: tl.constexpr,
    SF_WIDTH: tl.constexpr,
    E_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
    S_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    es = tl.arange(0, E_BLOCK)
    ns = tl.load(counts + es, mask=es < EXPERTS, other=0)
    ends = tl.cumsum(ns, 0)
    # Empty experts have repeated end offsets and are skipped by this search.
    expert = tl.sum(((row >= ends) & (es < EXPERTS)).to(tl.int32), 0)
    begin = tl.sum(tl.where(es < expert, ns, 0), 0)
    valid = expert < EXPERTS
    slot = expert * CAP + row - begin
    xs = tl.arange(0, X_BLOCK)
    values = tl.load(recv + slot * WIDTH + xs, mask=valid & (xs < WIDTH), other=0)
    tl.store(packed + row * WIDTH + xs, values, mask=xs < WIDTH)
    if SF_WIDTH:
        ss = tl.arange(0, S_BLOCK)
        scales = tl.load(
            recv_sf + slot * SF_WIDTH + ss,
            mask=valid & (ss < SF_WIDTH),
            other=0,
        )
        tl.store(packed_sf + row * SF_WIDTH + ss, scales, mask=ss < SF_WIDTH)
    tl.store(ids + row, tl.where(valid, expert + EXPERT_OFFSET, GLOBAL_EXPERTS))
    tl.store(weights + row, tl.where(valid, 1.0, 0.0))
    tl.store(slot_map + row, tl.where(valid, slot, -1))


@triton.jit
def _unpack_and_scale(
    compact_output,
    slot_map,
    expert_output,
    global_scales,
    HIDDEN: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slot_map + row)
    xs = tl.arange(0, X_BLOCK)
    values = tl.load(
        compact_output + row * HIDDEN + xs,
        mask=(slot >= 0) & (xs < HIDDEN),
        other=0,
    )
    tl.store(
        expert_output + slot * HIDDEN + xs, values, mask=(slot >= 0) & (xs < HIDDEN)
    )
    amax = tl.max(tl.abs(values.to(tl.float32)), 0)
    # Match Torch's scalar/tensor division: round the reciprocal, then multiply.
    # Different scale bits can change FP4 codes at quantization thresholds.
    safe_amax = tl.maximum(amax, 1e-20)
    scale = tl.where(amax > 0, tl.div_rn(1.0, safe_amax) * 2688.0, 1.0)
    tl.store(global_scales + slot, scale, mask=slot >= 0)


def pack_expert_rows(
    recv,
    recv_sf,
    counts,
    packed,
    packed_sf,
    ids,
    weights,
    slot_map,
    *,
    ep_rank,
    num_experts,
):
    local_experts, cap, width = recv.shape
    sf_width = recv_sf.shape[-1]
    _pack[(packed.shape[0],)](
        recv,
        recv_sf,
        counts,
        packed,
        packed_sf,
        ids,
        weights,
        slot_map,
        EXPERTS=local_experts,
        EXPERT_OFFSET=ep_rank * local_experts,
        GLOBAL_EXPERTS=num_experts,
        CAP=cap,
        WIDTH=width,
        SF_WIDTH=sf_width,
        E_BLOCK=triton.next_power_of_2(local_experts),
        X_BLOCK=triton.next_power_of_2(width),
        S_BLOCK=triton.next_power_of_2(max(sf_width, 1)),
        num_warps=4,
    )


def unpack_expert_rows(compact_output, slot_map, expert_output, global_scales):
    _unpack_and_scale[(compact_output.shape[0],)](
        compact_output,
        slot_map,
        expert_output,
        global_scales,
        HIDDEN=compact_output.shape[1],
        X_BLOCK=triton.next_power_of_2(compact_output.shape[1]),
        num_warps=4,
    )
