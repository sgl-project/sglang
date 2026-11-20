"""Fixed-shape expert-major/row-major conversion for the FlashInfer RDMA dispatcher.

Only valid received rows enter the expert compute path. The compact buffer's
capacity is EP * per-rank token capacity * top-k, independent of expert count.
All counts and maps stay on device so the conversions can be graph captured.

Received rows fill a prefix of the compact buffer. Both kernels launch a fixed
number of programs and read that prefix length from the device counts, so their
cost follows the rows that carry data rather than the buffer size.
"""

import triton
import triton.language as tl

# Upper limit of programs per launch; program p walks rows p, p + step, ... where
# step is the number of programs launched.
_GRID = 1024


@triton.jit(do_not_specialize=["rows", "step"])
def _pack(
    recv,
    recv_sf,
    counts,
    packed,
    packed_sf,
    ids,
    weights,
    slot_map,
    rows,
    step,
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
    es = tl.arange(0, E_BLOCK)
    ns = tl.load(counts + es, mask=es < EXPERTS, other=0)
    ends = tl.cumsum(ns, 0)
    # Rows beyond the buffer are dropped, as when one program ran per row.
    total = tl.minimum(tl.sum(ns, 0), rows)
    xs = tl.arange(0, X_BLOCK)
    ss = tl.arange(0, S_BLOCK)
    row = tl.program_id(0)
    while row < total:
        # Empty experts have repeated end offsets and are skipped by this search.
        expert = tl.sum(((row >= ends) & (es < EXPERTS)).to(tl.int32), 0)
        begin = tl.sum(tl.where(es < expert, ns, 0), 0)
        slot = expert * CAP + row - begin
        values = tl.load(recv + slot * WIDTH + xs, mask=xs < WIDTH, other=0)
        tl.store(packed + row * WIDTH + xs, values, mask=xs < WIDTH)
        if SF_WIDTH:
            scales = tl.load(
                recv_sf + slot * SF_WIDTH + ss, mask=ss < SF_WIDTH, other=0
            )
            tl.store(packed_sf + row * SF_WIDTH + ss, scales, mask=ss < SF_WIDTH)
        tl.store(ids + row, expert + EXPERT_OFFSET)
        tl.store(weights + row, 1.0)
        tl.store(slot_map + row, slot)
        row += step
    # Rows without data keep whatever payload the buffer held: their id is out
    # of range, so expert compute never reads them and unpack never visits them.
    while row < rows:
        tl.store(ids + row, GLOBAL_EXPERTS)
        tl.store(weights + row, 0.0)
        tl.store(slot_map + row, -1)
        row += step


@triton.jit(do_not_specialize=["rows", "step"])
def _unpack_and_scale(
    compact_output,
    slot_map,
    counts,
    expert_output,
    global_scales,
    rows,
    step,
    HIDDEN: tl.constexpr,
    X_BLOCK: tl.constexpr,
    EXPERTS: tl.constexpr,
    E_BLOCK: tl.constexpr,
):
    es = tl.arange(0, E_BLOCK)
    ns = tl.load(counts + es, mask=es < EXPERTS, other=0)
    total = tl.minimum(tl.sum(ns, 0), rows)
    xs = tl.arange(0, X_BLOCK)
    row = tl.program_id(0)
    while row < total:
        slot = tl.load(slot_map + row)
        values = tl.load(
            compact_output + row * HIDDEN + xs,
            mask=(slot >= 0) & (xs < HIDDEN),
            other=0,
        )
        tl.store(
            expert_output + slot * HIDDEN + xs,
            values,
            mask=(slot >= 0) & (xs < HIDDEN),
        )
        amax = tl.max(tl.abs(values.to(tl.float32)), 0)
        # Match Torch's scalar/tensor division: round the reciprocal, then multiply.
        # Different scale bits can change FP4 codes at quantization thresholds.
        safe_amax = tl.maximum(amax, 1e-20)
        scale = tl.where(amax > 0, tl.div_rn(1.0, safe_amax) * 2688.0, 1.0)
        tl.store(global_scales + slot, scale, mask=slot >= 0)
        row += step


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
    grid = min(_GRID, packed.shape[0])
    _pack[(grid,)](
        recv,
        recv_sf,
        counts,
        packed,
        packed_sf,
        ids,
        weights,
        slot_map,
        packed.shape[0],
        grid,
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


def unpack_expert_rows(compact_output, slot_map, counts, expert_output, global_scales):
    local_experts = counts.shape[0]
    grid = min(_GRID, compact_output.shape[0])
    _unpack_and_scale[(grid,)](
        compact_output,
        slot_map,
        counts,
        expert_output,
        global_scales,
        compact_output.shape[0],
        grid,
        HIDDEN=compact_output.shape[1],
        X_BLOCK=triton.next_power_of_2(compact_output.shape[1]),
        EXPERTS=local_experts,
        E_BLOCK=triton.next_power_of_2(local_experts),
        num_warps=4,
    )
