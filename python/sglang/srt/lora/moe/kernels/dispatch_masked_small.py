"""Prepare masked rows, pair maps and both GEMM schedules in one launch.

Admission is limited to small BF16 batches with one token cluster per expert.
The kernel also supports FP8 quantization, but provider dispatch excludes it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
    OUTPUT_CLUSTER_SHIFT,
)
from sglang.srt.lora.moe.kernels.dispatch_checks import (
    check_fp8_width,
    check_source_rows,
)

_SMALL_PREPARE_MAX_PAIRS = 32
# The narrowest decode token tile: with at most this many tokens every expert
# fits one token cluster whatever tile the provider picks.
_SMALL_PREPARE_MAX_TOKENS = 8
# Pairs per program: the row-copy tile is [_PPP, BLOCK_H] with at most 8192 elements.
_PPP = 8


def _column_block(hidden: int, fp8: bool) -> int:
    """Column extent of the row-copy tile: a power of two (tl.arange), 128 to 1024
    columns; the copy loop masks the tail. FP8 rows are quantized in 128-wide
    groups whose scales are stored unmasked, so their block must also divide the
    width (FP8 widths are multiples of 128)."""
    cap = min(hidden, 8192 // _PPP)
    if not fp8:
        return max(128, triton.next_power_of_2(cap))
    block = 128
    while block * 2 <= cap and hidden % (block * 2) == 0:
        block *= 2
    return block


def small_masked_prepare_applies(
    num_tokens: int, num_pairs: int, *, fp8: bool = False
) -> bool:
    return (
        not fp8
        and 0 < num_pairs <= _SMALL_PREPARE_MAX_PAIRS
        and num_tokens <= _SMALL_PREPARE_MAX_TOKENS
    )


@triton.jit
def _small_masked_prepare_kernel(
    input_ptr,  # bf16 [num_tokens, hidden]
    rows_ptr,  # [E, m_max, hidden] bf16, or fp8 with scales
    scale_ptr,  # fp32 [E, m_max, hidden // 128] (FP8 only)
    topk_ids_ptr,  # int32 [num_tokens * TOPK]; < 0 = no expert
    pair_to_row_ptr,  # int32 [num_tokens * TOPK] out
    masked_m_ptr,  # int32 [E] out: the per-expert counts
    scratch_ptr,  # int32 [2 * E_P2]: the stage begins per expert
    schedule1_ptr,
    tiles1_ptr,
    schedule2_ptr,
    tiles2_ptr,
    num_pairs,
    m_max,
    hidden,
    token_width,
    out_clusters1,
    out_clusters2,
    E: tl.constexpr,
    E_P2: tl.constexpr,
    PAIRS: tl.constexpr,
    PPP: tl.constexpr,  # pairs whose rows one program moves
    TOPK: tl.constexpr,
    OC_P2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GB: tl.constexpr,  # 128-groups per chunk (FP8)
    FP8: tl.constexpr,
    GROUPS: tl.constexpr,  # 128-groups per row (FP8)
    OUTPUT_SHIFT: tl.constexpr,
):
    # A pair's slot in its expert is its rank among the expert's pairs (stable
    # by pair id), so every program derives the same placement without atomics.
    pid = tl.program_id(0)
    p = tl.arange(0, PAIRS)
    pmask = p < num_pairs
    expert = tl.load(topk_ids_ptr + p, mask=pmask, other=-1)
    live = pmask & (expert >= 0)
    if pid == 0:
        same = (expert[None, :] == expert[:, None]) & live[None, :] & live[:, None]
        rank = tl.sum((same & (p[None, :] < p[:, None])).to(tl.int32), axis=1)
        count = tl.sum(same.to(tl.int32), axis=1)
        head = live & (rank == 0)
        e_safe = tl.where(live, expert, 0)
        dst = e_safe.to(tl.int64) * m_max + rank
        bins = tl.arange(0, E_P2)
        tl.store(masked_m_ptr + bins, tl.zeros([E_P2], dtype=tl.int32), mask=bins < E)
        tl.debug_barrier()
        tl.store(masked_m_ptr + e_safe, count, mask=head)
        tl.debug_barrier()
        counts = tl.load(
            masked_m_ptr + bins, mask=bins < E, other=0, cache_modifier=".cg"
        )
        clusters = (counts + token_width - 1) // token_width  # one per touched expert
        ent1 = clusters * out_clusters1
        ent2 = clusters * out_clusters2
        tl.store(scratch_ptr + bins, tl.cumsum(ent1, axis=0) - ent1)
        tl.store(scratch_ptr + E_P2 + bins, tl.cumsum(ent2, axis=0) - ent2)
        tl.store(tiles1_ptr, tl.sum(ent1, axis=0))
        tl.store(tiles2_ptr, tl.sum(ent2, axis=0))
        tl.debug_barrier()
        tl.store(pair_to_row_ptr + p, tl.where(live, dst, -1).to(tl.int32), mask=pmask)
        # The first row of each touched expert writes the expert's entries:
        # one token cluster (index 0) times the stage's output clusters.
        begin1 = tl.load(scratch_ptr + e_safe, cache_modifier=".cg")
        begin2 = tl.load(scratch_ptr + E_P2 + e_safe, cache_modifier=".cg")
        oc = tl.arange(0, OC_P2)
        packed = (
            expert.to(tl.int64)[:, None] | (oc.to(tl.int64) << OUTPUT_SHIFT)[None, :]
        )
        tl.store(
            schedule1_ptr + begin1[:, None] + oc[None, :],
            packed,
            mask=head[:, None] & (oc < out_clusters1)[None, :],
        )
        tl.store(
            schedule2_ptr + begin2[:, None] + oc[None, :],
            packed,
            mask=head[:, None] & (oc < out_clusters2)[None, :],
        )
    # Rows: this program moves pairs [pid * PPP, (pid + 1) * PPP) as one
    # [PPP, BLOCK_H] tile per column chunk.
    q = pid * PPP + tl.arange(0, PPP)
    qmask = q < num_pairs
    my_expert = tl.load(topk_ids_ptr + q, mask=qmask, other=-1)
    my_live = qmask & (my_expert >= 0)
    my_rank = tl.sum(
        (
            (expert[None, :] == my_expert[:, None])
            & live[None, :]
            & (p[None, :] < q[:, None])
        ).to(tl.int32),
        axis=1,
    )
    my_dst = tl.where(my_live, my_expert, 0).to(tl.int64) * m_max + my_rank
    src = input_ptr + (q // TOPK).to(tl.int64)[:, None] * hidden
    out = rows_ptr + my_dst[:, None] * hidden
    for off in tl.range(0, hidden, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        m = my_live[:, None] & (cols < hidden)[None, :]
        if FP8:
            # Per 128-group scales, the group quantizer's formula (fp8_quant.py).
            x = tl.reshape(
                tl.load(src + cols[None, :], mask=m, other=0.0).to(tl.float32),
                (PPP, GB, 128),
            )
            amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-10)
            qv = tl.clamp(x * (448.0 / amax)[:, :, None], -448.0, 448.0)
            tl.store(
                out + cols[None, :],
                tl.reshape(qv, (PPP, BLOCK_H)).to(rows_ptr.dtype.element_ty),
                mask=m,
            )
            gcols = off // 128 + tl.arange(0, GB)
            tl.store(
                scale_ptr + my_dst[:, None] * GROUPS + gcols[None, :],
                amax * (1.0 / 448.0),
                mask=my_live[:, None],
            )
        else:
            tl.store(out + cols[None, :], tl.load(src + cols[None, :], mask=m), mask=m)


def small_masked_prepare(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    rows_out: torch.Tensor,
    scale_out: torch.Tensor | None,
    scratch: torch.Tensor,
    token_width: int,
    out_clusters1: int,
    out_clusters2: int,
    schedule1_out: torch.Tensor,
    tiles1_out: torch.Tensor,
    schedule2_out: torch.Tensor,
    tiles2_out: torch.Tensor,
) -> None:
    check_source_rows(hidden_states, topk_ids, top_k)
    num_tokens, hidden = hidden_states.shape
    num_pairs = topk_ids.numel()
    num_experts = masked_m_out.numel()
    fp8 = scale_out is not None
    if fp8:
        check_fp8_width(hidden)
    e_p2 = max(16, triton.next_power_of_2(num_experts))
    if scratch.numel() < 2 * e_p2:
        raise ValueError("the small prepare needs a 2 x E_P2 int32 scratch")
    pairs = max(16, triton.next_power_of_2(num_pairs))
    ppp = _PPP
    block_h = _column_block(hidden, fp8)
    _small_masked_prepare_kernel[(triton.cdiv(num_pairs, ppp),)](
        hidden_states,
        rows_out,
        scale_out if fp8 else rows_out,
        topk_ids.view(-1),
        pair_to_row_out,
        masked_m_out,
        scratch,
        schedule1_out,
        tiles1_out,
        schedule2_out,
        tiles2_out,
        num_pairs,
        rows_out.size(1),
        hidden,
        token_width,
        out_clusters1,
        out_clusters2,
        E=num_experts,
        E_P2=e_p2,
        PAIRS=pairs,
        PPP=ppp,
        TOPK=top_k,
        OC_P2=max(2, triton.next_power_of_2(max(out_clusters1, out_clusters2))),
        BLOCK_H=block_h,
        GB=block_h // 128 if fp8 else 1,
        FP8=fp8,
        GROUPS=hidden // 128 if fp8 else 1,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
        num_warps=8,
    )
