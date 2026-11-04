"""DSA indexer top-k under DCP on ROCm: the exact merge of ``dsa_dcp_sharded``
over the group all-gather, without multicast or the v2 plan."""

from functools import partial
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.dcp_topk_hip import (
    dcp_global_slots_hip,
    dcp_local_candidates_hip,
    supports_dcp_topk_hip,
)
from sglang.srt.runtime_context import get_exec, get_parallel


@triton.jit
def _pack_candidates_kernel(
    logits,
    logits_stride,
    local_idx,
    out,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    TOPK: tl.constexpr,
):
    # local top-k positions -> (fp32 score bits, global position or -1) pairs
    row = tl.program_id(0)
    offs = tl.arange(0, TOPK)
    i = tl.load(local_idx + row.to(tl.int64) * TOPK + offs)
    valid = i >= 0
    score = tl.load(
        logits + row.to(tl.int64) * logits_stride + tl.where(valid, i, 0),
        mask=valid,
        other=0.0,
    )
    base = out + row.to(tl.int64) * TOPK * 2
    tl.store(base + offs * 2, score.to(tl.int32, bitcast=True))
    tl.store(base + offs * 2 + 1, tl.where(valid, i * WORLD + RANK, -1))


@triton.jit
def _gather_candidates_kernel(
    gathered,
    num_rows,
    global_lens,
    scores,
    positions,
    lengths,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Rank-major [WORLD * rows, TOPK] (score bits, position) pairs -> row-major
    # [rows, WORLD * TOPK] scores (-inf where none) and positions (-1 where none).
    src_row = tl.program_id(0)
    rank = src_row // num_rows
    row = src_row % num_rows
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_block = offs < TOPK
    # Prefix-only gathers leave the padding undefined; read this rank's share only.
    n = (tl.load(global_lens + row) + WORLD - 1 - rank) // WORLD
    mask = offs < tl.minimum(n, TOPK)
    base = gathered + src_row.to(tl.int64) * TOPK * 2
    score = tl.load(base + offs * 2, mask=mask, other=0).to(tl.float32, bitcast=True)
    pos = tl.load(base + offs * 2 + 1, mask=mask, other=-1)
    keep = mask & (pos >= 0)
    dst = row.to(tl.int64) * WORLD * TOPK + rank * TOPK + offs
    tl.store(scores + dst, tl.where(keep, score, float("-inf")), mask=in_block)
    tl.store(positions + dst, tl.where(keep, pos, -1), mask=in_block)
    if rank == 0 and tl.program_id(1) == 0:
        tl.store(lengths + row, WORLD * TOPK)


@triton.jit
def _candidates_to_slots_kernel(
    idx,
    positions,
    page_table,
    page_table_stride,
    out,
    span,
    NUM_CANDIDATES: tl.constexpr,
    TOPK: tl.constexpr,
):
    # selected candidate -> its global position p -> widened slot, -1 if none
    row = tl.program_id(0)
    offs = tl.arange(0, TOPK)
    i = tl.load(idx + row.to(tl.int64) * TOPK + offs)
    p = tl.load(
        positions + row.to(tl.int64) * NUM_CANDIDATES + i, mask=i >= 0, other=-1
    )
    valid = p >= 0
    page = tl.load(
        page_table + row.to(tl.int64) * page_table_stride + p // span,
        mask=valid,
        other=0,
    )
    tl.store(
        out + row.to(tl.int64) * TOPK + offs,
        tl.where(valid, page * span + p % span, -1),
    )


@triton.jit
def _all_visible_slots_kernel(
    global_lens,
    page_table,
    page_table_stride,
    out,
    span,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, TOPK)
    valid = offs < tl.load(global_lens + row)
    page = tl.load(
        page_table + row.to(tl.int64) * page_table_stride + offs // span,
        mask=valid,
        other=0,
    )
    tl.store(
        out + row.to(tl.int64) * TOPK + offs,
        tl.where(valid, page * span + offs % span, -1),
    )


def _local_candidates(
    logits: torch.Tensor, local_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    # sgl-kernel's HIP top-k emits local positions (-1 padded); ties resolve in
    # its own order rather than by position as the CUDA v2 kernel does.
    from sgl_kernel import fast_topk_v2

    parallel = get_parallel()
    rows = local_lens.shape[0]
    out = torch.empty((rows, topk, 2), dtype=torch.int32, device=local_lens.device)
    if rows == 0:
        return out
    if logits.shape[1] == 0:
        out[..., 1] = -1
        return out
    local_idx = fast_topk_v2(logits, local_lens.to(torch.int32), topk)
    _pack_candidates_kernel[(rows,)](
        logits,
        logits.stride(0),
        local_idx,
        out,
        WORLD=parallel.attn_dcp_size,
        RANK=parallel.attn_dcp_rank,
        TOPK=topk,
    )
    return out


def dcp_global_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    span: int,
    all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """``dsa_dcp_sharded.dcp_global_topk`` into ``out`` for its non-empty rows.

    The global top-k is the top-k of the ranks' local top-k candidates, so it selects
    over those (WORLD * topk per row) instead of a context-wide dense buffer."""
    from sgl_kernel import fast_topk_v2

    parallel = get_parallel()
    world = parallel.attn_dcp_size
    rows, topk = out.shape
    if all_gather is None:
        all_gather = partial(parallel.dcp_group.all_gather, dim=0)
    # indexer top-k capture needs every rank's slots
    if supports_dcp_topk_hip(topk, world) and not (
        get_exec().features.enable_return_indexer_topk
    ):
        # only the scores travel: every rank selects the same winners from them and
        # emits its own, which is all dcp_localize_compact keeps
        scores, positions = dcp_local_candidates_hip(
            logits, local_lens, world=world, rank=parallel.attn_dcp_rank
        )
        return dcp_global_slots_hip(
            all_gather(scores),
            positions,
            global_lens,
            page_table,
            out,
            span=span,
            world=world,
            rank=parallel.attn_dcp_rank,
        )
    candidates = _local_candidates(logits, local_lens, topk)
    # AITER's custom all-gather takes float dtypes only; the int32 (score bits,
    # position) pairs travel bit-exact as float32, else they would go to RCCL.
    gathered = all_gather(candidates.view(torch.float32)).view(torch.int32)
    num_candidates = world * topk
    scores = torch.empty(
        (rows, num_candidates), dtype=torch.float32, device=logits.device
    )
    positions = torch.empty_like(scores, dtype=torch.int32)
    lengths = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    block = 1024
    _gather_candidates_kernel[(gathered.shape[0], triton.cdiv(topk, block))](
        gathered,
        rows,
        global_lens,
        scores,
        positions,
        lengths,
        WORLD=world,
        TOPK=topk,
        BLOCK=block,
    )
    selected = fast_topk_v2(scores, lengths, topk)
    _candidates_to_slots_kernel[(rows,)](
        selected,
        positions,
        page_table,
        page_table.stride(0),
        out,
        span,
        NUM_CANDIDATES=num_candidates,
        TOPK=topk,
    )
    return out


def dcp_all_visible(
    global_lens: torch.Tensor, page_table: torch.Tensor, topk: int, span: int
) -> torch.Tensor:
    """``dsa_dcp_sharded.dcp_all_visible`` without the v2 top-k pass."""
    rows = global_lens.shape[0]
    out = torch.empty((rows, topk), dtype=torch.int32, device=global_lens.device)
    if rows == 0:
        return out
    _all_visible_slots_kernel[(rows,)](
        global_lens.to(torch.int32),
        page_table,
        page_table.stride(0),
        out,
        span,
        TOPK=topk,
    )
    return out
