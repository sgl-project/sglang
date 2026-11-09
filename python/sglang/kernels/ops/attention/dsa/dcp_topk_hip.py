"""DCP sparse-attention top-2048 on AMD (csrc/dsa/dcp_topk_hip.cuh).

One launch picks a rank's local candidates; after the scores alone are all-gathered, one picks
the global top-2048 and keeps the rank's own winners as KV slots. Positions never leave the
rank. Row lengths are read on the device, so the logits buffer may be wider than any row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK = 2048
# Scores per thread held in registers (1024 threads); longer rows re-read the excess
# per pass. 16 covers a DCP8 rank's share of a 131K context; 32 spills on gfx950.
_LOCAL_PER_THREAD = 16


@cache_once
def _local_module(per: int) -> Module:
    args = make_cpp_args(per)
    return load_jit(
        "dcp_local_candidates_hip",
        *args,
        cuda_files=["dsa/dcp_topk_hip.cuh"],
        cuda_wrappers=[("run", f"DcpLocalCandidatesHip<{args}>::run")],
    )


@cache_once
def _local_chunked_module(per: int) -> Module:
    args = make_cpp_args(per)
    return load_jit(
        "dcp_local_candidates_chunked_hip",
        *args,
        cuda_files=["dsa/dcp_topk_hip.cuh"],
        cuda_wrappers=[("run", f"DcpLocalCandidatesChunkedHip<{args}>::run")],
    )


@cache_once
def _global_module(world: int) -> Module:
    args = make_cpp_args(world)
    return load_jit(
        "dcp_global_slots_hip",
        *args,
        cuda_files=["dsa/dcp_topk_hip.cuh"],
        cuda_wrappers=[("run", f"DcpGlobalSlotsHip<{args}>::run")],
    )


def supports_dcp_topk_hip(topk: int, world: int) -> bool:
    return topk == TOPK and world in (2, 4, 8)


def dcp_local_candidates_hip(
    logits: torch.Tensor, local_lens: torch.Tensor, world: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-2048 of logits[i, :local_lens[i]]: [rows, 2048] fp32 scores (-inf padded), to
    all-gather, and their [rows, 2048] int32 global positions (-1 padded), kept on the rank.
    Local position j is global position j * world + rank."""
    rows = local_lens.shape[0]
    scores = torch.empty((rows, TOPK), dtype=torch.float32, device=logits.device)
    positions = torch.empty((rows, TOPK), dtype=torch.int32, device=logits.device)
    max_chunks = _local_max_chunks(logits.shape[1])
    if max_chunks > 1:
        chunk_scores = torch.empty(
            (rows, max_chunks, TOPK), dtype=torch.float32, device=logits.device
        )
        chunk_positions = torch.empty_like(chunk_scores, dtype=torch.int32)
        _local_chunked_module(_LOCAL_PER_THREAD).run(
            logits,
            local_lens.to(torch.int32),
            scores,
            positions,
            chunk_scores,
            chunk_positions,
            world,
            rank,
        )
        return scores, positions
    _local_module(_LOCAL_PER_THREAD).run(
        logits, local_lens.to(torch.int32), scores, positions, world, rank
    )
    return scores, positions


# 1 selects the single-block kernel; the merge holds at most one block's registers of candidates.
def _local_max_chunks(width: int) -> int:
    if not envs.SGLANG_DSA_DCP_TOPK_CHUNKED.get():
        return 1
    chunk = _LOCAL_PER_THREAD * 1024
    return min(triton.cdiv(width, chunk), chunk // TOPK)


def dcp_global_slots_hip(
    gathered: torch.Tensor,
    positions: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    out: torch.Tensor,
    span: int,
    world: int,
    rank: int,
) -> torch.Tensor:
    """This rank's share of the top-2048 per row of the rank-major [world * rows, 2048]
    gathered scores, with ``positions`` this rank's from dcp_local_candidates_hip.

    Written to out [rows, 2048] as slots page_table[p // span] * span + p % span where the
    winner is this rank's, -1 elsewhere: every rank computes the same winner set, so the
    ranks' outputs are disjoint and their union is the global top-k."""
    _global_module(world).run(
        gathered, positions, global_lens, page_table, out, span, rank
    )
    return out


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
