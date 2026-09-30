"""Candidate blocks of the DeepSeek V4.1 two-level indexer on ROCm: level-one block scores and
top blocks, the top-k within them, and the sorted paged top-k (the HIP side of candidate_blocks)."""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import topk_transform_paged

# the k the AOT fast_topk op (topk_hip.hip) is instantiated for; any other k takes torch.topk
_AOT_FAST_TOPK_K = 2048


def topk_transform_paged_sorted(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: Optional[torch.Tensor],
) -> None:
    """topk_transform_paged with every row sorted ascending by position in the AOT
    kernel's epilogue, -1 padding last."""
    torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
        scores, seq_lens, page_table, page_indices, page_size, raw_indices, True
    )


# Candidate blocks one Triton program reduces; times block_size positions of logits.
_LEVEL_ONE_BLOCKS_PER_PROGRAM = 256


@triton.jit
def _candidate_block_scores_kernel(
    logits_ptr,
    seq_lens_ptr,
    out_ptr,
    logits_stride,
    width,
    num_blocks,
    out_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    FILL_TAIL: tl.constexpr,
):
    """out[row, blk] = max(logits[row, blk * BLOCK_SIZE : (blk + 1) * BLOCK_SIZE])
    over the positions < seq_lens[row]; +inf for the block holding the newest
    position, -inf for blocks past the reach (written only with FILL_TAIL). A
    program whose blocks all lie past the reach reads no logits."""
    row = tl.program_id(0)
    block0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    blocks = block0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    in_table = blocks < num_blocks
    if block0 * BLOCK_SIZE >= length:
        if FILL_TAIL:
            tl.store(out_ptr + row * out_stride + blocks, float("-inf"), mask=in_table)
        return
    cols = blocks[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    vals = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=(cols < length) & (cols < width),
        other=float("-inf"),
    )
    scores = tl.max(vals, axis=1)
    last = (length - 1) // BLOCK_SIZE
    scores = tl.where(blocks == last, float("inf"), scores)
    tl.store(out_ptr + row * out_stride + blocks, scores, mask=in_table)


@triton.jit
def _gather_candidate_blocks_kernel(
    logits_ptr,
    seq_lens_ptr,
    ids_ptr,
    out_ptr,
    logits_stride,
    width,
    ids_stride,
    out_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """out[row, j * BLOCK_SIZE + t] = logits[row, ids[row, j] * BLOCK_SIZE + t] for
    the reachable positions of the kept blocks, -inf elsewhere (an id of -1, or a
    position past the reach in the newest block)."""
    row = tl.program_id(0)
    j0 = tl.program_id(1) * BLOCKS_PER_PROGRAM
    length = tl.load(seq_lens_ptr + row)
    j = j0 + tl.arange(0, BLOCKS_PER_PROGRAM)
    ids = tl.load(ids_ptr + row * ids_stride + j)
    t = tl.arange(0, BLOCK_SIZE)[None, :]
    cols = ids[:, None] * BLOCK_SIZE + t
    vals = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=(ids[:, None] >= 0) & (cols < length) & (cols < width),
        other=float("-inf"),
    )
    tl.store(out_ptr + row * out_stride + j[:, None] * BLOCK_SIZE + t, vals)


def candidate_block_scores(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    fill_tail: bool,
) -> torch.Tensor:
    """[rows, num_blocks] fp32 block maxima of logits over the reachable positions
    (see _candidate_block_scores_kernel). seq_lens int32 [rows], contiguous."""
    assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.stride(1) == 1
    assert block_size & (block_size - 1) == 0, f"{block_size = } must be a power of 2"
    rows, width = logits.shape
    num_blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, num_blocks), dtype=torch.float32, device=logits.device)
    grid = (rows, triton.cdiv(num_blocks, _LEVEL_ONE_BLOCKS_PER_PROGRAM))
    _candidate_block_scores_kernel[grid](
        logits,
        seq_lens,
        scores,
        logits.stride(0),
        width,
        num_blocks,
        scores.stride(0),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=_LEVEL_ONE_BLOCKS_PER_PROGRAM,
        FILL_TAIL=fill_tail,
    )
    return scores


@triton.jit
def _candidate_lengths_kernel(
    seq_lens_ptr,
    block_lens_ptr,
    compact_lens_ptr,
    rows,
    topk_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """block_lens = ceil(len / BLOCK_SIZE): the blocks with a reachable position;
    compact_lens = min(block_lens, topk_blocks) * BLOCK_SIZE: the width of the
    compact candidate row (every block is kept while there are at most
    topk_blocks of them)."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = r < rows
    length = tl.load(seq_lens_ptr + r, mask=mask, other=0)
    block_lens = (length + (BLOCK_SIZE - 1)) // BLOCK_SIZE
    tl.store(block_lens_ptr + r, block_lens, mask=mask)
    tl.store(
        compact_lens_ptr + r,
        tl.minimum(block_lens, topk_blocks) * BLOCK_SIZE,
        mask=mask,
    )


@triton.jit
def _map_compact_selection_kernel(
    compact_pos_ptr,
    ids_ptr,
    seq_lens_ptr,
    page_table_ptr,
    page_indices_ptr,
    raw_indices_ptr,
    ids_stride,
    pt_stride,
    out_stride,
    n_pages,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    SORT: tl.constexpr,
    PAD_KEY: tl.constexpr,
):
    """Compact position c -> real position ids[c // BLOCK_SIZE] * BLOCK_SIZE + c % BLOCK_SIZE ->
    slot via the row's page table; reachable selections packed first, -1 after. With SORT the valid
    prefix is ascending by (position, slot)."""
    row = tl.program_id(0)
    length = tl.load(seq_lens_ptr + row)
    j = tl.arange(0, BLOCK)
    in_k = j < TOPK
    c = tl.load(compact_pos_ptr + row * TOPK + j, mask=in_k, other=-1)
    valid = c >= 0
    cc = tl.where(valid, c, 0)
    blk = tl.load(ids_ptr + row * ids_stride + cc // BLOCK_SIZE)
    real = blk * BLOCK_SIZE + cc % BLOCK_SIZE
    valid = valid & (blk >= 0) & (real < length)
    page = tl.load(
        page_table_ptr + row * pt_stride + tl.minimum(real // PAGE_SIZE, n_pages - 1),
        mask=valid,
        other=0,
    )
    slot = page * PAGE_SIZE + real % PAGE_SIZE
    if SORT:
        # BLOCK == TOPK (power of two): the sort key is the position with raw indices, else the slot
        if WRITE_RAW:
            hi = real
        else:
            hi = slot
        key = tl.where(valid, hi, PAD_KEY).to(tl.int64) << 32
        key = tl.sort(
            key | (tl.where(valid, slot, -1).to(tl.int64) & 0xFFFFFFFF), dim=0
        )
        pad = (key >> 32) == PAD_KEY
        tl.store(
            page_indices_ptr + row * out_stride + j,
            tl.where(pad, -1, (key & 0xFFFFFFFF).to(tl.int32)),
        )
        if WRITE_RAW:
            tl.store(
                raw_indices_ptr + row * out_stride + j,
                tl.where(pad, -1, (key >> 32).to(tl.int32)),
            )
    else:
        v = valid.to(tl.int32)
        count = tl.sum(v, axis=0)
        wpos = tl.cumsum(v, axis=0) - 1
        # The packed entries and the padding never share an address.
        tl.store(page_indices_ptr + row * out_stride + j, -1, mask=in_k & (j >= count))
        tl.store(page_indices_ptr + row * out_stride + wpos, slot, mask=valid)
        if WRITE_RAW:
            tl.store(
                raw_indices_ptr + row * out_stride + j, -1, mask=in_k & (j >= count)
            )
            tl.store(raw_indices_ptr + row * out_stride + wpos, real, mask=valid)


class CandidateBlocks(NamedTuple):
    """What the candidate-source layer publishes for the decode rows of a step."""

    # int32 [rows, topk_blocks]: kept block ids in no particular order, -1 padded after the last
    ids: torch.Tensor
    # int32 [rows]: kept blocks * block_size, the width of the compact row.
    compact_lens: torch.Tensor
    # int32 [rows, 1] zeros; with page_size = compact_page_size a position maps to itself
    compact_page_table: torch.Tensor
    compact_page_size: int
    block_size: int


def slice_candidate_blocks(candidates: CandidateBlocks, rows: slice) -> CandidateBlocks:
    """The rows rows of a per-request publication."""
    return candidates._replace(
        ids=candidates.ids[rows],
        compact_lens=candidates.compact_lens[rows],
        compact_page_table=candidates.compact_page_table[rows],
    )


def cat_candidate_blocks(pieces: List[CandidateBlocks]) -> CandidateBlocks:
    """Row chunks of one request's publication, in row order."""
    if len(pieces) == 1:
        return pieces[0]
    return pieces[0]._replace(
        ids=torch.cat([p.ids for p in pieces]),
        compact_lens=torch.cat([p.compact_lens for p in pieces]),
        compact_page_table=torch.cat([p.compact_page_table for p in pieces]),
    )


def select_candidate_blocks_hip(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> CandidateBlocks:
    """Level one over a [rows, capacity] logits rectangle, bounded by seq_lens: the topk_blocks
    best blocks of each row, the block with its newest position always among them; as a set per
    row the ids equal the reference's select_candidate_blocks. Graph-safe: no host sync."""
    seq_lens = seq_lens.to(torch.int32).contiguous()
    rows, width = logits.shape
    device = logits.device
    num_blocks = triton.cdiv(width, block_size)
    use_aot = topk_blocks == _AOT_FAST_TOPK_K
    scores = candidate_block_scores(
        logits, seq_lens, block_size=block_size, fill_tail=not use_aot
    )
    block_lens = torch.empty(rows, dtype=torch.int32, device=device)
    compact_lens = torch.empty(rows, dtype=torch.int32, device=device)
    _candidate_lengths_kernel[(triton.cdiv(rows, 1024),)](
        seq_lens,
        block_lens,
        compact_lens,
        rows,
        topk_blocks,
        BLOCK_SIZE=block_size,
        BLOCK=1024,
    )
    if use_aot:
        # exact top-k over the first ceil(len / block_size) block scores of each row, -1 padded
        ids = torch.empty((rows, topk_blocks), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.fast_topk(scores, ids, block_lens, None)
    else:
        picked = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        ids = picked.indices.to(torch.int32).masked_fill(
            picked.values == -torch.inf, -1
        )
        # the gather takes the width from ids and tiles it in 256-block programs
        ids = F.pad(ids, (0, topk_blocks - ids.shape[1]), value=-1)
    compact_width = topk_blocks * block_size
    return CandidateBlocks(
        ids=ids,
        compact_lens=compact_lens,
        compact_page_table=torch.zeros((rows, 1), dtype=torch.int32, device=device),
        compact_page_size=triton.next_power_of_2(compact_width),
        block_size=block_size,
    )


def gather_candidate_blocks(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    ids: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """The compact [rows, topk_blocks * block_size] fp32 row of each request's
    candidate positions in id order (see _gather_candidate_blocks_kernel)."""
    rows, width = logits.shape
    topk_blocks = ids.shape[1]
    assert ids.shape[0] == rows and ids.dtype == torch.int32 and ids.stride(1) == 1
    per_program = min(topk_blocks, _LEVEL_ONE_BLOCKS_PER_PROGRAM)
    assert topk_blocks % per_program == 0, topk_blocks
    compact = torch.empty(
        (rows, topk_blocks * block_size), dtype=torch.float32, device=logits.device
    )
    grid = (rows, topk_blocks // per_program)
    _gather_candidate_blocks_kernel[grid](
        logits,
        seq_lens.to(torch.int32).contiguous(),
        ids,
        compact,
        logits.stride(0),
        width,
        ids.stride(0),
        compact.stride(0),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=per_program,
    )
    return compact


def topk_within_candidate_blocks_hip(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    candidates: CandidateBlocks,
    *,
    page_table: torch.Tensor,
    page_size: int,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    sort_output: bool = False,
) -> None:
    """Level two for a consumer layer: the top-k of logits inside the published candidate
    blocks, written as the paged transform writes it (-1 padded, valid prefix first; ascending with
    sort_output, k a power of two). Runs on the compact row, so the cost stops growing with context."""
    rows, width = logits.shape
    topk = page_indices.shape[1]
    block_size = candidates.block_size
    seq_lens = seq_lens.to(torch.int32).contiguous()
    compact = gather_candidate_blocks(
        logits, seq_lens, candidates.ids, block_size=block_size
    )
    assert compact.shape[1] <= candidates.compact_page_size
    compact_pos = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    topk_transform_paged(
        compact,
        candidates.compact_lens,
        candidates.compact_page_table,
        compact_pos,
        candidates.compact_page_size,
        None,
    )
    assert page_indices.stride(1) == 1 and page_indices.shape == (rows, topk)
    assert page_table.stride(1) == 1 and page_table.shape[0] == rows
    write_raw = raw_indices is not None
    if write_raw:
        assert raw_indices.shape == (rows, topk) and raw_indices.stride(1) == 1
        assert raw_indices.stride(0) == page_indices.stride(0)
    assert not sort_output or topk & (topk - 1) == 0, topk
    _map_compact_selection_kernel[(rows,)](
        compact_pos,
        candidates.ids,
        seq_lens,
        page_table,
        page_indices,
        raw_indices if write_raw else page_indices,
        candidates.ids.stride(0),
        page_table.stride(0),
        page_indices.stride(0),
        page_table.shape[1],
        BLOCK_SIZE=block_size,
        PAGE_SIZE=page_size,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        WRITE_RAW=write_raw,
        SORT=sort_output,
        PAD_KEY=torch.iinfo(torch.int32).max,
    )
