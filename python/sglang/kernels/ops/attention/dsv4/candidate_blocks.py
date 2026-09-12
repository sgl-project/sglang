"""Candidate-block scores and visibility masking for paged indexer logits."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_scores_kernel(
    X,
    LENS,
    OUT,
    SCORES,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, GROUP_PAD)
    cols = blocks[:, None] * GROUP + offsets[None, :]
    length = tl.load(LENS + row)
    in_bounds = (cols < WIDTH) & (offsets[None, :] < GROUP)
    values = tl.load(
        X + row * STRIDE + cols, in_bounds & (cols < length), other=-float("inf")
    ).to(tl.float32)
    if WRITE_OUTPUT:
        tl.store(OUT + row * WIDTH + cols, values, in_bounds)
    scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
    scores = tl.where(
        (length > 0) & (blocks == (length - 1) // GROUP), float("inf"), scores
    )
    tl.store(SCORES + row * BLOCKS + blocks, scores, blocks < BLOCKS)


@triton.jit
def _candidate_mask_kernel(
    X,
    LENS,
    KEEP,
    OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    visible = (cols < WIDTH) & (cols < tl.load(LENS + row))
    keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
    values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf")).to(
        tl.float32
    )
    tl.store(OUT + row * WIDTH + cols, values, cols < WIDTH)


@triton.jit
def _publish_candidate_mask_kernel(
    INDICES,
    VALUES,
    KEEP,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    selected = tl.load(INDICES + row * TOPK + i // GROUP, i < TOPK * GROUP, 0)
    score = tl.load(VALUES + row * TOPK + i // GROUP, i < TOPK * GROUP, -float("inf"))
    cols = selected * GROUP + i % GROUP
    # torch.topk returns unique block indices: each output position has one writer.
    tl.store(
        KEEP + row * WIDTH + cols,
        score > -float("inf"),
        (i < TOPK * GROUP) & (cols < WIDTH),
    )


@triton.jit
def _sort_candidate_blocks_kernel(
    BLOCKS,
    SCORES,
    LENS,
    SORTED,
    COUNTS,
    BLOCK_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TOPK_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TOPK_BLOCKS)
    blocks = tl.load(BLOCKS + row * BLOCK_STRIDE + offsets).to(tl.int32)
    safe_blocks = tl.maximum(blocks, 0)
    scores = tl.load(SCORES + row * SCORE_STRIDE + safe_blocks)
    valid = (blocks >= 0) & (scores > -float("inf"))
    blocks = tl.where(valid, blocks, NUM_BLOCKS)
    blocks = tl.sort(blocks)
    tl.store(SORTED + row * TOPK_BLOCKS + offsets, blocks)

    length = tl.load(LENS + row)
    remaining = length - blocks * BLOCK_SIZE
    contribution = tl.minimum(tl.maximum(remaining, 0), BLOCK_SIZE)
    contribution = tl.where(blocks < NUM_BLOCKS, contribution, 0)
    tl.store(COUNTS + row, tl.sum(contribution, axis=0))


@triton.jit
def _finalize_candidate_topk_kernel(
    SELECTED,
    SCORES,
    LENS,
    REQ_TO_TOKEN,
    REQ,
    CANDIDATE_BLOCKS,
    PAGE_INDICES,
    RAW_INDICES,
    SELECTED_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    CANDIDATE_BLOCK_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TOPK)
    selected = tl.load(SELECTED + row * SELECTED_STRIDE + offsets).to(tl.int64)
    length = tl.load(LENS + row).to(tl.int64)
    valid = (selected >= 0) & (selected < length)
    score = tl.load(
        SCORES + row * SCORE_STRIDE + tl.maximum(selected, 0),
        mask=valid,
        other=-float("inf"),
    )
    valid &= score > -float("inf")
    selected = tl.where(valid, selected, SOURCE_WIDTH)
    selected = tl.sort(selected)
    valid = selected < length
    safe = tl.minimum(selected, SOURCE_WIDTH - 1)

    if USE_CANDIDATES:
        block_col = safe // CANDIDATE_BLOCK_SIZE
        within = safe % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            CANDIDATE_BLOCKS + row * CANDIDATE_BLOCK_STRIDE + block_col,
            mask=valid,
            other=0,
        )
        logical = block * CANDIDATE_BLOCK_SIZE + within
    else:
        logical = safe

    req = tl.load(REQ + row).to(tl.int64)
    slot = tl.load(
        REQ_TO_TOKEN + req * REQ_STRIDE + logical * RATIO,
        mask=valid,
        other=0,
    )
    slot = slot // RATIO
    tl.store(
        PAGE_INDICES + row * OUTPUT_STRIDE + offsets,
        tl.where(valid, slot, -1),
    )
    if HAS_RAW:
        tl.store(
            RAW_INDICES + row * OUTPUT_STRIDE + offsets,
            tl.where(valid, logical, -1),
        )


def candidate_block_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    published: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep torch.topk's block selection, including its tie behavior.

    A source masks the unread tail while reducing each block. A consumer masks
    visibility and the published candidates in one pass, without copying the
    capacity-sized logits before each masked_fill.
    """
    rows, width = logits.shape
    output = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    if published is not None:
        _candidate_mask_kernel[(rows, triton.cdiv(width, 4096))](
            logits,
            seq_lens,
            published,
            output,
            width,
            logits.stride(0),
            published.stride(0),
            published.stride(1),
            4096,
        )
        return output, None

    blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, blocks), dtype=torch.float32, device=logits.device)
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        seq_lens,
        output,
        scores,
        width,
        logits.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
        True,
    )
    # Publication only needs membership; sorting the selected pairs is unused.
    top = scores.topk(min(topk_blocks, blocks), dim=-1, sorted=False)
    keep = torch.zeros((rows, width), dtype=torch.bool, device=logits.device)
    _publish_candidate_mask_kernel[
        (rows, triton.cdiv(top.indices.shape[1] * block_size, 256))
    ](
        top.indices,
        top.values,
        keep,
        width,
        block_size,
        top.indices.shape[1],
        256,
        num_warps=4,
    )
    return output, keep


def candidate_block_indices(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select candidate blocks without copying or masking the full logits tensor."""
    rows, width = logits.shape
    blocks = triton.cdiv(width, block_size)
    scores = torch.empty((rows, blocks), dtype=torch.float32, device=logits.device)
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        seq_lens,
        logits,
        scores,
        width,
        logits.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
        False,
    )
    top = scores.topk(min(topk_blocks, blocks), dim=-1)
    return top.indices, top.values > -torch.inf


def candidate_block_state(
    block_scores: torch.Tensor,
    block_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish sorted candidate blocks, valid position counts, and a shared plan."""
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    rows, num_blocks = block_scores.shape
    assert triton.next_power_of_2(topk_blocks) == topk_blocks
    assert block_lens.shape == seq_lens.shape == (rows,)

    selected_blocks = torch.empty(
        (rows, topk_blocks), dtype=torch.int32, device=block_scores.device
    )
    topk_transform_paged_v2(
        block_scores,
        block_lens,
        None,
        selected_blocks,
        1,
        plan_topk_v2(block_lens),
    )

    sorted_blocks = torch.empty_like(selected_blocks)
    counts = torch.empty(rows, dtype=torch.int32, device=block_scores.device)
    _sort_candidate_blocks_kernel[(rows,)](
        selected_blocks,
        block_scores,
        seq_lens,
        sorted_blocks,
        counts,
        selected_blocks.stride(0),
        block_scores.stride(0),
        topk_blocks,
        block_size,
        num_blocks,
        num_warps=8,
    )
    return sorted_blocks, counts, plan_topk_v2(counts)


def finalize_candidate_topk(
    selected: torch.Tensor,
    scores: torch.Tensor,
    score_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
    *,
    ratio: int,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 1,
) -> None:
    """Finalize sorted sparse-attention slots without PyTorch elementwise launches."""
    use_candidates = candidate_blocks is not None
    assert triton.next_power_of_2(page_indices.shape[1]) == page_indices.shape[1]
    source_width = (
        candidate_blocks.shape[1] * candidate_block_size
        if use_candidates
        else req_to_token.shape[1] // ratio
    )
    _finalize_candidate_topk_kernel[(selected.shape[0],)](
        selected,
        scores,
        score_lens,
        req_to_token,
        req,
        candidate_blocks if use_candidates else req_to_token,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        req_to_token.stride(0),
        candidate_blocks.stride(0) if use_candidates else 0,
        page_indices.stride(0),
        page_indices.shape[1],
        source_width,
        ratio,
        candidate_block_size,
        use_candidates,
        raw_indices is not None,
        num_warps=8,
    )


@triton.jit
def _candidate_row_lens_kernel(
    LENS,
    NBLOCKS,
    VALID,
    ROWS,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = rows < ROWS
    if USE_PDL:
        tl.extra.cuda.gdc_wait()  # LENS is the previous kernel's output
    length = tl.load(LENS + rows, mask, 0).to(tl.int32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    nblocks = (length + (BLOCK - 1)) // BLOCK
    kept = tl.minimum(nblocks, TOPK)
    # the kept blocks laid out back to back, the newest one possibly partial
    valid = BLOCK * (kept - 1) + (length - 1) % BLOCK + 1
    valid = tl.where(length > 0, valid, 0)
    tl.store(NBLOCKS + rows, nblocks, mask)
    tl.store(VALID + rows, valid, mask)


def candidate_row_lens(
    seq_lens: torch.Tensor, topk_blocks: int, block_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: its number of blocks ``ceil(seq_len / block_size)`` and the
    length of its sparse logits row once the ``min(topk_blocks, blocks)`` kept
    blocks are laid out back to back (the newest block possibly partial):
    ``block_size * (kept - 1) + (seq_len - 1) % block_size + 1``. Both int32
    ``[rows]``; a zero-length row gets 0 for both."""
    assert seq_lens.dim() == 1 and seq_lens.is_contiguous()
    rows = seq_lens.numel()
    nblocks = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    valid = torch.empty_like(nblocks)
    tile = 256
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _candidate_row_lens_kernel[(triton.cdiv(rows, tile),)](
        seq_lens,
        nblocks,
        valid,
        rows,
        topk_blocks,
        block_size,
        tile,
        use_pdl,
        num_warps=4,
        **pdl_kwargs,
    )
    return nblocks, valid
