"""Candidate-block scores and visibility masking for paged indexer logits."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_scores_kernel(
    X,
    LENS,
    OUT,
    SCORES,
    BLOCK_LENS,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
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
    tl.store(SCORES + row * SCORE_STRIDE + blocks, scores, blocks < BLOCKS)
    tl.store(
        BLOCK_LENS + row,
        (length + GROUP - 1) // GROUP,
        mask=tl.program_id(1) == 0,
    )


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
def _materialize_candidate_slots_kernel(
    SORTED,
    LENS,
    REQ_TO_TOKEN,
    REQ,
    POSITIONS,
    SLOTS,
    WIDTH: tl.constexpr,
    TOPK_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block_col = offsets // BLOCK_SIZE
    within = offsets % BLOCK_SIZE
    block = tl.load(
        SORTED + row * TOPK_BLOCKS + block_col,
        mask=block_col < TOPK_BLOCKS,
        other=NUM_BLOCKS,
    )
    position = block * BLOCK_SIZE + within
    length = tl.load(LENS + row)
    valid = (offsets < WIDTH) & (block < NUM_BLOCKS) & (position < length)
    req = tl.load(REQ + row).to(tl.int64)
    slot = tl.load(
        REQ_TO_TOKEN + req * REQ_STRIDE + position * RATIO,
        mask=valid,
        other=0,
    )
    tl.store(POSITIONS + row * WIDTH + offsets, position, mask=offsets < WIDTH)
    tl.store(SLOTS + row * WIDTH + offsets, slot // RATIO, mask=offsets < WIDTH)


@triton.jit
def _finalize_candidate_topk_kernel(
    SELECTED,
    SCORES,
    LENS,
    REQ_TO_TOKEN,
    REQ,
    CANDIDATE_POSITIONS,
    CANDIDATE_SLOTS,
    PAGE_INDICES,
    RAW_INDICES,
    SELECTED_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    CANDIDATE_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
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
        logical = tl.load(
            CANDIDATE_POSITIONS + row * CANDIDATE_STRIDE + safe,
            mask=valid,
            other=0,
        )
        slot = tl.load(
            CANDIDATE_SLOTS + row * CANDIDATE_STRIDE + safe,
            mask=valid,
            other=0,
        )
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
    block_lens = torch.empty(rows, dtype=torch.int32, device=logits.device)
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        seq_lens,
        output,
        scores,
        block_lens,
        width,
        logits.stride(0),
        scores.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
        True,
    )
    top = scores.topk(min(topk_blocks, blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return output, keep.repeat_interleave(block_size, dim=-1)[..., :width]


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
    score_storage = torch.empty(
        (rows, triton.cdiv(blocks, 4) * 4),
        dtype=torch.float32,
        device=logits.device,
    )
    scores = score_storage[:, :blocks]
    block_lens = torch.empty(rows, dtype=torch.int32, device=logits.device)
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[(rows, triton.cdiv(blocks, tile))](
        logits,
        seq_lens,
        logits,
        scores,
        block_lens,
        width,
        logits.stride(0),
        scores.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
        False,
    )
    top = scores.topk(min(topk_blocks, blocks), dim=-1)
    return top.indices, top.values > -torch.inf


def candidate_slots(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    ratio: int,
    block_scores: torch.Tensor | None = None,
    block_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish sorted logical positions and physical slots for Reindex layers."""
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    rows, logical_width = logits.shape
    num_blocks = triton.cdiv(logical_width, block_size)
    if block_scores is None:
        score_storage = torch.empty(
            (rows, triton.cdiv(num_blocks, 4) * 4),
            dtype=torch.float32,
            device=logits.device,
        )
        block_scores = score_storage[:, :num_blocks]
        block_lens = torch.empty(rows, dtype=torch.int32, device=logits.device)
        group_pad = triton.next_power_of_2(block_size)
        block_tile = max(1, 1024 // group_pad)
        _candidate_scores_kernel[(rows, triton.cdiv(num_blocks, block_tile))](
            logits,
            seq_lens,
            logits,
            block_scores,
            block_lens,
            logical_width,
            logits.stride(0),
            block_scores.stride(0),
            num_blocks,
            block_size,
            group_pad,
            block_tile,
            False,
        )
    else:
        assert block_scores.shape == (rows, num_blocks)
        assert block_lens is not None and block_lens.shape == (rows,)

    selected_blocks = torch.empty(
        (rows, topk_blocks), dtype=torch.int32, device=logits.device
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
    counts = torch.empty(rows, dtype=torch.int32, device=logits.device)
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

    candidate_width = topk_blocks * block_size
    positions = torch.empty(
        (rows, candidate_width), dtype=torch.int64, device=logits.device
    )
    slots = torch.empty(
        (rows, candidate_width), dtype=torch.int32, device=logits.device
    )
    tile = 256
    _materialize_candidate_slots_kernel[(rows, triton.cdiv(candidate_width, tile))](
        sorted_blocks,
        seq_lens,
        req_to_token,
        req,
        positions,
        slots,
        candidate_width,
        topk_blocks,
        block_size,
        num_blocks,
        req_to_token.stride(0),
        ratio,
        tile,
    )
    return positions, slots, counts


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
    candidate_positions: torch.Tensor | None = None,
    candidate_slots: torch.Tensor | None = None,
) -> None:
    """Finalize sorted sparse-attention slots without PyTorch elementwise launches."""
    use_candidates = candidate_positions is not None
    assert use_candidates == (candidate_slots is not None)
    source_width = (
        candidate_positions.shape[1]
        if use_candidates
        else req_to_token.shape[1] // ratio
    )
    _finalize_candidate_topk_kernel[(selected.shape[0],)](
        selected,
        scores,
        score_lens,
        req_to_token,
        req,
        candidate_positions if use_candidates else req_to_token,
        candidate_slots if use_candidates else req_to_token,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        req_to_token.stride(0),
        candidate_positions.stride(0) if use_candidates else 0,
        page_indices.stride(0),
        page_indices.shape[1],
        source_width,
        ratio,
        use_candidates,
        raw_indices is not None,
        num_warps=8,
    )
