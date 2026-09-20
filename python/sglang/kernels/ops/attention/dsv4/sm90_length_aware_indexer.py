"""Opt-in 32-head Hopper indexer with device-side prefix lengths.

Score buffers keep their graph-stable capacity, but only the visible prefix
(and its final 64-element tile) is initialized. Every consumer MUST use the
device lengths. TopK v2 is exact by score, with arbitrary cutoff-tie selection;
it does not promise PyTorch's choice of indices with equal scores.
"""

import torch
import triton
import triton.language as tl

from .sm90_fp4_indexer import _e2m1_decode
from .topk import topk_transform_ragged_v2


@triton.jit
def _prepare_lengths(
    POS,
    CAPACITY,
    VISIBLE,
    LENGTHS,
    ROWS: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    pos = tl.load(POS + row * POS_STRIDE, row < ROWS, other=-1)
    n = tl.minimum(tl.maximum((pos + 1) // RATIO, 0), WIDTH).to(tl.int32)
    tl.store(VISIBLE + row, n, row < ROWS)
    if HAS_CANDIDATES:
        capacity = tl.load(CAPACITY + row, row < ROWS, other=0)
        tl.store(LENGTHS + row, tl.minimum(n, capacity), row < ROWS)


def prepare_candidate_lengths(pos, ratio, width, candidates=None):
    """Fuse position conversion, clamping, casting and compact score bounds."""
    visible = torch.empty(pos.shape[0], dtype=torch.int32, device=pos.device)
    lengths = torch.empty_like(visible) if candidates is not None else visible
    _prepare_lengths[(triton.cdiv(pos.shape[0], 256),)](
        pos,
        candidates.lengths if candidates is not None else visible,
        visible,
        lengths,
        pos.shape[0],
        pos.stride(0),
        ratio,
        width,
        candidates is not None,
        num_warps=4,
    )
    return visible, lengths


@triton.jit
def _prefix_logits(
    Q,
    W,
    MAPPING,
    REQ,
    LENS,
    TABLE,
    MASK,
    OUT,
    BLOCKS,
    IS_PREFIX,
    VISIBLE,
    Q_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.load(LENS + row)
    request = tl.load(REQ + row).to(tl.int64)
    h = tl.arange(0, 32)
    d = tl.arange(0, 64)
    # Reuse query/weights across the tiles owned by this CTA.
    qe = tl.load(Q + row * Q_STRIDE + h[:, None] * 128 + 2 * d[None, :])
    qo = tl.load(Q + row * Q_STRIDE + h[:, None] * 128 + 2 * d[None, :] + 1)
    weight = tl.load(W + row * W_STRIDE + h).to(tl.float32)
    if HAS_BLOCKS:
        visible_n = tl.load(VISIBLE + row)
        is_prefix = tl.load(IS_PREFIX + row)
    for tile in range(tl.program_id(1), tl.cdiv(n, 64), tl.num_programs(1)):
        col = tile * 64 + tl.arange(0, 64)
        visible = col < n
        valid = visible
        position = col
        if HAS_BLOCKS:
            # If all selected blocks form a prefix, keep the short-context
            # scoring path's contiguous accesses and avoid the block gather.
            if is_prefix == 0:
                block = tl.load(
                    BLOCKS + row * BLOCK_STRIDE + col // BLOCK_SIZE,
                    visible,
                    other=-1,
                )
                position = block * BLOCK_SIZE + col % BLOCK_SIZE
            valid = valid & (position >= 0) & (position < visible_n)
        if HAS_MASK:
            valid = valid & tl.load(
                MASK + row * MASK_STRIDE + col, visible, other=False
            )
        if tl.sum(valid.to(tl.int32), 0) > 0:
            slot = (
                tl.load(
                    MAPPING + request * MAP_STRIDE + position * RATIO, valid, other=0
                ).to(tl.int64)
                // RATIO
            )
            base = (slot // PAGE) * TABLE_STRIDE
            off = slot % PAGE
            payload = tl.load(
                TABLE + base[:, None] + off[:, None] * 64 + d[None, :],
                valid[:, None],
                other=0,
            )
            scales = tl.load(
                TABLE
                + base[:, None]
                + PAGE * 64
                + off[:, None] * 4
                + tl.arange(0, 4)[None, :],
                valid[:, None],
                other=127,
            )
            scale = tl.exp2(scales.to(tl.float32) - 127.0)[:, :, None]
            lo = _e2m1_decode(payload & 15)
            hi = _e2m1_decode(payload >> 4)
            kl = tl.reshape(tl.reshape(lo, (64, 4, 16)) * scale, (64, 64)).to(
                tl.bfloat16
            )
            kh = tl.reshape(tl.reshape(hi, (64, 4, 16)) * scale, (64, 64)).to(
                tl.bfloat16
            )
            acc = tl.dot(qe, tl.trans(kl))
            acc += tl.dot(qo, tl.trans(kh))
            score = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
            score = (score * weight[:, None]).to(tl.bfloat16).to(tl.float32)
            score = tl.sum(score, 0).to(tl.bfloat16).to(tl.float32)
            score = tl.where(valid, score, -float("inf"))
        else:
            score = tl.full((64,), -float("inf"), tl.float32)
        # TopK uses 16-byte loads at the prefix boundary. Initialize the whole
        # last tile, including the <= 3 padding floats beyond the logical width.
        tl.store(OUT + row * OUT_STRIDE + col, score, col < OUT_STRIDE)


def prefix_logits(
    q,
    weights,
    mapping,
    req,
    lens,
    table,
    page,
    ratio,
    width,
    mask=None,
    *,
    candidates=None,
    visible=None,
):
    """Return partially initialized scores, bounded by int32 device lengths.

    lens bounds the score buffer in [0, width]. With candidates, visible is the
    original prefix length and lens bounds the compact candidate positions.
    """
    assert candidates is None or (visible is not None and mask is None)
    rows = q.shape[0]
    stride = triton.cdiv(width, 4) * 4
    out = torch.empty((rows, stride), dtype=torch.float32, device=q.device)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    tiles = min(
        triton.cdiv(width, 64),
        max(8, triton.next_power_of_2(triton.cdiv(8 * sm_count, rows))),
    )
    _prefix_logits[(rows, tiles)](
        q,
        weights,
        mapping,
        req,
        lens,
        table,
        mask if mask is not None else out,
        out,
        candidates.blocks if candidates is not None else out,
        candidates.is_prefix if candidates is not None else lens,
        visible if visible is not None else lens,
        q.stride(0),
        weights.stride(0),
        mapping.stride(0),
        table.stride(0),
        mask.stride(0) if mask is not None else 0,
        stride,
        candidates.blocks.stride(0) if candidates is not None else 0,
        page,
        ratio,
        mask is not None,
        candidates is not None,
        candidates.block_size if candidates is not None else 1,
        num_warps=4,
    )
    return out


def select_prefix_topk(scores, lens, k):
    """Exact TopK by score over each visible prefix; -1 pads short rows."""
    selected = torch.empty(
        (scores.shape[0], k), dtype=torch.int32, device=scores.device
    )
    # The paged small-batch dispatch can request a 16-CTA cluster, while H20
    # supports at most 8. Reuse v2's length-aware non-cluster entry point.
    offsets = torch.zeros_like(lens)
    topk_transform_ragged_v2(scores, lens, out_offsets=offsets, out_indices=selected)
    return selected


@triton.jit
def _block_scores(
    SCORES,
    LENS,
    OUT,
    BLOCK_LENS,
    SCORE_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.load(LENS + row)
    count = tl.cdiv(n, BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(BLOCK_LENS + row, count)
    for tile in range(tl.program_id(1), tl.cdiv(count, TILE), tl.num_programs(1)):
        block = tile * TILE + tl.arange(0, TILE)
        pos = block[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        v = tl.load(SCORES + row * SCORE_STRIDE + pos, pos < n, other=-float("inf"))
        score = tl.max(v, 1)
        score = tl.where(block == count - 1, float("inf"), score)
        tl.store(OUT + row * OUT_STRIDE + block, score, block < OUT_STRIDE)


@triton.jit
def _publish_blocks(
    SELECTED,
    SCORES,
    MASK,
    K: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.program_id(1) * 256 + tl.arange(0, 256)
    block = tl.load(SELECTED + row * K + i // BLOCK_SIZE, i < K * BLOCK_SIZE, other=-1)
    valid = (i < K * BLOCK_SIZE) & (block >= 0)
    score = tl.load(
        SCORES + row * SCORE_STRIDE + tl.maximum(block, 0), valid, other=-float("inf")
    )
    pos = block * BLOCK_SIZE + i % BLOCK_SIZE
    # Keep the whole last block, exactly as the existing candidate mask does;
    # visibility is enforced independently by every score consumer.
    tl.store(
        MASK + row * WIDTH + pos, True, valid & (score > -float("inf")) & (pos < WIDTH)
    )


def _candidate_block_topk(scores, lens, width, topk_blocks, block_size):
    rows = scores.shape[0]
    blocks = triton.cdiv(width, block_size)
    stride = triton.cdiv(blocks, 4) * 4
    block_scores = torch.empty(
        (rows, stride), dtype=torch.float32, device=scores.device
    )
    block_lens = torch.empty_like(lens)
    _block_scores[(rows, min(16, triton.cdiv(blocks, 128)))](
        scores,
        lens,
        block_scores,
        block_lens,
        scores.stride(0),
        stride,
        block_size,
        128,
        num_warps=4,
    )
    k = min(topk_blocks, blocks)
    selected = select_prefix_topk(block_scores, block_lens, k)
    return selected, block_scores


def candidate_mask(scores, lens, width, topk_blocks, block_size):
    selected, block_scores = _candidate_block_topk(
        scores, lens, width, topk_blocks, block_size
    )
    rows, k = selected.shape
    # Preserve CandidateMasks' full-width contract for fallback consumers. Only
    # the source layer initializes this bool buffer; consumers read valid tiles.
    mask = torch.zeros((rows, width), dtype=torch.bool, device=scores.device)
    _publish_blocks[(rows, triton.cdiv(k * block_size, 256))](
        selected,
        block_scores,
        mask,
        k,
        block_scores.stride(0),
        width,
        block_size,
        num_warps=4,
    )
    return mask


@triton.jit
def _sort_blocks(
    SELECTED,
    SCORES,
    LENS,
    BLOCKS,
    LENGTHS,
    IS_PREFIX,
    K: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    count = tl.cdiv(tl.load(LENS + row), BLOCK_SIZE)
    if count <= K:
        # Short contexts keep every finite block. Avoid a 2048-element bitonic
        # sort when the candidate list is already the contiguous prefix.
        block = i
        valid = (i < K) & (i < count)
    else:
        block = tl.load(SELECTED + row * K + i, i < K, other=-1)
        valid = (i < K) & (block >= 0)
    score = tl.load(
        SCORES + row * SCORE_STRIDE + tl.maximum(block, 0),
        valid,
        other=-float("inf"),
    )
    valid = valid & (score > -float("inf"))
    block = tl.where(valid, block, 0x7FFFFFFF)
    # All -inf blocks are excluded even when K covers the prefix. Such holes
    # need compaction just like the selected subset of a long context does.
    if count > K or tl.sum(valid.to(tl.int32), 0) != count:
        block = tl.sort(block, descending=False)
    valid = block != 0x7FFFFFFF
    count = tl.sum(valid.to(tl.int32), 0)
    tl.store(BLOCKS + row * K + i, tl.where(valid, block, -1), i < K)
    # Keep entire selected blocks, matching CandidateMasks. Consumer visibility
    # independently masks the final partial block and any differing ratio.
    tl.store(LENGTHS + row, tl.minimum(count * BLOCK_SIZE, WIDTH))
    tl.store(IS_PREFIX + row, tl.sum((valid & (block != i)).to(tl.int32), 0) == 0)


def candidate_blocks(scores, lens, width, topk_blocks, block_size):
    """Publish a compact list using prefix scores and non-cluster TopK."""
    from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateBlocks

    selected, block_scores = _candidate_block_topk(
        scores, lens, width, topk_blocks, block_size
    )
    rows, k = selected.shape
    blocks = torch.empty_like(selected)
    lengths = torch.empty_like(lens)
    is_prefix = torch.empty_like(lens)
    _sort_blocks[(rows,)](
        selected,
        block_scores,
        lens,
        blocks,
        lengths,
        is_prefix,
        k,
        block_scores.stride(0),
        width,
        block_size,
        triton.next_power_of_2(k),
        num_warps=4,
    )
    return CandidateBlocks(blocks, lengths, is_prefix, width, block_size)


@triton.jit
def _blocks_to_mask(
    BLOCKS,
    MASK,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.program_id(1) * 256 + tl.arange(0, 256)
    block = tl.load(BLOCKS + row * K + i // BLOCK_SIZE, i < K * BLOCK_SIZE, other=-1)
    pos = block * BLOCK_SIZE + i % BLOCK_SIZE
    tl.store(
        MASK + row * WIDTH + pos,
        True,
        (i < K * BLOCK_SIZE) & (block >= 0) & (pos < WIDTH),
    )


def materialize_candidate_mask(candidates):
    """Build the legacy mask only when a later layer takes a fallback path."""
    rows, k = candidates.blocks.shape
    mask = torch.zeros(
        (rows, candidates.width), dtype=torch.bool, device=candidates.blocks.device
    )
    _blocks_to_mask[(rows, triton.cdiv(k * candidates.block_size, 256))](
        candidates.blocks,
        mask,
        k,
        candidates.width,
        candidates.block_size,
        num_warps=4,
    )
    return mask


@triton.jit
def _publish_topk(
    SELECTED,
    SCORES,
    LENS,
    REQ,
    MAPPING,
    PAGES,
    RAW,
    BLOCKS,
    K: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_RAW: tl.constexpr,
    FILTER_MASKED: tl.constexpr,
    HAS_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    logical = tl.load(SELECTED + row * K + col, col < K, other=-1)
    n = tl.load(LENS + row)
    valid = (logical >= 0) & (logical < n) & (col < K)
    if FILTER_MASKED:
        score = tl.load(
            SCORES + row * SCORE_STRIDE + tl.maximum(logical, 0),
            valid,
            other=-float("inf"),
        )
        valid = valid & (score > -float("inf"))
    if HAS_BLOCKS:
        block = tl.load(
            BLOCKS + row * BLOCK_STRIDE + tl.maximum(logical, 0) // BLOCK_SIZE,
            valid,
            other=-1,
        )
        logical = block * BLOCK_SIZE + logical % BLOCK_SIZE
    logical = tl.sort(tl.where(valid, logical, 0x7FFFFFFF), descending=False)
    valid = (logical != 0x7FFFFFFF) & (col < K)
    request = tl.load(REQ + row).to(tl.int64)
    position = tl.where(valid, logical, 0).to(tl.int64)
    slot = tl.load(MAPPING + request * MAP_STRIDE + position * RATIO, valid, other=0)
    tl.store(
        PAGES + row * PAGE_STRIDE + col, tl.where(valid, slot // RATIO, -1), col < K
    )
    if HAS_RAW:
        tl.store(RAW + row * RAW_STRIDE + col, tl.where(valid, logical, -1), col < K)


def publish_topk(
    selected,
    scores,
    lens,
    req,
    mapping,
    pages,
    raw,
    ratio,
    filter_masked,
    *,
    candidates=None,
):
    rows, k = selected.shape
    _publish_topk[(rows,)](
        selected,
        scores,
        lens,
        req,
        mapping,
        pages,
        raw if raw is not None else pages,
        candidates.blocks if candidates is not None else selected,
        k,
        scores.stride(0),
        mapping.stride(0),
        pages.stride(0),
        raw.stride(0) if raw is not None else 0,
        candidates.blocks.stride(0) if candidates is not None else 0,
        ratio,
        raw is not None,
        filter_masked,
        candidates is not None,
        candidates.block_size if candidates is not None else 1,
        triton.next_power_of_2(k),
        num_warps=4,
    )
