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
def _prefix_logits(
    Q,
    W,
    MAPPING,
    REQ,
    LENS,
    TABLE,
    MASK,
    OUT,
    Q_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_MASK: tl.constexpr,
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
    for tile in range(tl.program_id(1), tl.cdiv(n, 64), tl.num_programs(1)):
        col = tile * 64 + tl.arange(0, 64)
        visible = col < n
        valid = visible
        if HAS_MASK:
            valid = valid & tl.load(
                MASK + row * MASK_STRIDE + col, visible, other=False
            )
        if tl.sum(valid.to(tl.int32), 0) > 0:
            slot = (
                tl.load(
                    MAPPING + request * MAP_STRIDE + col * RATIO, valid, other=0
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


def prefix_logits(q, weights, mapping, req, lens, table, page, ratio, width, mask=None):
    """Return partially initialized scores; lens must be int32 in [0, width]."""
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
        q.stride(0),
        weights.stride(0),
        mapping.stride(0),
        table.stride(0),
        mask.stride(0) if mask is not None else 0,
        stride,
        page,
        ratio,
        mask is not None,
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


def candidate_mask(scores, lens, width, topk_blocks, block_size):
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
    # Preserve CandidateMasks' full-width contract for fallback consumers. Only
    # the source layer initializes this bool buffer; consumers read valid tiles.
    mask = torch.zeros((rows, width), dtype=torch.bool, device=scores.device)
    _publish_blocks[(rows, triton.cdiv(k * block_size, 256))](
        selected,
        block_scores,
        mask,
        k,
        stride,
        width,
        block_size,
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
    K: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_RAW: tl.constexpr,
    FILTER_MASKED: tl.constexpr,
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
    logical = tl.sort(tl.where(valid, logical, 0x7FFFFFFF), descending=False)
    valid = (logical < n) & (col < K)
    request = tl.load(REQ + row).to(tl.int64)
    position = tl.where(valid, logical, 0).to(tl.int64)
    slot = tl.load(MAPPING + request * MAP_STRIDE + position * RATIO, valid, other=0)
    tl.store(
        PAGES + row * PAGE_STRIDE + col, tl.where(valid, slot // RATIO, -1), col < K
    )
    if HAS_RAW:
        tl.store(RAW + row * RAW_STRIDE + col, tl.where(valid, logical, -1), col < K)


def publish_topk(
    selected, scores, lens, req, mapping, pages, raw, ratio, filter_masked
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
        k,
        scores.stride(0),
        mapping.stride(0),
        pages.stride(0),
        raw.stride(0) if raw is not None else 0,
        ratio,
        raw is not None,
        filter_masked,
        triton.next_power_of_2(k),
        num_warps=4,
    )
