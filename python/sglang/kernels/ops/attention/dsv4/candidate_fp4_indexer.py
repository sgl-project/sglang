"""Score only the published candidate blocks with the original MXFP4 values."""

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
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr,
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
    tl.store(OUT + row * OUT_STRIDE + cols, values, in_bounds)
    scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
    scores = tl.where(
        (length > 0) & (blocks == (length - 1) // GROUP), float("inf"), scores
    )
    tl.store(SCORES + row * BLOCKS + blocks, scores, blocks < BLOCKS)


@triton.jit
def _candidate_fp4_logits(
    Q,
    QS,
    K,
    KS,
    W,
    BLOCK_IDS,
    LENS,
    OUT,
    H: tl.constexpr,
    BLOCKS: tl.constexpr,
    GROUP: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * N + tl.arange(0, N)
    block = tl.load(
        BLOCK_IDS + row * ID_STRIDE + cols // GROUP, cols < BLOCKS * GROUP, other=-1
    )
    pos = block * GROUP + cols % GROUP
    valid = (cols < BLOCKS * GROUP) & (block >= 0) & (pos < tl.load(LENS + row))
    h = tl.arange(0, H)
    d = tl.arange(0, 64)
    s = tl.arange(0, 4)
    q = tl.load(Q + row * H * 64 + h[:, None] * 64 + d[None, :])
    qs = tl.load(QS + row * H + h)
    qs = ((qs[:, None] >> (s[None, :] * 8)) & 255).to(tl.uint8)
    k = tl.load(K + pos[:, None] * 64 + d[None, :], valid[:, None], other=0)
    ks = tl.load(KS + pos, valid, other=0)
    ks = ((ks[:, None] >> (s[None, :] * 8)) & 255).to(tl.uint8)
    acc = tl.dot_scaled(
        q.to(tl.uint8), qs, "e2m1", tl.trans(k.to(tl.uint8)), ks, "e2m1"
    )
    weights = tl.load(W + row * H + h)
    # DeepGEMM accumulates four interleaved head streams with FP32 FMA.
    doubled = acc + tl.abs(acc)
    sums = tl.full((4, N), 0, tl.float32)
    lanes = tl.arange(0, 4)
    for group in tl.static_range(H // 4):
        scores = tl.gather(
            doubled, tl.broadcast_to((group * 4 + lanes)[:, None], (4, N)), 0
        )
        w = tl.gather(weights, group * 4 + lanes, 0)
        sums = tl.fma(scores, w[:, None], sums)
    even = tl.sum(tl.where((lanes % 2)[:, None] == 0, sums, 0), 0)
    odd = tl.sum(tl.where((lanes % 2)[:, None] == 1, sums, 0), 0)
    logits = (even + odd) * 0.5
    tl.store(
        OUT + row * BLOCKS * GROUP + cols,
        tl.where(valid, logits, -float("inf")),
        cols < BLOCKS * GROUP,
    )


def candidate_fp4_mqa_logits(q_fp4, k_fp4, weights, block_ids, seq_lens, block_size):
    rows, heads, _ = q_fp4[0].shape
    blocks = block_ids.shape[1]
    output = torch.empty(
        (rows, blocks * block_size), dtype=torch.float32, device=weights.device
    )
    if rows and blocks:
        _candidate_fp4_logits[(rows, triton.cdiv(output.shape[1], 64))](
            *q_fp4,
            *k_fp4,
            weights,
            block_ids,
            seq_lens,
            output,
            heads,
            blocks,
            block_size,
            block_ids.stride(0),
            64,
        )
    return output


def select_candidate_block_indices(logits, seq_lens, topk_blocks, block_size):
    """Mask the unread tail in place and retain block IDs instead of position masks."""
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
        logits.stride(0),
        blocks,
        block_size,
        group_pad,
        tile,
    )
    top = scores.topk(min(topk_blocks, blocks), dim=-1)
    # Ascending IDs put a partial final block and invalid padding last.
    return (
        top.indices.masked_fill(~(top.values > -torch.inf), blocks)
        .sort(-1)
        .values.to(torch.int32)
    )
