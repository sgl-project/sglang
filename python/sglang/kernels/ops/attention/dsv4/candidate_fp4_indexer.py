"""Score only the published candidate blocks with the original MXFP4 values."""

import torch
import triton
import triton.language as tl


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
    # An uncapped prefill can cross 2**31 elements in the compact output.
    row = tl.program_id(0).to(tl.int64)
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
        tile = 128 if rows >= 32 else 64
        _candidate_fp4_logits[(rows, triton.cdiv(output.shape[1], tile))](
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
            tile,
        )
    return output
