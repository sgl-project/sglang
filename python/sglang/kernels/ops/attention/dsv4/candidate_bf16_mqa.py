"""Small prefill scorer; Torch remains the reference and default backend.

Read candidates from shared, already dequantized K. Neither gathered K nor
per-head scores are materialized in global memory. Top-K stays with the caller.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_bf16_mqa(
    Q,
    K,
    W,
    BLOCKS,
    LENS,
    OUT,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    QR: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KR: tl.constexpr,
    KD: tl.constexpr,
    WR: tl.constexpr,
    WH: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    LS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    head = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    block = tl.load(BLOCKS + row * BR + (col // 8) * BC, mask=col < WIDTH, other=-1).to(
        tl.int64
    )
    pos = block * 8 + col % 8
    visible = tl.minimum(tl.load(LENS + row * LS), LENGTH)
    valid = (col < WIDTH) & (pos >= 0) & (pos < visible)
    # Masked K loads also protect padded/invalid candidate addresses.
    keys = tl.load(
        K + dim[:, None] * KD + pos[None, :] * KR, mask=valid[None, :], other=0
    )
    query = tl.load(Q + row * QR + head[:, None] * QH + dim[None, :] * QD)
    dot = tl.dot(query, keys).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W + row * WR + head * WH).to(tl.float32)
    weighted = (tl.maximum(dot, 0.0) * weight[:, None]).to(tl.bfloat16)
    # Preserve Torch's BF16 output rounding after multiplication and head sum.
    score = tl.sum(weighted.to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(
        OUT + row * WIDTH + col, tl.where(valid, score, -float("inf")), mask=col < WIDTH
    )


def candidate_bf16_mqa_logits(q, keys, weights, blocks, visible_lens):
    """Score candidate blocks without a per-query K copy.

    q [R,32,128], keys [L,128], weights [R,32] are BF16. blocks [R,C]
    contains logical block IDs (8 positions/block), with -1 padding.
    visible_lens [R] is exclusive causal visibility in the same K coordinates.
    Return FP32 [R,C*8], with -inf for invalid positions. Candidate order is
    preserved, so the caller can reuse existing Top-K and index restoration.
    Only finite model activations are supported; this is not a quantizer.
    """
    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [rows, 32, 128]")
    rows = q.shape[0]
    if keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("keys must have shape [length, 128]")
    if weights.shape != (rows, 32):
        raise ValueError("weights must have shape [rows, 32]")
    if blocks.ndim != 2 or blocks.shape[0] != rows:
        raise ValueError("blocks must have shape [rows, candidate_blocks]")
    if visible_lens.shape != (rows,):
        raise ValueError("visible_lens must have shape [rows]")
    if any(t.dtype != torch.bfloat16 for t in (q, keys, weights)):
        raise ValueError("q, keys and weights must be BF16")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (blocks, visible_lens)):
        raise ValueError("blocks and visible_lens must be int32 or int64")
    if not q.is_cuda or any(
        t.device != q.device for t in (keys, weights, blocks, visible_lens)
    ):
        raise ValueError("all inputs must be on the same CUDA device")
    width = blocks.shape[1] * 8
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if not rows or not width:
        return out
    if not keys.shape[0]:
        return out.fill_(-torch.inf)
    _candidate_bf16_mqa[(rows, triton.cdiv(width, 64))](
        q,
        keys,
        weights,
        blocks,
        visible_lens,
        out,
        keys.shape[0],
        width,
        *q.stride(),
        *keys.stride(),
        *weights.stride(),
        *blocks.stride(),
        visible_lens.stride(0),
        TILE=64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
