"""Interleave and pad MegaAttention queries in one device launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_query(
    Q,
    OUT,
    tokens,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (PADDED_HEADS * 512)
    within = offset % (PADDED_HEADS * 512)
    head = (within // 16) % PADDED_HEADS
    channel = (within // (PADDED_HEADS * 16)) * 16 + within % 16
    value = tl.load(
        Q + row * S0 + head * S1 + channel * S2,
        (row < tokens) & (head < HEADS),
        other=0,
    )
    tl.store(OUT + offset, value, row < tokens)


def pack_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    if q.ndim != 3 or q.shape[-1] != 512 or padded_heads < q.shape[1]:
        raise ValueError("Expected [tokens, heads, 512] with enough padded heads")
    out = q.new_empty((q.shape[0], padded_heads, 512))
    if q.shape[0]:
        block = 4096
        _pack_query[(triton.cdiv(out.numel(), block),)](
            q,
            out,
            q.shape[0],
            *q.stride(),
            q.shape[1],
            padded_heads,
            block,
            num_warps=4,
        )
    return out
