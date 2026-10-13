from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 512
# The fused kernel reads Q as 16-element chunks interleaved across its 64/128 heads.
Q_CHUNK = 16


@triton.jit
def _pack_query_kernel(
    Q,
    OUT,
    stride_token,
    stride_head,
    HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    h = (i // CHUNK) % PADDED_HEADS
    d = i // (PADDED_HEADS * CHUNK) * CHUNK + i % CHUNK
    value = tl.load(
        Q + token * stride_token + h * stride_head + d,
        (h < HEADS) & (i < PADDED_HEADS * HEAD_DIM),
        other=0,
    )
    tl.store(
        OUT + token * PADDED_HEADS * HEAD_DIM + i,
        value,
        i < PADDED_HEADS * HEAD_DIM,
    )


def pack_mega_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    if (
        q.ndim != 3
        or q.dtype != torch.bfloat16
        or q.shape[-1] != HEAD_DIM
        or q.stride(-1) != 1
        or padded_heads not in (64, 128)
        or not 0 < q.shape[1] <= padded_heads
    ):
        raise ValueError("pack_mega_query needs BF16 [tokens, heads <= 64/128, 512]")
    out = q.new_empty(q.shape[0], padded_heads, HEAD_DIM)
    _pack_query_kernel[(q.shape[0], triton.cdiv(padded_heads * HEAD_DIM, 512))](
        q,
        out,
        q.stride(0),
        q.stride(1),
        HEADS=q.shape[1],
        PADDED_HEADS=padded_heads,
        HEAD_DIM=HEAD_DIM,
        CHUNK=Q_CHUNK,
        BLOCK=512,
    )
    return out
