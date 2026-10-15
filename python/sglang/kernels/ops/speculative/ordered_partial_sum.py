from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_partial_sum_kernel(
    src_ptr,  # [M, world * N] fp32: rank r's partial at columns [r * N, (r + 1) * N)
    dst_ptr,  # [M, N] bf16
    N,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = src_ptr + row * WORLD * N + offs
    acc = tl.load(base, mask=mask, other=0.0)
    for r in tl.static_range(1, WORLD):
        acc += tl.load(base + r * N, mask=mask, other=0.0)
    tl.store(dst_ptr + row * N + offs, acc.to(tl.bfloat16), mask=mask)


def ordered_partial_sum(gathered: torch.Tensor, world: int, out: torch.Tensor) -> torch.Tensor:
    """Sum ``world`` fp32 partials in rank order, rounding once to bf16 into ``out``."""
    m, n = out.shape
    block = 1024
    _ordered_partial_sum_kernel[(m, triton.cdiv(n, block))](
        gathered, out, n, WORLD=world, BLOCK=block, num_warps=4
    )
    return out
