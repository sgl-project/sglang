"""Row-wise argmax for the tall-and-thin speculative verify logits.

``torch.argmax`` dispatches ``at::native::reduce_kernel`` here, which needs
~24 us for the [6, 129280] FP32 verify logits on GB300 -- two orders of
magnitude off the 3 MB the reduction actually reads.  The rows are few and very
wide, so a flat two-stage split over the vocabulary saturates the machine
instead of leaving it to one block per row.

Ties resolve to the lowest index, matching ``ArgMaxOps``' strict ``>``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_partial_kernel(
    X, OUTV, OUTI, N, SX, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    per = tl.cdiv(N, SPLITS)
    start = part * per
    best_v = float("-inf")
    best_i = N
    for off in tl.range(start, tl.minimum(start + per, N), BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < tl.minimum(start + per, N)
        v = tl.load(X + row * SX + idx, m, float("-inf"))
        cur_v = tl.max(v, 0)
        # lowest index among the maxima of this tile
        cur_i = tl.min(tl.where(v == cur_v, idx, N), 0)
        take = (cur_v > best_v) | ((cur_v == best_v) & (cur_i < best_i))
        best_i = tl.where(take, cur_i, best_i)
        best_v = tl.where(take, cur_v, best_v)
    tl.store(OUTV + row * SPLITS + part, best_v)
    tl.store(OUTI + row * SPLITS + part, best_i)


@triton.jit
def _argmax_final_kernel(INV, INI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < SPLITS
    v = tl.load(INV + row * SPLITS + o, m, float("-inf"))
    i = tl.load(INI + row * SPLITS + o, m, 0x7FFFFFFF)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, 0x7FFFFFFF), 0)
    tl.store(OUT + row, best_i.to(tl.int64))


_SPLITS = 64


def fast_row_argmax(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``x.argmax(dim=-1)`` for a 2D FP32 tensor with few rows and a wide vocab."""
    assert x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    if out is None:
        out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    pv = torch.empty((rows, _SPLITS), dtype=torch.float32, device=x.device)
    pi = torch.empty((rows, _SPLITS), dtype=torch.int32, device=x.device)
    _argmax_partial_kernel[(rows, _SPLITS)](
        x, pv, pi, n, x.stride(0), SPLITS=_SPLITS, BLOCK=2048, num_warps=8
    )
    _argmax_final_kernel[(rows,)](pv, pi, out, SPLITS=_SPLITS, BLOCK=64, num_warps=2)
    return out
