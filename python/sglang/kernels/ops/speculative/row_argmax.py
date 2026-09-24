"""Row-wise argmax for the tall-and-thin speculative verify logits.

``torch.argmax`` reduces each row in a single block, which leaves most of the
machine idle for a handful of very wide rows; a flat two-stage split over the
vocabulary saturates it instead.

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


@triton.jit
def _argmax_pair(av, ai, bv, bi):
    # Torch gives NaNs priority and picks the first index for all ties.
    an, bn = av != av, bv != bv
    take_a = (an & ~bn) | ((an == bn) & ((av > bv) | (((av == bv) | an) & (ai < bi))))
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


@triton.jit
def _medium_argmax_partial_kernel(
    X,
    PV,
    PI,
    N: tl.constexpr,
    SX: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row * SX + ix, valid, float("-inf"))
    i = tl.where(valid, ix, N)
    best_v, best_i = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(PV + row * SPLITS + part, best_v)
    tl.store(PI + row * SPLITS + part, best_i)


@triton.jit
def _medium_argmax_final_kernel(
    PV,
    PI,
    OUT,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    PROBS=None,
    WRITE_PROBS: tl.constexpr = False,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    _, index = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))
    if WRITE_PROBS:
        tl.store(PROBS + row, 1.0)


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """``x.argmax(dim=-1)`` for FP32 speculative logits with a wide vocab."""
    assert x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    if rows > 64:
        # Whole aligned tiles avoid the loop and unaligned partition starts.
        # Larger batches use wider tiles to limit the total number of CTAs.
        block = 4096 if rows <= 256 else 8192
        splits = triton.cdiv(n, block)
        pv = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
        pi = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
        _medium_argmax_partial_kernel[(rows, splits)](
            x, pv, pi, n, x.stride(0), splits, block, num_warps=4
        )
        _medium_argmax_final_kernel[(rows,)](
            pv, pi, out, splits, triton.next_power_of_2(splits), num_warps=1
        )
        return out
    pv = torch.empty((rows, _SPLITS), dtype=torch.float32, device=x.device)
    pi = torch.empty((rows, _SPLITS), dtype=torch.int32, device=x.device)
    _argmax_partial_kernel[(rows, _SPLITS)](
        x, pv, pi, n, x.stride(0), SPLITS=_SPLITS, BLOCK=2048, num_warps=8
    )
    _argmax_final_kernel[(rows,)](pv, pi, out, SPLITS=_SPLITS, BLOCK=64, num_warps=2)
    return out


def speculative_argmax(x: torch.Tensor, with_probs: bool = False):
    assert x.ndim == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    assert n > 0
    out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    probs = (
        torch.empty((rows, 1), dtype=torch.float32, device=x.device)
        if with_probs
        else None
    )
    if rows:
        block = 2048 if rows <= 64 else 4096 if rows <= 256 else 8192
        splits = triton.cdiv(n, block)
        pv = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
        pi = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
        _medium_argmax_partial_kernel[(rows, splits)](
            x, pv, pi, n, x.stride(0), splits, block, num_warps=4
        )
        _medium_argmax_final_kernel[(rows,)](
            pv,
            pi,
            out,
            splits,
            triton.next_power_of_2(splits),
            PROBS=probs,
            WRITE_PROBS=with_probs,
            num_warps=1,
        )
    return (probs, out.view(rows, 1)) if with_probs else out


@triton.jit
def _argmax_verify_chain_kernel(
    PV,
    PI,
    CANDIDATES,
    TARGET,
    PREDICT,
    ACCEPT,
    CORRECT,
    WIDTH: tl.constexpr,
    CSTRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    TOKENS: tl.constexpr,
    PARTS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, TOKENS)
    part = tl.arange(0, PARTS)
    row = req * WIDTH + col
    valid = (col[:, None] < WIDTH) & (part[None, :] < SPLITS)
    v = tl.load(PV + row[:, None] * SPLITS + part[None, :], valid, float("-inf"))
    i = tl.load(PI + row[:, None] * SPLITS + part[None, :], valid, 0x7FFFFFFF)
    _, target = tl.reduce((v, i), 1, _argmax_pair)
    candidate = tl.load(CANDIDATES + req * CSTRIDE + col + 1, col + 1 < WIDTH, other=-1)
    first = tl.min(
        tl.where((col + 1 < WIDTH) & (candidate != target), col, WIDTH - 1), 0
    )
    tl.store(TARGET + row, target.to(tl.int64), col < WIDTH)
    tl.store(PREDICT + row, tl.where(col <= first, target, 0), col < WIDTH)
    tl.store(ACCEPT + row, tl.where(col <= first, row, -1), col < WIDTH)
    tl.store(CORRECT + req, first)


def greedy_verify_chain(logits, candidates):
    bs, width = candidates.shape
    rows, n = logits.shape
    assert rows == bs * width and n > 0 and width > 0
    assert (
        logits.dtype == torch.float32 and logits.stride(1) == candidates.stride(1) == 1
    )
    predict = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    accept = torch.empty((bs, width), dtype=torch.int32, device=logits.device)
    correct = torch.empty((bs,), dtype=torch.int32, device=logits.device)
    target = torch.empty((bs, width), dtype=torch.int64, device=logits.device)
    if bs:
        block = 2048 if rows <= 64 else 4096 if rows <= 256 else 8192
        splits = triton.cdiv(n, block)
        pv = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
        pi = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
        _medium_argmax_partial_kernel[(rows, splits)](
            logits, pv, pi, n, logits.stride(0), splits, block, num_warps=4
        )
        _argmax_verify_chain_kernel[(bs,)](
            pv,
            pi,
            candidates,
            target,
            predict,
            accept,
            correct,
            width,
            candidates.stride(0),
            splits,
            triton.next_power_of_2(width),
            triton.next_power_of_2(splits),
            num_warps=4,
        )
    return predict, accept, correct, target
