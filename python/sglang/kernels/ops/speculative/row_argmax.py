"""Row-wise argmax for the tall-and-thin speculative verify logits.

``torch.argmax`` reduces each row in a single block, which leaves most of the
machine idle for a handful of very wide rows; a flat two-stage split over the
vocabulary saturates it instead.

Ties resolve to the lowest index, matching ``ArgMaxOps``' strict ``>``.
"""

from typing import Optional

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
    DIVISOR=None,
    SD: tl.constexpr = 0,
):
    row, part = tl.program_id(0), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row * SX + ix, valid, float("-inf"))
    if DIVISOR is not None:
        divisor = tl.load(DIVISOR + row * SD + ix, valid, 1.0)
        v = tl.div_rn(v, divisor)
    i = tl.where(valid, ix, N)
    best_v, best_i = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(PV + row * SPLITS + part, best_v)
    tl.store(PI + row * SPLITS + part, best_i)


@triton.jit
def _medium_argmax_final_kernel(PV, PI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    _, index = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))


def div_argmax(x: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """FP32 row-wise ``(x / divisor).argmax(-1)`` with unit column strides."""
    assert x.ndim == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    assert (
        divisor.shape == x.shape
        and divisor.dtype == x.dtype
        and divisor.device == x.device
        and divisor.stride(1) == 1
    )
    return _medium_row_argmax(x, divisor)


def _medium_row_argmax(x, divisor=None):
    rows, n = x.shape
    block = 4096 if rows <= 256 else 8192
    splits = triton.cdiv(n, block)
    values = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    _medium_argmax_partial_kernel[(rows, splits)](
        x,
        values,
        indices,
        n,
        x.stride(0),
        splits,
        block,
        DIVISOR=divisor,
        SD=divisor.stride(0) if divisor is not None else 0,
        num_warps=4,
    )
    _medium_argmax_final_kernel[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """``x.argmax(dim=-1)`` for FP32 speculative logits with a wide vocab."""
    assert x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    if rows > 64:
        return _medium_row_argmax(x)
    out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    pv = torch.empty((rows, _SPLITS), dtype=torch.float32, device=x.device)
    pi = torch.empty((rows, _SPLITS), dtype=torch.int32, device=x.device)
    _argmax_partial_kernel[(rows, _SPLITS)](
        x, pv, pi, n, x.stride(0), SPLITS=_SPLITS, BLOCK=2048, num_warps=8
    )
    _argmax_final_kernel[(rows,)](pv, pi, out, SPLITS=_SPLITS, BLOCK=64, num_warps=2)
    return out


@triton.jit
def _draft_tail_final_kernel(
    PV,
    PI,
    OUT,
    PROBS,
    SEED_SRC,
    SEED_INDEX,
    SEED_OUT,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    SEED_WIDTH: tl.constexpr,
    SEED_SRC_STRIDE: tl.constexpr,
    SEED_INDEX_STRIDE: tl.constexpr,
    SEED_BLOCK: tl.constexpr,
    HAS_SEED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    _, index = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))
    tl.store(PROBS + row, 1.0)
    if HAS_SEED:
        # The row's seed: SEED_SRC[SEED_INDEX[row]], as an indexed gather.
        src = tl.load(SEED_INDEX + row * SEED_INDEX_STRIDE).to(tl.int64)
        for start in tl.static_range(0, SEED_WIDTH, SEED_BLOCK):
            cols = start + tl.arange(0, SEED_BLOCK)
            m = cols < SEED_WIDTH
            seed = tl.load(SEED_SRC + src * SEED_SRC_STRIDE + cols, m)
            tl.store(SEED_OUT + row * SEED_WIDTH + cols, seed, m)


def draft_extend_tail(
    logits: torch.Tensor,
    seed_src: Optional[torch.Tensor] = None,
    seed_index: Optional[torch.Tensor] = None,
):
    """The topk-1 draft-extend tail in two launches.

    Returns ``(topk_p, topk_index, seed)`` equal to
    ``torch.ones_like(i, dtype=torch.float32)``, ``i = logits.argmax(-1,
    keepdim=True)`` (NaN first, lowest index on ties) and ``seed_src[seed_index]``
    (None without ``seed_src``). ``logits``: FP32 rows with a unit column
    stride; ``seed_src``: 2-D with a unit column stride; ``seed_index``: 1-D
    non-negative row indices, one per logits row.
    """
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert logits.stride(1) == 1
    rows, n = logits.shape
    assert n > 0
    has_seed = seed_src is not None
    index = torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
    probs = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
    seed = None
    if has_seed:
        assert seed_src.ndim == 2 and seed_src.stride(1) == 1
        assert seed_index is not None and seed_index.shape == (rows,)
        seed = torch.empty(
            (rows, seed_src.shape[1]), dtype=seed_src.dtype, device=seed_src.device
        )
    if rows:
        block = 2048 if rows <= 64 else 4096 if rows <= 256 else 8192
        splits = triton.cdiv(n, block)
        pv = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
        pi = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
        _medium_argmax_partial_kernel[(rows, splits)](
            logits, pv, pi, n, logits.stride(0), splits, block, num_warps=4
        )
        seed_width = seed_src.shape[1] if has_seed else 1
        _draft_tail_final_kernel[(rows,)](
            pv,
            pi,
            index,
            probs,
            seed_src if has_seed else pv,
            seed_index if has_seed else pi,
            seed if has_seed else pv,
            splits,
            triton.next_power_of_2(splits),
            seed_width,
            seed_src.stride(0) if has_seed else 0,
            seed_index.stride(0) if has_seed else 0,
            min(1024, triton.next_power_of_2(seed_width)),
            has_seed,
            num_warps=4,
        )
    return probs, index, seed
