"""Row-wise argmax for the tall-and-thin speculative verify logits.

``torch.argmax`` reduces each row in a single block, which leaves most of the
machine idle for a handful of very wide rows; a flat two-stage split over the
vocabulary saturates it instead.

Ties resolve to the lowest index, matching ``ArgMaxOps``' strict ``>``.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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


# SGLANG_SPEC_ROW_ARGMAX (default off, read once): the speculative verify's target argmax
# (eagle_utils._target_argmax) and the draft extend's top-1 (eagle_worker_v2) on the medium
# split below instead of torch.argmax, whose reduce runs one CTA row-group per output row.
SPEC_ROW_ARGMAX_ENV = "SGLANG_SPEC_ROW_ARGMAX"
SPEC_ROW_ARGMAX = os.environ.get(SPEC_ROW_ARGMAX_ENV, "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_INT32_MAX = 2**31 - 1
_noted = set()


def argmax_like_torch(x: torch.Tensor) -> Optional[torch.Tensor]:
    """``torch.argmax(x, dim=-1)`` bit for bit on the medium split, or None where it does not apply.

    ``_argmax_pair`` orders candidates as ATen's ``ArgMaxOps`` (``GreaterOrNan``): a NaN beats any
    number and the first NaN wins; otherwise the larger value, and a value tie (-0.0 == +0.0
    included) goes to the lower index. That is a total order on (value, index), so the per-split
    tree reduction and the final merge pick the index torch picks, whatever the split. The
    small-batch split of ``row_argmax`` (rows <= 64) does not order NaNs, so every row count takes
    the medium kernels here. FP32, 2-D, unit column stride, CUDA, int32 offsets.
    """
    if not (
        x.is_cuda and x.dim() == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    ):
        return None
    rows, n = x.shape
    if rows == 0 or n == 0 or (rows - 1) * x.stride(0) + n >= _INT32_MAX:
        return None
    # Measured on VR200 at a 248,320 vocab (H577 sweep): fastest of 2K-32K blocks x 4-16 warps.
    block, warps = (2048, 4) if rows <= 64 else (8192, 8)
    splits = triton.cdiv(n, block)
    values = torch.empty((rows, splits), dtype=torch.float32, device=x.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=x.device)
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    _medium_argmax_partial_kernel[(rows, splits)](
        x, values, indices, n, x.stride(0), splits, block, num_warps=warps
    )
    _medium_argmax_final_kernel[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out


def spec_row_argmax(x: torch.Tensor, site: str) -> torch.Tensor:
    """``torch.argmax(x, dim=-1)`` under SGLANG_SPEC_ROW_ARGMAX: the medium split where
    ``argmax_like_torch`` applies, torch.argmax otherwise. One INFO line per site and outcome."""
    out = argmax_like_torch(x)
    key = (site, out is not None)
    if key not in _noted:
        _noted.add(key)
        logger.info(
            "%s: %s argmax %s: rows=%d vocab=%d dtype=%s stride=%s",
            SPEC_ROW_ARGMAX_ENV,
            site,
            "on the row_argmax medium split" if out is not None else "kept torch.argmax",
            x.shape[0] if x.dim() else 0,
            x.shape[-1] if x.dim() else 0,
            x.dtype,
            tuple(x.stride()),
        )
    if out is None:
        return torch.argmax(x, dim=-1)
    return out
