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
    DIVISOR_MIN: tl.constexpr = None,
    IGNORE_NAN: tl.constexpr = False,
):
    row, part = tl.program_id(0), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row * SX + ix, valid, float("-inf"))
    if DIVISOR is not None:
        divisor = tl.load(DIVISOR + row * SD + ix, valid, 1.0)
        if DIVISOR_MIN is not None:
            # clamp_min keeps a NaN divisor; tl.maximum would replace it.
            divisor = tl.where(
                divisor != divisor, divisor, tl.maximum(divisor, DIVISOR_MIN)
            )
        v = tl.div_rn(v, divisor)
    if IGNORE_NAN:
        v = tl.where(v == v, v, float("-inf"))
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


def div_argmax(
    x: torch.Tensor,
    divisor: torch.Tensor,
    *,
    divisor_min: Optional[float] = None,
    ignore_nan: bool = False,
) -> torch.Tensor:
    """FP32 row-wise ``(x / divisor).argmax(-1)`` with unit column strides.

    ``divisor_min`` first applies ``divisor.clamp_min(divisor_min)``. A NaN quotient
    wins as in ``torch.argmax`` unless ``ignore_nan``; then it never wins, the rule
    of the CUDA ``sampling.argmax`` kernel (a row with nothing above -inf gives 0).
    """
    assert x.ndim == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    assert (
        divisor.shape == x.shape
        and divisor.dtype == x.dtype
        and divisor.device == x.device
        and divisor.stride(1) == 1
    )
    return _medium_row_argmax(
        x, divisor, divisor_min=divisor_min, ignore_nan=ignore_nan
    )


def _medium_row_argmax(x, divisor=None, divisor_min=None, ignore_nan=False):
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
        DIVISOR_MIN=divisor_min,
        IGNORE_NAN=ignore_nan,
        num_warps=4,
    )
    _medium_argmax_final_kernel[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out
