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


@triton.jit
def _npu_medium_argmax_partial_kernel(
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
    # Ascend variant: the two-operand tl.reduce((v, i), combine) form lowers
    # to a region tt.reduce the Ascend triton-mlir-opt cannot parse, so the
    # NPU path uses one-operand tl.max/tl.min instead. Ties resolve to the
    # lowest index; an all-NaN tile leaves the index at the N sentinel.
    row, part = tl.program_id(0), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row * SX + ix, valid, float("-inf"))
    if DIVISOR is not None:
        divisor = tl.load(DIVISOR + row * SD + ix, valid, 1.0)
        v = tl.div_rn(v, divisor)
    i = tl.where(valid, ix, N)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, N), 0)
    tl.store(PV + row * SPLITS + part, best_v)
    tl.store(PI + row * SPLITS + part, best_i)


@triton.jit
def _npu_medium_argmax_final_kernel(
    PV, PI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    v = tl.load(PV + row * SPLITS + part, part < SPLITS, float("-inf"))
    i = tl.load(PI + row * SPLITS + part, part < SPLITS, 0x7FFFFFFF)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, 0x7FFFFFFF), 0)
    tl.store(OUT + row, best_i.to(tl.int64))


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
    # The Ascend triton-mlir-opt cannot parse the two-operand tt.reduce the
    # tl.reduce variants lower to; NPU runs the one-operand variants instead.
    if x.device.type == "npu":
        partial, final = (
            _npu_medium_argmax_partial_kernel,
            _npu_medium_argmax_final_kernel,
        )
    else:
        partial, final = _medium_argmax_partial_kernel, _medium_argmax_final_kernel
    partial[(rows, splits)](
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
    final[(rows,)](
        values, indices, out, splits, triton.next_power_of_2(splits), num_warps=1
    )
    return out
