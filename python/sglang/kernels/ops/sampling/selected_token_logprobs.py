"""Selected-token logprobs for a verified speculative chain, fused.

A chain verify (DSpark) needs, per accepted position, the logprob of one
token: ``log_softmax(x[, x / temperature])[row, ids[row]]``. Computing that
through the full log-softmax materializes two ``[rows, vocab]`` tensors per
step just to extract one scalar per row.

These kernels build the same shift-invariant ``logprob[i] = (x[i] - max) -
log_sum`` quantity in two tile passes: partial statistics per
``_BLOCK``-column tile, then one combine that reads only each row's selected
logit. The only allocations are two ``[rows, tiles]`` fp32 statistic buffers
and the result row.

Numerical semantics match the materialized path including the edge cases:
all-``-inf`` and NaN-poisoned rows end NaN, and the results are clamped to
fp32 min like ``torch.finfo(torch.float32).min``.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["selected_token_logprobs"]

_BLOCK = 2048


@triton.jit
def _selected_logprob_partial_kernel(
    X,
    T,
    M,
    S,
    V: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    SCALED: tl.constexpr,
):
    """Per-row, per-B-column-tile ``(max, sum_exp)`` statistics."""
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * B + tl.arange(0, B)
    z = tl.load(X + row.to(tl.int64) * V + col, col < V, other=-float("inf"))
    if SCALED:
        z = tl.div_rn(z, tl.load(T + row // W))
    m = tl.max(z, 0)
    has_nan = tl.sum((z != z).to(tl.int32), 0) > 0
    # All -inf tiles contribute zero, unless the tile also contains a NaN.
    s = tl.sum(tl.where(m == -float("inf"), 0.0, tl.exp(z - m)), 0)
    s = tl.where(has_nan, float("nan"), s)
    tl.store(M + row * N + block, m)
    tl.store(S + row * N + block, s)


@triton.jit
def _selected_logprob_finish_kernel(
    X,
    IDS,
    T,
    M,
    S,
    Y,
    V: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    SCALED: tl.constexpr,
):
    """Combine the tile statistics, then read only each row's selected logit."""
    row = tl.program_id(0)
    tile = tl.arange(0, P)
    m = tl.load(M + row * N + tile, tile < N, other=-float("inf"))
    s = tl.load(S + row * N + tile, tile < N, other=0.0)
    row_max = tl.max(m, 0)
    safe_max = tl.where(row_max == -float("inf"), 0.0, row_max)
    zsum = tl.sum(s * tl.exp(m - safe_max), 0)
    token = tl.load(IDS + row)
    selected = tl.load(
        X + row.to(tl.int64) * V + token, (token >= 0) & (token < V), other=float("nan")
    )
    if SCALED:
        selected = tl.div_rn(selected, tl.load(T + row // W))
    value = (selected - row_max) - tl.log(zsum)
    value = tl.where(
        (zsum != zsum) | (row_max == -float("inf")) | (row_max == float("inf")),
        float("nan"),
        value,
    )
    low = -3.4028234663852886e38
    tl.store(Y + row, tl.where(value < low, low, value))


def selected_token_logprobs(
    x: torch.Tensor,
    ids: torch.Tensor,
    temperatures: Optional[torch.Tensor],
    width: int,
) -> torch.Tensor:
    """``log_softmax(x[, x / temperature])[row, ids[row]]`` as ``[rows // width, width]``.

    ``temperatures`` is the raw per-request tensor (``numel == rows // width``
    on x's device) — the kernels index ``T + row // width``, so passing an
    already-expanded ``[rows]`` layout would silently select wrong requests.
    Pass ``None`` for the unscaled (greedy / original-logprob) path.
    """
    assert x.ndim == 2 and x.dtype == torch.float32 and x.is_cuda
    rows, vocab = x.shape
    tiles = triton.cdiv(vocab, _BLOCK)
    maxima = torch.empty((rows, tiles), device=x.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    result = torch.empty(rows, device=x.device, dtype=torch.float32)
    temp_arg = x if temperatures is None else temperatures
    scaled = temperatures is not None
    _selected_logprob_partial_kernel[(rows, tiles)](
        x, temp_arg, maxima, sums, vocab, width, tiles, _BLOCK, scaled
    )
    _selected_logprob_finish_kernel[(rows,)](
        x,
        ids,
        temp_arg,
        maxima,
        sums,
        result,
        vocab,
        width,
        tiles,
        triton.next_power_of_2(tiles),
        scaled,
    )
    return result.reshape(-1, width)
