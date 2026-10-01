"""Reduce exponential-race ratios without materializing a vocabulary matrix.

The caller generates noise with its existing PyTorch generator. Precise FP32
division and first-index/first-NaN argmax semantics preserve the sampled IDs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _exponential_race_partials(
    PROBS,
    NOISE,
    VALUES,
    INDICES,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    col = chunk * BLOCK + tl.arange(0, BLOCK)
    offset = row.to(tl.int64) * VOCAB + col
    p = tl.load(PROBS + offset, col < VOCAB, other=0)
    noise = tl.load(NOISE + offset, col < VOCAB, other=1)
    ratio = tl.where(col < VOCAB, tl.div_rn(p, noise), -float("inf"))
    nan = (col < VOCAB) & (ratio != ratio)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(ratio, 0)
    index = tl.min(
        tl.where(
            (col < VOCAB) & tl.where(has_nan, nan, ratio == best), col, 2147483647
        ),
        0,
    )
    tl.store(VALUES + row * CHUNKS + chunk, tl.where(has_nan, float("nan"), best))
    tl.store(INDICES + row * CHUNKS + chunk, index)


@triton.jit
def _exponential_race_merge(
    VALUES, INDICES, OUT, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.arange(0, BLOCK)
    values = tl.load(VALUES + row * CHUNKS + chunk, chunk < CHUNKS, other=-float("inf"))
    indices = tl.load(INDICES + row * CHUNKS + chunk, chunk < CHUNKS, other=2147483647)
    nan = (chunk < CHUNKS) & (values != values)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    best = tl.max(values, 0)
    index = tl.min(
        tl.where(
            (chunk < CHUNKS) & tl.where(has_nan, nan, values == best),
            indices,
            2147483647,
        ),
        0,
    )
    tl.store(OUT + row, index.to(tl.int64))


def exponential_race_argmax(probabilities: torch.Tensor, noise: torch.Tensor):
    """Return `(probabilities / noise).argmax(-1)` for contiguous FP32 rows."""
    assert probabilities.shape == noise.shape and probabilities.ndim == 2
    assert probabilities.dtype == noise.dtype == torch.float32
    assert probabilities.is_contiguous() and noise.is_contiguous()
    rows, vocab = probabilities.shape
    assert 0 < vocab < 2147483647
    block = 4096
    chunks = triton.cdiv(vocab, block)
    values = torch.empty(
        (rows, chunks), device=probabilities.device, dtype=torch.float32
    )
    indices = torch.empty(
        (rows, chunks), device=probabilities.device, dtype=torch.int32
    )
    out = torch.empty(rows, device=probabilities.device, dtype=torch.int64)
    _exponential_race_partials[(rows, chunks)](
        probabilities,
        noise,
        values,
        indices,
        vocab,
        chunks,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _exponential_race_merge[(rows,)](
        values, indices, out, chunks, triton.next_power_of_2(chunks), num_warps=4
    )
    return out
