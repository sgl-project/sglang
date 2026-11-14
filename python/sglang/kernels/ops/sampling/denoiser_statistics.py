"""Chunked FP32 statistics for large-vocabulary block denoising.

The partials carry a maximum, exponential sum and centered first moment.
Merging these computes entropy without materializing log-probabilities. A
separate pass writes FP32 probabilities for the existing categorical sampler
and self-conditioning GEMM. Sampling and its generator state stay unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _denoiser_partial_statistics(
    LOGITS,
    TEMPERATURES,
    PARTIALS,
    PARTIAL_ARGMAX,
    VOCAB: tl.constexpr,
    CANVAS: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    j = part * BLOCK + tl.arange(0, BLOCK)
    temp = tl.load(TEMPERATURES + row // CANVAS)
    raw = tl.load(
        LOGITS + row.to(tl.int64) * VOCAB + j, mask=j < VOCAB, other=-float("inf")
    ).to(tl.float32)
    # Approximate reciprocal scaling changes sharp distributions enough to
    # exceed the statistics tolerance. Match FP32 division rounding.
    scaled = tl.div_rn(raw, temp)
    mx = tl.max(scaled, 0)
    d = scaled - tl.where(mx == -float("inf"), 0.0, mx)
    e = tl.exp(d)
    z = tl.sum(e, 0)
    s = tl.sum(tl.where(j < VOCAB, d * e, 0.0), 0)
    # Preserve PyTorch argmax's first-NaN and first-maximum behavior.
    # An explicit NaN reduction is required; testing the reduced maximum
    # itself is not equivalent after Triton lowering.
    rawmax = tl.max(raw, 0)
    has_nan = tl.sum(((j < VOCAB) & (raw != raw)).to(tl.int32), 0) > 0
    idx = tl.min(
        tl.where(
            (j < VOCAB) & tl.where(has_nan, raw != raw, raw == rawmax), j, 2147483647
        ),
        0,
    )
    off = row * CHUNKS + part
    tl.store(PARTIALS + off, mx)
    tl.store(PARTIALS + ROWS * CHUNKS + off, z)
    tl.store(PARTIALS + 2 * ROWS * CHUNKS + off, s)
    tl.store(PARTIALS + 3 * ROWS * CHUNKS + off, rawmax)
    tl.store(PARTIAL_ARGMAX + off, idx)


@triton.jit
def _denoiser_merge_statistics(
    PARTIALS,
    PARTIAL_ARGMAX,
    NORMALIZER,
    ENTROPY,
    ARGMAX,
    ROWS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.arange(0, BLOCK)
    off = row * CHUNKS + p
    mx = tl.load(PARTIALS + off, mask=p < CHUNKS, other=-float("inf"))
    z = tl.load(PARTIALS + ROWS * CHUNKS + off, mask=p < CHUNKS, other=0.0)
    moment = tl.load(PARTIALS + 2 * ROWS * CHUNKS + off, mask=p < CHUNKS, other=0.0)
    maximum = tl.max(mx, 0)
    shift = mx - maximum
    alpha = tl.exp(shift)
    denom = tl.sum(alpha * z, 0)
    total = tl.sum(
        tl.where(p < CHUNKS, alpha * (moment + tl.where(z > 0, z * shift, 0.0)), 0.0), 0
    )
    logz = tl.log(denom)
    tl.store(NORMALIZER + row, maximum)
    tl.store(NORMALIZER + ROWS + row, logz)
    tl.store(ENTROPY + row, logz - total / denom)
    rawmax = tl.load(
        PARTIALS + 3 * ROWS * CHUNKS + off, mask=p < CHUNKS, other=-float("inf")
    )
    best = tl.max(rawmax, 0)
    idx = tl.load(PARTIAL_ARGMAX + off, mask=p < CHUNKS, other=2147483647)
    has_nan = tl.sum(((p < CHUNKS) & (rawmax != rawmax)).to(tl.int32), 0) > 0
    tl.store(
        ARGMAX + row,
        tl.min(
            tl.where(
                (p < CHUNKS) & tl.where(has_nan, rawmax != rawmax, rawmax == best),
                idx,
                2147483647,
            ),
            0,
        ),
    )


@triton.jit
def _denoiser_write_probabilities(
    LOGITS,
    TEMPERATURES,
    NORMALIZER,
    PROBABILITIES,
    SOFT_PROBABILITIES,
    VOCAB: tl.constexpr,
    CANVAS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_SOFT: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    temp = tl.load(TEMPERATURES + row // CANVAS)
    mx = tl.load(NORMALIZER + row)
    logz = tl.load(NORMALIZER + ROWS + row)
    raw = tl.load(
        LOGITS + row.to(tl.int64) * VOCAB + j, mask=j < VOCAB, other=-float("inf")
    ).to(tl.float32)
    p = tl.exp(tl.div_rn(raw, temp) - mx - logz)
    tl.store(PROBABILITIES + row.to(tl.int64) * VOCAB + j, p, mask=j < VOCAB)
    if WRITE_SOFT:
        tl.store(
            SOFT_PROBABILITIES + row.to(tl.int64) * VOCAB + j,
            p.to(tl.bfloat16),
            mask=j < VOCAB,
        )


def denoiser_statistics(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    soft_probabilities: torch.Tensor | None = None,
):
    """Statistics for contiguous FP32 [B, M, V] logits and [B] temperatures.

    soft_probabilities, when supplied, is a contiguous BF16 buffer shaped like logits.
    """
    assert logits.ndim == 3 and temperatures.shape == (logits.shape[0],)
    block = 4096
    batch, canvas, vocab = logits.shape
    rows = batch * canvas
    parts = triton.cdiv(vocab, block)
    partials = torch.empty((4, rows, parts), device=logits.device, dtype=torch.float32)
    partial_argmax = torch.empty((rows, parts), device=logits.device, dtype=torch.int32)
    normalizer = torch.empty((2, rows), device=logits.device, dtype=torch.float32)
    probabilities = torch.empty_like(logits, dtype=torch.float32)
    entropies = torch.empty((batch, canvas), device=logits.device, dtype=torch.float32)
    argmax = torch.empty((batch, canvas), device=logits.device, dtype=torch.int64)
    _denoiser_partial_statistics[(rows, parts)](
        logits,
        temperatures,
        partials,
        partial_argmax,
        vocab,
        canvas,
        rows,
        parts,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _denoiser_merge_statistics[(rows,)](
        partials,
        partial_argmax,
        normalizer,
        entropies,
        argmax,
        rows,
        parts,
        triton.next_power_of_2(parts),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _denoiser_write_probabilities[(rows, triton.cdiv(vocab, block))](
        logits,
        temperatures,
        normalizer,
        probabilities,
        soft_probabilities if soft_probabilities is not None else probabilities,
        vocab,
        canvas,
        rows,
        block,
        soft_probabilities is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return probabilities, entropies, argmax
