"""Stands in for the stock seeded FP64 Gumbel sampler (layers/sampler.py multinomial_with_seed) for speed, with the
same request-local semantics: that one scans the whole vocabulary in one row reduction; this races 1024-column tiles.

Request-local exponential-race sampling (new numerical policy).

Unlike the retained FP64 Gumbel reference, this uses FP32 race scores. Seeded
outputs intentionally change. The hash remains a function of request seed,
position and token ID, independent of batch shape and global RNG state.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32


@triton.jit
def _choose_winner(a_value, a_index, b_value, b_index):
    # Match torch.argmax: the first NaN wins, otherwise the first maximum wins.
    take_a = (
        (a_value > b_value)
        | ((a_value == b_value) & (a_index < b_index))
        | ((a_value != a_value) & ((b_value == b_value) | (a_index < b_index)))
    )
    return tl.where(take_a, a_value, b_value), tl.where(take_a, a_index, b_index)


@triton.jit
def _merge_winners(
    values, indices, output, num_tiles: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    tile = tl.arange(0, block)
    value = tl.load(values + row * num_tiles + tile, tile < num_tiles, -float("inf"))
    index = tl.load(indices + row * num_tiles + tile, tile < num_tiles, 2147483647)
    _, index = tl.reduce((value, index), 0, _choose_winner)
    tl.store(output + row, index)


@triton.jit
def _exponential_noise(hashed):
    # Midpoints of 2**32 equiprobable bins: U=(h+0.5)/2**32, strictly in (0,1).
    # For the upper half, form 1-U from the integer complement BEFORE casting.
    # This retains tail resolution and avoids rounding U to 1 in FP32.
    lower = (hashed.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    upper = ((0xFFFFFFFF - hashed).to(tl.float32) + 0.5) * 2.3283064365386963e-10
    lower = tl.minimum(lower, 0.5)
    upper = tl.minimum(upper, 0.5)
    return tl.where(
        hashed < 0x80000000, -libdevice.log(lower), -libdevice.log1p(-upper)
    )


@triton.jit
def _race_tiles(
    H,
    L,
    M,
    V,
    I,
    RS: tl.constexpr,
    CS: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * B + tl.arange(0, B)
    h = tl.load(H + row * N + col, col < N, 0)
    lp = tl.load(L + row * RS + col * CS, col < N, -float("inf"))
    maximum = tl.load(M + row)
    # Row-max shift prevents overflow and makes results independent of a common
    # logweight offset. Underflowed tails cannot beat the max-weight token:
    # midpoint noise is in [~1.16e-10, ~22.88], so a winner has relative weight
    # at least ~5e-12, far above FP32 underflow.
    delta = (lp - maximum).to(tl.float32)
    score = libdevice.exp(delta) / _exponential_noise(h)
    score = tl.where(lp == -float("inf"), -float("inf"), score)
    score = tl.where(col < N, score, -float("inf"))
    index = tl.where(col < N, col, 2147483647)
    value, index = tl.reduce((score, index), 0, _choose_winner)
    tl.store(V + row * T + tile, value)
    tl.store(I + row * T + tile, index)


def sample_from_logprobs(logprobs, seeds, positions):
    """Sample finite logweights (with optional -inf masks); return int64 [rows,1].

    All-masked rows retain the legacy index-zero sentinel, not a valid draw.
    NaN/+inf logweights are outside the sampling domain. Probability estimates
    and trainer logprobs are not modified by this sampling-only operation.
    """
    if logprobs.ndim != 2 or logprobs.dtype not in (
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("TOP race requires 2D BF16, FP32 or FP64 logweights")
    rows, vocab = logprobs.shape
    if logprobs.device.type != "cuda" or not 0 < vocab < 2147483647:
        raise ValueError(
            "TOP race requires CUDA and a nonempty int32-indexed vocabulary"
        )
    for name, tensor in (("seeds", seeds), ("positions", positions)):
        if (
            tensor.shape != (rows,)
            or tensor.device != logprobs.device
            or tensor.dtype
            not in (torch.int32, torch.int64, torch.uint32, torch.uint64)
        ):
            raise ValueError(
                f"TOP race {name} must be integer [rows] on the logits device"
            )
    output = torch.empty((rows, 1), dtype=torch.int64, device=logprobs.device)
    if rows == 0:
        return output
    hashes = murmur_hash32(
        seeds.to(torch.uint64).contiguous(),
        positions.contiguous(),
        torch.arange(vocab, device=logprobs.device),
    )
    maximum = torch.amax(logprobs, dim=1)
    block = 1024
    tiles = triton.cdiv(vocab, block)
    values = torch.empty((rows, tiles), dtype=torch.float32, device=logprobs.device)
    indices = torch.empty((rows, tiles), dtype=torch.int32, device=logprobs.device)
    _race_tiles[(rows, tiles)](
        hashes,
        logprobs,
        maximum,
        values,
        indices,
        *logprobs.stride(),
        vocab,
        tiles,
        block,
        num_warps=4,
    )
    _merge_winners[(rows,)](
        values, indices, output, tiles, triton.next_power_of_2(tiles), num_warps=4
    )
    return output
