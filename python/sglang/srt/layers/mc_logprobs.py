# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent categorical probes, with replacement, from processed logits."""

import torch

import triton
import triton.language as tl


@triton.jit
def _partials(
    X,
    Max,
    Sum,
    V: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - tl.maximum(m, -3.402823466e38)), 0)
    tl.store(Max + row * TILES + tile, m)
    tl.store(Sum + row * TILES + tile, z)


@triton.jit
def _draw(
    X,
    Max,
    Sum,
    Seeds,
    Positions,
    OutIds,
    OutLP,
    V: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, draw = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, REDUCE)
    m = tl.load(Max + row * TILES + t, t < TILES, other=-float("inf"))
    z = tl.load(Sum + row * TILES + t, t < TILES, other=0.0)
    maximum = tl.max(m, 0)
    mass = z * tl.exp(m - maximum)
    cdf = tl.cumsum(mass, 0)
    total = tl.sum(mass, 0)
    seed = tl.load(Seeds + row).to(tl.uint64) ^ 0x6A09E667F3BCC909
    pos = tl.load(Positions + row).to(tl.uint32)
    # A separate Philox domain leaves the response token's RNG untouched.
    u = tl.rand(seed, pos * 32 + draw)
    cdf_total = tl.sum(tl.where(t == TILES - 1, cdf, 0.0), 0)
    target = tl.minimum(u * cdf_total, cdf_total * (1.0 - 1.1920928955078125e-7))
    tile = tl.min(tl.where((cdf > target) & (mass > 0), t, TILES), 0)
    before = tl.sum(tl.where(t == tile - 1, cdf, 0.0), 0)
    tile_m = tl.sum(tl.where(t == tile, m, 0.0), 0)
    # Avoid 0 * -inf when reducing masked empty tiles.
    local_scale = tl.exp(tile_m - maximum)
    j = tl.arange(0, BLOCK)
    ids = tile * BLOCK + j
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float("inf")).to(tl.float32)
    weights = tl.exp(x - tile_m)
    local_cdf = tl.cumsum(weights, 0)
    local_total = tl.sum(tl.where(j == BLOCK - 1, local_cdf, 0.0), 0)
    local_target = tl.minimum(
        tl.maximum((target - before) / local_scale, 0.0),
        local_total * (1.0 - 1.1920928955078125e-7),
    )
    token = tl.min(tl.where((local_cdf > local_target) & (weights > 0), ids, V), 0)
    value = tl.load(X + row * STRIDE + token).to(tl.float32)
    logz = maximum + tl.log(total)
    tl.store(OutIds + row * 32 + draw, token)
    tl.store(OutLP + row * 32 + draw, value - logz)


def compute_mc_scores(logits, seeds, positions, block=4096):
    """32 independent categorical draws per row, with replacement.

    Input: processed, temperature-scaled logits, with finite support per row.
    Returns int32 IDs and float32 full-vocabulary normalized log probabilities.
    Separate seed domain leaves response sampling unchanged. No Python scalars
    or CPU synchronization are introduced in the draw path.
    """
    batch, vocab = logits.shape
    if logits.stride(1) != 1 or seeds.numel() != batch or positions.numel() != batch:
        raise ValueError(
            "MC32 requires contiguous vocabulary rows and one seed/position per row"
        )
    tiles = triton.cdiv(vocab, block)
    maxima = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    ids = torch.empty((batch, 32), device=logits.device, dtype=torch.int32)
    lp = torch.empty((batch, 32), device=logits.device, dtype=torch.float32)
    if batch:
        _partials[(batch, tiles)](
            logits, maxima, sums, vocab, logits.stride(0), tiles, block
        )
        _draw[(batch, 32)](
            logits,
            maxima,
            sums,
            seeds,
            positions,
            ids,
            lp,
            vocab,
            logits.stride(0),
            tiles,
            triton.next_power_of_2(tiles),
            block,
        )
    return lp, ids
