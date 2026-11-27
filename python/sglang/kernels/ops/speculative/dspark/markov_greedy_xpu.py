"""Score-bounded BF16 Markov proposals for a single XPU request."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pruned_markov_tiles(
    base,
    embed,
    weight,
    norms,
    seed,
    values,
    ids,
    V: tl.constexpr,
    R: tl.constexpr,
    BV: tl.constexpr,
):
    tile = tl.program_id(0)
    token = tile * BV + tl.arange(0, BV)
    lane = tl.arange(0, R)
    e = tl.load(embed + lane).to(tl.float32)
    seed_id = tl.load(seed).to(tl.int32)
    w = tl.load(weight + seed_id * R + lane).to(tl.float32)
    dtype = weight.dtype.element_ty
    score_dtype = base.dtype.element_ty
    en = tl.sqrt(tl.sum(e * e, 0))
    # Allow for FP32 norm and dot-product error before BF16 rounding.
    slack = tl.load(norms + seed_id) * en * 0.001
    lower_bias = (tl.sum(w * e, 0) - slack).to(dtype)
    lower = (
        (tl.load(base + seed_id).to(tl.float32) + lower_bias.to(tl.float32))
        .to(score_dtype)
        .to(tl.float32)
    )

    b = tl.load(base + token, token < V, other=-float("inf")).to(tl.float32)
    n = tl.load(norms + token, token < V, other=0)
    upper_bias = (n * en * 1.001).to(dtype)
    upper = (b + upper_bias.to(tl.float32)).to(score_dtype).to(tl.float32)
    # Retain equality so the lowest token ID still wins a rounded tie.
    live = (token < V) & ((upper >= lower) | (lower != lower) | (upper != upper))
    best = -float("inf")
    index = V
    first_nan = V
    if tl.sum(live.to(tl.int32), 0) > 0:
        rows = tl.load(
            weight + token[:, None] * R + lane[None, :], live[:, None], other=0
        ).to(tl.float32)
        bias = tl.sum(rows * e[None, :], 1).to(dtype).to(tl.float32)
        score = (b + bias).to(score_dtype).to(tl.float32)
        best = tl.max(tl.where(live, score, -float("inf")), 0)
        index = tl.min(tl.where(live & (score == best), token, V), 0)
        first_nan = tl.min(tl.where(live & (score != score), token, V), 0)
    tl.store(values + tile, tl.where(first_nan < V, float("nan"), best))
    tl.store(ids + tile, tl.where(first_nan < V, first_nan, index))


@triton.jit
def _pruned_markov_argmax(
    values, ids, out, T: tl.constexpr, BT: tl.constexpr, V: tl.constexpr
):
    i = tl.arange(0, BT)
    v = tl.load(values + i, i < T, other=-float("inf"))
    idx = tl.load(ids + i, i < T, other=V)
    nan = tl.min(tl.where(v != v, idx, V), 0)
    v = tl.where(v == v, v, -float("inf"))
    best = tl.max(v, 0)
    winner = tl.min(tl.where(v == best, idx, V), 0)
    tl.store(out, tl.where(nan < V, nan, winner))


def markov_greedy_xpu(base, embed, w2, norms):
    # The base-logit winner provides a lower bound after its bias is evaluated.
    seed = base.argmax(dim=-1)
    block = 32
    vocab, rank = w2.shape
    tiles = triton.cdiv(vocab, block)
    out = torch.empty((1,), dtype=torch.int64, device=base.device)
    values = torch.empty((tiles,), dtype=torch.float32, device=base.device)
    ids = torch.empty((tiles,), dtype=torch.int32, device=base.device)
    _pruned_markov_tiles[(tiles,)](
        base, embed, w2, norms, seed, values, ids, vocab, rank, block, num_warps=4
    )
    _pruned_markov_argmax[(1,)](
        values, ids, out, tiles, triton.next_power_of_2(tiles), vocab, num_warps=8
    )
    return out
