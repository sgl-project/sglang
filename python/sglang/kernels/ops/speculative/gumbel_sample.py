"""Fused Gumbel-max draw for the draft sampler.

One kernel per draw: each (row, split) program scans its vocab chunk with
in-register Exp(1) noise (Philox via ``tl.rand``; q = -log(1 - u)), writes its
chunk winner, and the LAST program of a row (split-K last-block pattern,
``acq_rel`` counter) reduces the chunk winners, gathers the winning
probability, resets the counter for the next launch, and advances the
device-side RNG state — so a CUDA-graph replay draws fresh noise every step
without any host-side RNG bookkeeping.

The per-row counters and partial buffers live in a per-(device, bs) workspace
that is never freed (captured graphs bake its addresses); the counters are
self-cleaning, so the workspace is allocated once and never re-zeroed. Two
same-(device, bs) draws must not run concurrently on different streams — they
would race on that workspace (the draft sampler only ever draws on the
forward stream).

Ties (bitwise-equal scores) resolve to the smaller vocab index; continuous
noise makes them measure-zero. ``noise`` may be passed explicitly (tests /
non-Philox reproduction) to read Exp(1) noise from memory instead of
generating it in-register.
"""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

_MAX_SPLITS = 128
_BLOCK = 2048

_workspaces: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, ...]] = {}


def _workspace(device: torch.device, bs: int):
    key = (device, bs)
    ws = _workspaces.get(key)
    if ws is None:
        partial_scores = torch.empty(
            (bs, _MAX_SPLITS), dtype=torch.float32, device=device
        )
        partial_indices = torch.empty(
            (bs, _MAX_SPLITS), dtype=torch.int64, device=device
        )
        counters = torch.zeros((bs,), dtype=torch.int32, device=device)
        rng_state = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64, device=device)
        ws = (partial_scores, partial_indices, counters, rng_state)
        _workspaces[key] = ws
    return ws


@triton.jit
def _gumbel_argmax_kernel(
    probs_ptr,
    noise_ptr,
    partial_scores_ptr,
    partial_indices_ptr,
    counters_ptr,
    rng_state_ptr,
    sample_p_ptr,
    sample_index_ptr,
    vocab,
    vocab_padded4,
    probs_stride,
    noise_stride,
    chunk,
    num_splits,
    SPLITS_CEIL: tl.constexpr,
    GEN_NOISE: tl.constexpr,
    TINY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    p_base = probs_ptr + row * probs_stride

    if GEN_NOISE:
        seed = tl.load(rng_state_ptr).to(tl.int32)

    lo = split * chunk
    hi = tl.minimum(lo + chunk, vocab)

    best_score = float("-inf")
    best_index = 0
    for start in range(lo, hi, BLOCK):
        if GEN_NOISE:
            # One Philox call yields 4 lanes; each serves one QUARTER-sized
            # sub-block, so RNG cost is 1/4 of per-element tl.rand. `chunk` is
            # BLOCK-aligned and vocab_padded4 is a multiple of 4, so the
            # quarter offsets are globally unique across rows and splits.
            QUARTER: tl.constexpr = BLOCK // 4
            k = tl.arange(0, QUARTER)
            quarter_base = ((row * vocab_padded4 + start) // 4).to(tl.int32)
            u0, u1, u2, u3 = tl.rand4x(seed, quarter_base + k)
            for j in tl.static_range(4):
                if j == 0:
                    u = u0
                elif j == 1:
                    u = u1
                elif j == 2:
                    u = u2
                else:
                    u = u3
                offs = start + j * QUARTER + k
                mask = offs < hi
                p = tl.load(p_base + offs, mask=mask, other=0.0).to(tl.float32)
                q = tl.maximum(-tl.log(1.0 - u), TINY)
                s = tl.where(mask, p / q, float("-inf"))
                blk_best = tl.max(s, axis=0)
                blk_arg = start + j * QUARTER + tl.argmax(s, axis=0)
                take = (blk_best > best_score) | (
                    (blk_best == best_score) & (blk_arg < best_index)
                )
                best_index = tl.where(take, blk_arg, best_index)
                best_score = tl.where(take, blk_best, best_score)
        else:
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hi
            p = tl.load(p_base + offs, mask=mask, other=0.0).to(tl.float32)
            q = tl.load(noise_ptr + row * noise_stride + offs, mask=mask, other=1.0)
            q = tl.maximum(q, TINY)
            s = tl.where(mask, p / q, float("-inf"))
            blk_best = tl.max(s, axis=0)
            blk_arg = start + tl.argmax(s, axis=0)
            take = (blk_best > best_score) | (
                (blk_best == best_score) & (blk_arg < best_index)
            )
            best_index = tl.where(take, blk_arg, best_index)
            best_score = tl.where(take, blk_best, best_score)

    out = row * SPLITS_CEIL + split
    tl.store(partial_scores_ptr + out, best_score)
    tl.store(partial_indices_ptr + out, best_index.to(tl.int64))

    # Split-K last-block reduction: acq_rel orders this program's partial
    # stores before the count (release) and the winner's partial loads after
    # it (acquire).
    done = tl.atomic_add(counters_ptr + row, 1, sem="acq_rel")
    if done == num_splits - 1:
        offs = tl.arange(0, SPLITS_CEIL)
        live = offs < num_splits
        scores = tl.load(
            partial_scores_ptr + row * SPLITS_CEIL + offs,
            mask=live,
            other=float("-inf"),
            volatile=True,
        )
        indices = tl.load(
            partial_indices_ptr + row * SPLITS_CEIL + offs,
            mask=live,
            other=0,
            volatile=True,
        )
        top = tl.max(scores, axis=0)
        big = tl.zeros_like(indices) + 0x7FFFFFFFFFFFFFFF
        top_index = tl.min(tl.where(live & (scores == top), indices, big), axis=0)
        tl.store(sample_index_ptr + row, top_index)
        tl.store(sample_p_ptr + row, tl.load(p_base + top_index))
        # Self-clean so the next launch (or graph replay) starts from zero.
        tl.store(counters_ptr + row, 0)
        if row == 0 and GEN_NOISE:
            # Single writer; late-starting programs of this launch that read
            # the bumped value still draw from a valid, distinct Philox stream.
            tl.store(rng_state_ptr, tl.load(rng_state_ptr) + 1)


def gumbel_argmax_sample(probs: torch.Tensor, noise: Optional[torch.Tensor] = None):
    """One Gumbel-max draw per row: argmax(probs / max(q, tiny)), q ~ Exp(1).

    Returns ``(sample_p, sample_index)`` shaped [bs, 1] — ``sample_p`` in
    ``probs``' dtype, ``sample_index`` int64 — matching
    ``probs.gather(1, sample_index)`` of the unfused path. Noise is Philox
    in-register by default; pass fp32 ``noise`` to read it from memory
    (deterministic given the noise — the test path).
    """
    assert probs.ndim == 2 and probs.stride(1) == 1
    bs, vocab = probs.shape
    sample_p = torch.empty((bs, 1), dtype=probs.dtype, device=probs.device)
    sample_index = torch.empty((bs, 1), dtype=torch.int64, device=probs.device)
    if bs == 0 or vocab == 0:
        return sample_p, sample_index
    if noise is None:
        noise_arg, noise_stride = probs, 0
    else:
        assert noise.shape == probs.shape and noise.dtype == torch.float32
        assert noise.stride(1) == 1
        noise_arg, noise_stride = noise, noise.stride(0)

    partial_scores, partial_indices, counters, rng_state = _workspace(probs.device, bs)
    # One BLOCK per program whenever vocab fits in _MAX_SPLITS blocks: at
    # decode sizes every program then runs a single load round (the kernel is
    # latency-bound, not bandwidth-bound). BLOCK alignment keeps the Philox
    # quarter offsets globally unique (see kernel).
    chunk = triton.cdiv(triton.cdiv(vocab, _BLOCK), _MAX_SPLITS) * _BLOCK
    num_splits = triton.cdiv(vocab, chunk)
    _gumbel_argmax_kernel[(bs, num_splits)](
        probs,
        noise_arg,
        partial_scores,
        partial_indices,
        counters,
        rng_state,
        sample_p,
        sample_index,
        vocab,
        triton.cdiv(vocab, 4) * 4,
        probs.stride(0),
        noise_stride,
        chunk,
        num_splits,
        SPLITS_CEIL=_MAX_SPLITS,
        GEN_NOISE=noise is None,
        TINY=torch.finfo(torch.float32).tiny,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return sample_p, sample_index
