"""Triton kernels for the LiLiCorr candidate-lattice reranker.

lilicorr_topk_lse returns an exact per-row top-k over the candidate vocab logits and
the full-vocab log-partition from one pass over [n, V].

lilicorr_sample_path commits one path through the candidate lattice and emits the
per-slot proposal verify needs to accept it by rejection sampling; lilicorr_greedy_path
is that walk with every row greedy. Both adapt the head's factors to
selector_walk_triton, DFlash2's candidate selector, and dispatch to a value-identical
torch implementation off CUDA.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dflash import selector_walk_triton

# Tile width for the scan, and tiles per program (TILE * TPP is the load block).
# TILE is the contiguous dimension, so a wide TILE is what turns the scan into
# long coalesced loads; 1024 x 8 = 8192 elements per program. These are launch
# geometry for an H100-class vocabulary head, not a property of the method.
_TILE = 1024
_TILES_PER_PROGRAM = 8
_NUM_WARPS = 4

# Widest candidate pool the selector walk holds in one lane group. Exported because the
# config refuses a head wider than this rather than serving it on the torch path; the
# two must not drift.
MAX_FUSED_CANDIDATE_TOPK = 16

_NEG = -3.0e38


@triton.jit
def _tiled_scan(
    lp,  # [N, V] logits (any float dtype)
    tmax,  # [N, T] fp32 per-tile max
    pm,  # [N, P] fp32 online-softmax running max
    ps,  # [N, P] fp32 online-softmax running sum
    V,
    T,
    stride_lp_n,
    stride_tmax_n,
    stride_p_n,
    TILE: tl.constexpr,
    TPP: tl.constexpr,
):
    """One pass over [N, V]: per-tile maxima plus one (m, s) partial per program.

    grid = (N, cdiv(T, TPP)). Each program covers TPP contiguous tiles, so the block is
    [TPP, TILE] and the tile maxima are a single axis-1 reduction.
    """
    neg = -3.0e38
    row = tl.program_id(0)
    grp = tl.program_id(1)

    t0 = grp * TPP
    tile_off = t0 + tl.arange(0, TPP)  # [TPP]
    offs = tile_off[:, None] * TILE + tl.arange(0, TILE)[None, :]
    mask = (tile_off[:, None] < T) & (offs < V)

    x = tl.load(lp + row * stride_lp_n + offs, mask=mask, other=neg)
    x = x.to(tl.float32)

    tm = tl.max(x, axis=1)  # [TPP]
    tl.store(tmax + row * stride_tmax_n + tile_off, tm, mask=tile_off < T)

    # Online-softmax partial for this program's whole block.
    m = tl.max(tm, axis=0)
    s = tl.sum(tl.exp(x - m), axis=0)
    s = tl.sum(s, axis=0)
    tl.store(pm + row * stride_p_n + grp, m)
    tl.store(ps + row * stride_p_n + grp, s)


@triton.jit
def _tiled_select(
    lp,  # [N, V] logits
    tids,  # [N, KT] int64 candidate tile ids (ascending)
    ov,  # [N, K] fp32 out vals (descending)
    oi,  # [N, K] int64 out ids
    V,
    stride_lp_n,
    stride_tid_n,
    stride_ov_n,
    stride_oi_n,
    TILE: tl.constexpr,
    KT: tl.constexpr,
    K: tl.constexpr,
    NSEL: tl.constexpr,
):
    """Exact top-K over the KT selected tiles of one row, in registers."""
    neg = -3.0e38
    row = tl.program_id(0)

    tid = tl.load(tids + row * stride_tid_n + tl.arange(0, KT))  # [KT]
    offs2 = tid[:, None] * TILE + tl.arange(0, TILE)[None, :]  # [KT, TILE]
    mask2 = offs2 < V
    x2 = tl.load(lp + row * stride_lp_n + offs2, mask=mask2, other=neg)

    x = tl.reshape(x2.to(tl.float32), (NSEL,))
    ids = tl.reshape(offs2, (NSEL,))
    lane = tl.arange(0, NSEL)

    for j in range(K):
        mv = tl.max(x, axis=0)
        mp = tl.argmax(x, axis=0)
        mid = tl.sum(tl.where(lane == mp, ids, 0), axis=0)
        tl.store(ov + row * stride_ov_n + j, mv)
        tl.store(oi + row * stride_oi_n + j, mid)
        x = tl.where(lane == mp, neg, x)


def _combine_lse(pm: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    gm = pm.max(dim=-1, keepdim=True).values
    s = (ps * (pm - gm).exp()).sum(dim=-1)
    return gm.squeeze(-1) + s.log()


def _topk_lse_torch(
    logits: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Value-identical reference for the tiled path, and the fallback off CUDA.
    vals, ids = torch.topk(logits, k, dim=-1)
    rowmax = vals[:, 0:1]  # topk is sorted descending
    sumexp = (logits - rowmax).exp().sum(dim=-1, dtype=torch.float32)
    lse = rowmax.squeeze(-1).to(torch.float32) + sumexp.log()
    return vals.float(), ids.to(torch.int64), lse


def lilicorr_topk_lse(
    logits: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact per-row top-k and full-vocab logsumexp from one [N, V] read.

    Returns (vals [N, k] fp32 descending, tokens [N, k] int64, lse [N] fp32). The
    candidate log-prob the head consumes is val - lse.

    The tile pre-selection is exact, not approximate. Let e be a member of the row's true
    top-k, lying in tile T; then max(T) >= e. If T were not among the k tiles with the
    largest max, k other tiles would each hold an element >= max(T) >= e, so e has rank
    > k, a contradiction.

    Exactness is about the returned values. Which of an exactly-tied set is returned is
    unspecified here and in CUDA torch.topk alike, so the two can select different token
    ids for the same row; the tied candidates carry equal log-probs.

    k must be a power of two to take the tiled path, enforced at config parse on
    lilicorr_candidate_topk.
    """
    if not logits.is_cuda:
        return _topk_lse_torch(logits, k)

    logits = logits.contiguous()
    n, V = int(logits.shape[0]), int(logits.shape[1])
    K = int(k)
    T = (V + _TILE - 1) // _TILE
    if min(K, T) & (min(K, T) - 1):
        # With k validated at config parse, this can only mean the vocabulary spans
        # fewer than k tiles, where the pre-selection is vacuous anyway.
        return _topk_lse_torch(logits, k)
    P = (T + _TILES_PER_PROGRAM - 1) // _TILES_PER_PROGRAM
    device = logits.device

    tmax = torch.full((n, T), _NEG, dtype=torch.float32, device=device)
    pm = torch.full((n, P), _NEG, dtype=torch.float32, device=device)
    ps = torch.zeros((n, P), dtype=torch.float32, device=device)
    _tiled_scan[(n, P)](
        logits,
        tmax,
        pm,
        ps,
        V,
        T,
        logits.stride(0),
        tmax.stride(0),
        pm.stride(0),
        TILE=_TILE,
        TPP=_TILES_PER_PROGRAM,
        num_warps=_NUM_WARPS,
    )

    kt = min(K, T)
    tids = torch.topk(tmax, kt, dim=-1).indices
    tids = torch.sort(tids, dim=-1).values.to(torch.int64)

    ov = torch.empty((n, K), dtype=torch.float32, device=device)
    oi = torch.empty((n, K), dtype=torch.int64, device=device)
    _tiled_select[(n,)](
        logits,
        tids,
        ov,
        oi,
        V,
        logits.stride(0),
        tids.stride(0),
        ov.stride(0),
        oi.stride(0),
        TILE=_TILE,
        KT=kt,
        K=K,
        NSEL=kt * _TILE,
        num_warps=_NUM_WARPS,
    )
    return ov, oi, _combine_lse(pm, ps)


def _lattice_scores(log_start: torch.Tensor, log_pair: torch.Tensor) -> torch.Tensor:
    """Pack the head's factors into the selector's [bs, slots, K, K] layout.

    log_start [bs, K] is slot 0 and log_pair [bs, slots-1, K, K] is every slot after it.
    Slot 0 has no predecessor, so the walk only reads scores[:, 0, 0, :] there and the
    broadcast over the "from" axis is sound. fp32 because the commit's argmax runs on
    these values.
    """
    topk = int(log_start.shape[-1])
    start = log_start.float()[:, None, None, :].expand(-1, 1, topk, topk)
    return torch.cat([start, log_pair.float()], dim=1)


def _selector_walk_torch(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Value-identical reference for selector_walk_triton, and the arm off CUDA. Both
    # picks are computed for every row and selected rather than branched on, because the
    # rows of one batch disagree.
    _, num_slots, topk = candidate_ids.shape
    temps = temperatures.view(-1, 1).to(torch.float32)
    greedy = greedy_mask.view(-1)
    previous = torch.zeros(greedy.shape[0], dtype=torch.int64, device=scores.device)
    tokens, q_rows = [], []
    for slot in range(num_slots):
        node = torch.gather(
            scores[:, slot].float(), 1, previous.view(-1, 1, 1).expand(-1, 1, topk)
        ).squeeze(1)
        probs = torch.softmax(node / temps, dim=-1)
        sampled = (
            uniforms[:, slot : slot + 1]
            .ge(probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(topk - 1)
        )
        previous = torch.where(greedy, node.argmax(dim=-1), sampled)
        q_rows.append(
            torch.where(
                greedy.unsqueeze(-1),
                F.one_hot(previous, topk).to(torch.float32),
                probs,
            )
        )
        tokens.append(
            torch.gather(candidate_ids[:, slot], 1, previous.view(-1, 1)).squeeze(1)
        )
    # int64 tokens, because that is selector_walk_triton's output contract.
    return torch.stack(tokens, dim=-1).to(torch.int64), torch.stack(q_rows, dim=1)


def lilicorr_sample_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-to-right commit over the candidate lattice, plus the proposal it used.

    Shapes (single block): log_start [bs, k], log_pair [bs, slots-1, k, k],
    candidate_tokens [bs, slots, k], uniforms [bs, slots], temperatures [bs] and
    greedy_mask [bs]. Returns (tokens [bs, slots], q_rows [bs, slots, k] fp32).
    q_rows[b, s] is the distribution slot s was drawn from, over that slot's k candidates
    and zero elsewhere; the caller scatters it into the dense q the verify kernel reads.

    The proposal is softmax(psi_s / T), psi_s being the head's own log-factor row: the
    start factor at slot 0 and the transition row out of the committed predecessor after
    it. No other term, because the head was trained with no unary factor and no log-prob
    prior.

    Rows with greedy_mask set take the argmax and report a point mass, so a mixed batch
    is one launch and the greedy rows are unchanged.

    The walk itself is DFlash2's candidate selector: same recurrence, same lower-index
    tie break, same inverse-CDF draw. Fixed trip count and no host syncs, so the draft
    CUDA graph can capture it.
    """
    scores = _lattice_scores(log_start, log_pair)
    walk = selector_walk_triton if scores.is_cuda else _selector_walk_torch
    return walk(
        candidate_ids=candidate_tokens,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )


def lilicorr_greedy_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    """lilicorr_sample_path with every row greedy, for a batch carrying no sampling state.

    Commits the argmax candidate at each slot conditioned on the previously committed
    pick, and returns the selected tokens [bs, slots]. Shares the walk, so the commit is
    bit-identical to the greedy rows of a mixed sampled batch.
    """
    bsz, num_slots, _ = candidate_tokens.shape
    device = log_start.device
    tokens, _ = lilicorr_sample_path(
        log_start,
        log_pair,
        candidate_tokens,
        uniforms=torch.zeros(bsz, num_slots, dtype=torch.float32, device=device),
        temperatures=torch.ones(bsz, dtype=torch.float32, device=device),
        greedy_mask=torch.ones(bsz, dtype=torch.bool, device=device),
    )
    return tokens
