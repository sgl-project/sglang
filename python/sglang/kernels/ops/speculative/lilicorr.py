from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dflash import selector_walk_triton

# Launch geometry tuned on H100 for a ~150k vocabulary; TILE * TPP is the load block.
_TILE = 1024
_TILES_PER_PROGRAM = 8
_NUM_WARPS = 4

# One Triton lane group in the selector walk; the config refuses a wider head.
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
    neg = -3.0e38
    row = tl.program_id(0)
    grp = tl.program_id(1)

    t0 = grp * TPP
    tile_off = t0 + tl.arange(0, TPP)
    offs = tile_off[:, None] * TILE + tl.arange(0, TILE)[None, :]
    mask = (tile_off[:, None] < T) & (offs < V)

    x = tl.load(lp + row * stride_lp_n + offs, mask=mask, other=neg)
    x = x.to(tl.float32)

    tm = tl.max(x, axis=1)
    tl.store(tmax + row * stride_tmax_n + tile_off, tm, mask=tile_off < T)

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
    neg = -3.0e38
    row = tl.program_id(0)

    tid = tl.load(tids + row * stride_tid_n + tl.arange(0, KT))
    offs2 = tid[:, None] * TILE + tl.arange(0, TILE)[None, :]
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
    vals, ids = torch.topk(logits, k, dim=-1)
    rowmax = vals[:, 0:1]  # topk is sorted descending
    sumexp = (logits - rowmax).exp().sum(dim=-1, dtype=torch.float32)
    lse = rowmax.squeeze(-1).to(torch.float32) + sumexp.log()
    return vals.float(), ids.to(torch.int64), lse


def lilicorr_topk_lse(
    logits: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Tile pre-selection is exact: a top-k element's tile has a max at least as large,
    # so it is among the k largest-max tiles.
    if not logits.is_cuda:
        return _topk_lse_torch(logits, k)

    logits = logits.contiguous()
    n, V = int(logits.shape[0]), int(logits.shape[1])
    K = int(k)
    T = (V + _TILE - 1) // _TILE
    if min(K, T) & (min(K, T) - 1):
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
    # The walk reads only scores[:, 0, 0, :] at slot 0, so broadcasting is sound.
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
    return torch.stack(tokens, dim=-1).to(torch.int64), torch.stack(q_rows, dim=1)


def lilicorr_sample_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # No log-prob prior in the proposal: the head was trained without one. Greedy rows
    # report a point mass so min(1, p/q) stays the right acceptance test.
    scores = _lattice_scores(log_start, log_pair)
    walk = selector_walk_triton if scores.is_cuda else _selector_walk_torch
    return walk(
        candidate_ids=candidate_tokens,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
