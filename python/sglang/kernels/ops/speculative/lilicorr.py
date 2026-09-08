"""Triton kernels for the LiLiCorr candidate-lattice reranker.

``lilicorr_topk_lse`` returns an exact per-row top-k over the candidate vocab
logits and the full-vocab log-partition from one pass over ``[n, V]``; a separate
``topk`` plus a ``logsumexp`` epilogue reads the vocabulary three times and
materializes two more ``[n, V]`` temporaries, and that read is the single largest
cost in the block. ``lilicorr_greedy_path`` folds the whole left-to-right commit
into one launch, where the torch form issues roughly three kernels per slot over
``[bs, k]`` tensors and is pure launch overhead rather than arithmetic.
``lilicorr_sample_path`` is that same walk with a sampled commit, emitting the
per-slot proposal the verify needs to accept it by rejection sampling.

Both dispatch to a value-identical torch implementation off CUDA, which is also
what makes the head exercisable in a CPU unit test.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Tile width for the scan, and tiles per program (TILE * TPP is the load block).
# TILE is the contiguous dimension, so a wide TILE is what turns the scan into
# long coalesced loads; 1024 x 8 = 8192 elements per program. These are launch
# geometry for an H100-class vocabulary head, not a property of the method.
_TILE = 1024
_TILES_PER_PROGRAM = 8
_NUM_WARPS = 4

# Widest candidate pool the fused greedy commit holds in one lane group. Exported
# because the config refuses a head wider than this rather than serving it on the
# torch path; the two must not drift.
MAX_FUSED_CANDIDATE_TOPK = 16
_GREEDY_MAX_K = MAX_FUSED_CANDIDATE_TOPK

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

    grid = (N, cdiv(T, TPP)). Each program covers TPP contiguous tiles, so the
    block is [TPP, TILE] and the tile maxima are a single axis-1 reduction.
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


@triton.jit
def _greedy_path_kernel(
    ls_ptr,
    lp_ptr,
    ids_ptr,
    out_ptr,
    ls_sb,
    lp_sb,
    lp_ss,
    lp_sp,
    ids_sb,
    ids_ss,
    out_sb,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """The whole greedy path for one batch row, tokens included.

    ``c_0 = argmax(log_start)``, then ``c_s = argmax_c pair[s-1, c_{s-1}, c]``.
    ``tl.argmax`` breaks ties toward the lower index as ``Tensor.argmax`` does, so
    the committed path matches the torch reference. ``pair`` layout: dim -2
    ("from") has stride ``lp_sp``, dim -1 ("to") is unit-stride.
    """
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    neg = -3.0e38
    node = tl.load(ls_ptr + b * ls_sb + offs, mask=mask, other=neg)
    node = tl.where(mask, node, neg)
    cur = tl.argmax(node, axis=0).to(tl.int32)
    tl.store(out_ptr + b * out_sb, tl.load(ids_ptr + b * ids_sb + cur))
    for s in range(1, S):
        trans = tl.load(
            lp_ptr + b * lp_sb + (s - 1) * lp_ss + cur * lp_sp + offs,
            mask=mask,
            other=neg,
        )
        node = tl.where(mask, trans, neg)
        cur = tl.argmax(node, axis=0).to(tl.int32)
        tl.store(
            out_ptr + b * out_sb + s,
            tl.load(ids_ptr + b * ids_sb + s * ids_ss + cur),
        )


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
    """Exact per-row top-k and full-vocab logsumexp from one ``[N, V]`` read.

    Returns ``(vals [N, k] fp32 descending, tokens [N, k] int64, lse [N] fp32)``.
    The candidate log-prob the head consumes is ``val - lse``.

    The tile pre-selection is exact, not approximate. Let ``e`` be a member of the
    row's true top-k, lying in tile ``T``; then ``max(T) >= e``. If ``T`` were not
    among the k tiles with the largest max, k other tiles would each hold an
    element ``>= max(T) >= e``, so ``e`` has rank > k -- a contradiction.

    Exactness is about the returned *values*. Which of an exactly-tied set of
    values is returned is unspecified here and in CUDA ``torch.topk`` alike, so
    the two can select different token ids for the same row -- reachable on bf16
    logits, where a wide vocabulary rounds several entries together. The tied
    candidates carry equal log-probs, so the head scores an equally-ranked
    alternative rather than a wrong one.

    ``k`` must be a power of two to take the tiled path, which is enforced at
    config parse on ``lilicorr_candidate_topk``.
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


def _greedy_path_torch(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    num_slots = int(candidate_tokens.shape[1])
    topk = int(log_start.shape[-1])
    cur = log_start.argmax(dim=-1)
    cols = [cur]
    for slot in range(1, num_slots):
        trans = torch.gather(
            log_pair[:, slot - 1, :, :],
            1,
            cur.view(-1, 1, 1).expand(-1, 1, topk),
        ).squeeze(1)
        cur = trans.argmax(dim=-1)
        cols.append(cur)
    path = torch.stack(cols, dim=-1)
    return torch.gather(candidate_tokens, 2, path.unsqueeze(-1)).squeeze(-1)


def _as_fp32_unit_last(t: torch.Tensor) -> torch.Tensor:
    # The kernel reads the candidate (last) dim contiguously and every other dim
    # through a passed stride, so a unit-stride last dim is all it needs.
    if t.dtype != torch.float32:
        t = t.float()
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


def lilicorr_greedy_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    """Locally-optimal left-to-right decode of the candidate lattice.

    Shapes (single block): ``log_start [bs, k]``, ``log_pair [bs, slots-1, k,
    k]``, ``candidate_tokens [bs, slots, k]``. Returns the selected tokens ``[bs,
    slots]``.

    Commits the argmax candidate at each slot conditioned on the previously
    committed pick, which matches the locally-normalized training objective:
    under prefix acceptance, once a slot is wrong nothing after it is accepted,
    so the global MAP's freedom to trade an early slot for a richer tail has no
    value. Fixed trip count and no host syncs, so it is CUDA-graph safe.
    """
    if not (log_start.is_cuda and log_start.shape[-1] <= _GREEDY_MAX_K):
        return _greedy_path_torch(log_start, log_pair, candidate_tokens)

    bsz, num_slots, k = candidate_tokens.shape
    ls = _as_fp32_unit_last(log_start)
    lp = _as_fp32_unit_last(log_pair)
    ids = candidate_tokens
    if ids.stride(-1) != 1:
        ids = ids.contiguous()
    out = torch.empty(bsz, num_slots, dtype=ids.dtype, device=ls.device)
    _greedy_path_kernel[(bsz,)](
        ls,
        lp,
        ids,
        out,
        ls.stride(0),
        lp.stride(0),
        lp.stride(1),
        lp.stride(2),
        ids.stride(0),
        ids.stride(1),
        out.stride(0),
        S=num_slots,
        K=k,
        BLOCK_K=max(_GREEDY_MAX_K, triton.next_power_of_2(k)),
    )
    return out


@triton.jit
def _sample_path_kernel(
    ls_ptr,
    lp_ptr,
    ids_ptr,
    u_ptr,
    temp_ptr,
    greedy_ptr,
    out_ptr,
    q_ptr,
    ls_sb,
    lp_sb,
    lp_ss,
    lp_sp,
    ids_sb,
    ids_ss,
    u_sb,
    out_sb,
    q_sb,
    q_ss,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``_greedy_path_kernel`` plus the proposal it sampled from.

    Same recurrence and the same loads; the only additions are the per-slot draw and
    the ``q`` row the verify needs to run rejection sampling. A row with
    ``greedy_mask`` set takes ``tl.argmax`` on the same fp32 node values as the greedy
    kernel, so one captured graph serves greedy and sampling batches and the greedy
    rows walk a bit-identical path.

    ``q`` for a greedy row is the point mass at its pick, not the temperature softmax:
    that row's token was chosen deterministically, and telling verify anything else
    would make ``min(1, p/q)`` the wrong acceptance test for it.
    """
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    neg = -3.0e38
    temperature = tl.load(temp_ptr + b)
    greedy = tl.load(greedy_ptr + b) != 0

    node = tl.load(ls_ptr + b * ls_sb + offs, mask=mask, other=neg)
    node = tl.where(mask, node, neg)
    if greedy:
        cur = tl.argmax(node, axis=0).to(tl.int32)
        probs = tl.where(offs == cur, 1.0, 0.0)
    else:
        scaled = node / temperature
        expo = tl.exp(scaled - tl.max(scaled, axis=0))
        probs = expo / tl.sum(expo, axis=0)
        uniform = tl.load(u_ptr + b * u_sb)
        cur = tl.sum(tl.where(uniform >= tl.cumsum(probs, axis=0), 1, 0), axis=0)
        cur = tl.minimum(cur, K - 1).to(tl.int32)
    tl.store(q_ptr + b * q_sb + offs, probs, mask=mask)
    tl.store(out_ptr + b * out_sb, tl.load(ids_ptr + b * ids_sb + cur))

    for s in range(1, S):
        node = tl.load(
            lp_ptr + b * lp_sb + (s - 1) * lp_ss + cur * lp_sp + offs,
            mask=mask,
            other=neg,
        )
        node = tl.where(mask, node, neg)
        if greedy:
            cur = tl.argmax(node, axis=0).to(tl.int32)
            probs = tl.where(offs == cur, 1.0, 0.0)
        else:
            scaled = node / temperature
            expo = tl.exp(scaled - tl.max(scaled, axis=0))
            probs = expo / tl.sum(expo, axis=0)
            uniform = tl.load(u_ptr + b * u_sb + s)
            cur = tl.sum(tl.where(uniform >= tl.cumsum(probs, axis=0), 1, 0), axis=0)
            cur = tl.minimum(cur, K - 1).to(tl.int32)
        tl.store(q_ptr + b * q_sb + s * q_ss + offs, probs, mask=mask)
        tl.store(
            out_ptr + b * out_sb + s,
            tl.load(ids_ptr + b * ids_sb + s * ids_ss + cur),
        )


def _sample_path_torch(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Value-identical reference for the fused walk, and the fallback off CUDA. The
    # greedy and sampling picks are computed for every row and selected, rather than
    # branched on, because the rows of one batch disagree.
    num_slots = int(candidate_tokens.shape[1])
    topk = int(log_start.shape[-1])
    temps = temperatures.view(-1, 1).to(torch.float32)
    greedy = greedy_mask.view(-1)
    idx_cols, q_cols = [], []
    node = log_start.float()
    for slot in range(num_slots):
        if slot > 0:
            node = torch.gather(
                log_pair[:, slot - 1, :, :].float(),
                1,
                idx_cols[-1].view(-1, 1, 1).expand(-1, 1, topk),
            ).squeeze(1)
        probs = torch.softmax(node / temps, dim=-1)
        sampled = (
            uniforms[:, slot : slot + 1]
            .ge(probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(topk - 1)
        )
        picked = node.argmax(dim=-1)
        index = torch.where(greedy, picked, sampled)
        idx_cols.append(index)
        q_cols.append(
            torch.where(
                greedy.unsqueeze(-1),
                F.one_hot(index, topk).to(torch.float32),
                probs,
            )
        )
    path = torch.stack(idx_cols, dim=-1)
    tokens = torch.gather(candidate_tokens, 2, path.unsqueeze(-1)).squeeze(-1)
    return tokens, torch.stack(q_cols, dim=1)


def lilicorr_sample_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``lilicorr_greedy_path`` with a sampled commit, plus the proposal it used.

    Returns ``(tokens [bs, slots], q_rows [bs, slots, k] fp32)``. ``q_rows[b, s]`` is
    the distribution slot ``s`` was drawn from, over that slot's ``k`` candidates and
    zero everywhere else; the caller scatters it into the dense ``q`` the verify
    kernel reads.

    The proposal is ``softmax(psi_s / T)`` where ``psi_s`` is the head's own log-factor
    row -- the start factor at slot 0 and the transition row out of the committed
    predecessor after it. There is deliberately no other term: the shipped
    configuration ablates the unary factor away and carries no log-prob prior, so
    adding either here would serve a scoring function the head was not trained with.

    Rows with ``greedy_mask`` set take the argmax and report a point mass, so a mixed
    batch is one launch and the greedy rows are unchanged. As ``T -> 0`` the sampled
    rows converge on that same argmax, which is what makes this a superset of the
    greedy path rather than a different drafter.
    """
    if not (log_start.is_cuda and log_start.shape[-1] <= _GREEDY_MAX_K):
        return _sample_path_torch(
            log_start, log_pair, candidate_tokens, uniforms, temperatures, greedy_mask
        )

    bsz, num_slots, k = candidate_tokens.shape
    ls = _as_fp32_unit_last(log_start)
    lp = _as_fp32_unit_last(log_pair)
    u = _as_fp32_unit_last(uniforms)
    temps = temperatures.reshape(-1).to(torch.float32).contiguous()
    greedy = greedy_mask.reshape(-1).contiguous()
    ids = candidate_tokens
    if ids.stride(-1) != 1:
        ids = ids.contiguous()
    out = torch.empty(bsz, num_slots, dtype=ids.dtype, device=ls.device)
    q_rows = torch.empty(bsz, num_slots, k, dtype=torch.float32, device=ls.device)
    _sample_path_kernel[(bsz,)](
        ls,
        lp,
        ids,
        u,
        temps,
        greedy,
        out,
        q_rows,
        ls.stride(0),
        lp.stride(0),
        lp.stride(1),
        lp.stride(2),
        ids.stride(0),
        ids.stride(1),
        u.stride(0),
        out.stride(0),
        q_rows.stride(0),
        q_rows.stride(1),
        S=num_slots,
        K=k,
        BLOCK_K=max(_GREEDY_MAX_K, triton.next_power_of_2(k)),
        num_warps=1,
    )
    return out, q_rows
