# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Vocab-parallel DFLASH sampled accept and target argmax from tensor-parallel vocab shards.

The served sampled accept (``selector_logit_accept``) and the forced-acceptance simulator's target argmax read the
full softcapped verify logits, so every round all-gathers [rows, vocab] fp32 logits across the TP group. Here each
rank reduces its own vocab shard to a small per-row payload, the group all-gathers the payloads ([world, rows, P]
fp32, P = 6 + top_k), and every rank combines them into the same result:

- ``target_predict``: per row the argmax of the softcapped logits, lowest global index on ties. Same softcap
  instructions as ``gathered_softcap_logits`` and ``softcap_row_argmax``, so it equals ``torch.argmax`` over the
  gathered logits bit for bit (NaN logits excepted).
- the chain walk on the served uniforms, with p at the drafts from the global softmax stats
  M = max_r m_r and Z = sum_r s_r * exp(m_r - M) (p differs from the served p only by the fp32 order of Z).
- the residual / bonus draw by the Gumbel-max trick. Outside the selector's candidate ids q = 0, so the residual
  weight is p_v and log p_v = l_v * inv_temp - M - log Z: the argmax of (l_v * inv_temp + G_v) over a shard's
  non-candidate columns needs no global statistics. The candidate ids are scored in the combine as
  log(max(p - q, 0)) + G. All drafts accepted: the argmax of (l_v * inv_temp + G_v) over every column.

The draw is distribution-identical to the served inverse-CDF draw but uses different randomness, so sampled tokens
are not token-identical to the served sampler. The Gumbel noise is Philox(seed, row * vocab + global id) with seed =
the bits of the request's final uniform, the draw the served sampler reads there, so the torch RNG stream advances
exactly as served.

Cross-rank guard: the combine reads other ranks' payloads, which those ranks built from THEIR seed, candidate ids,
q, drafts, coins and temperatures, so a rank whose inputs drift would give a silently wrong law (the selector sync
cannot repair that). Each payload row therefore carries the seed bits and a 32-bit hash of the row's other inputs;
the combine compares every rank's two fields with rank 0's and, when given a counter, adds the requests that differ
(int32 [2]: seed, hash; an atomic only on a mismatch, so the results never depend on it).

Exchange (the first form measured slower per round at TP16): the payload all-gather
runs on the served multimem all-gather kernel (VpPayloadGather; faster per round than the NCCL ring) and each rank's shard
is cut into column chunks, one local program per (row, chunk) (vp_split). Every chunk
writes its own payload entry, so the combine sees world * splits virtual ranks in ascending vocab order; the target
argmax, the Gumbel winner and the candidate logits do not depend on the split, the softmax denominator's FP32 order does.

Sync fold (SGLANG_OPT_DFLASH_SELECTOR_SYNC_FOLD, as selector_sampled_accept_synced): given ``sync_out``, the combine
also writes SpecTpSync's padded fp32 all-reduce input (this rank's (num_correct, bonus) or zeros) and
``commit_from_fp32_sync`` commits from the all-reduce output, so the sync needs no zeros / copy / copy-back kernels.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.speculative.selector_sampled_accept import _commit_block_f32_kernel

# argmax value, argmax id, m, s, gumbel score, gumbel id, seed bits, input hash (int32 bits); then top_k candidate logits
_PAY_FIXED = 8
_BLOCK = 2048
_BIG = tl.constexpr(2147483647)
# Column split (SGLANG_OPT_K31_VP_SPLIT): each rank's shard is cut into `splits` column chunks, one program per
# (row, chunk), so a B1 verify (8 rows) runs ~128 programs instead of 8. Every chunk writes its own payload entry; the
# combine treats the world * splits entries as virtual ranks in ascending vocab order. The split is a function of
# (rows, shard width) only, so every rank picks the same one.
_TARGET_PROGRAMS = 128
_MAX_SPLITS = 16
_MIN_CHUNK = 128


def payload_width(top_k: int) -> int:
    return _PAY_FIXED + int(top_k)


def vp_split(rows: int, shard_width: int, forced: int = 0) -> tuple[int, int]:
    """(splits, chunk) for a [rows, shard_width] vocab shard: ``forced`` > 0 fixes the split count, else ~128 programs.
    chunk is a multiple of 64 columns and (splits - 1) * chunk < shard_width <= splits * chunk."""
    s = int(forced) if forced > 0 else max(1, _TARGET_PROGRAMS // max(int(rows), 1))
    s = max(1, min(s, _MAX_SPLITS, -(-int(shard_width) // _MIN_CHUNK)))
    chunk = -(-(-(-int(shard_width) // s)) // 64) * 64
    return -(-int(shard_width) // chunk), chunk


@triton.jit
def _gumbel(seed, offs):
    return -tl.log(-tl.log(tl.rand(seed, offs)))


@triton.jit
def _fmix(x):
    # int32 avalanche (wrapping multiply; the masks make the arithmetic shifts logical). Identical on every rank.
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * 0x45D9F3B
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * 0x45D9F3B
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


@triton.jit
def _softcap(x, softcap, APPLY_SOFTCAP: tl.constexpr):
    # gathered_softcap_logits' instructions, so the values carry its bits.
    if APPLY_SOFTCAP:
        x = x / softcap
        x = libdevice.tanh(x)
        x = x * softcap
    return x


@triton.jit
def _vp_local_kernel(
    lg_ptr,
    temps_ptr,
    ufin_ptr,
    cid_ptr,
    cand_ptr,
    q_ptr,
    uni_ptr,
    out_ptr,
    stride_lg,
    stride_cand,
    stride_uni,
    valid_width,
    col_offset,
    chunk,
    softcap,
    NUM_SLOTS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    P: tl.constexpr,
    SPLITS: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    APPLY_SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    sub = tl.program_id(1)
    # this program's columns [lo, hi) of the shard (empty past the valid width)
    lo = sub * chunk
    hi = tl.minimum(lo + chunk, valid_width)
    req = row // NUM_SLOTS
    j = row % NUM_SLOTS
    inv_temp = 1.0 / tl.load(temps_ptr + req)
    seed = tl.load(ufin_ptr + req).to(tl.int32, bitcast=True)
    has_q = j < NUM_SLOTS - 1
    cid_base = (req * (NUM_SLOTS - 1) + j) * TOPK
    lg_row = lg_ptr + row.to(tl.int64) * stride_lg
    a_val = -float("inf")
    a_idx = 0
    m = -float("inf")
    s = 0.0
    g_val = -float("inf")
    g_idx = 0
    for start in range(lo, hi, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < hi
        # 16-bit shards widen exactly to FP32 first (the served gather's cast).
        x = _softcap(tl.load(lg_row + cols, mask=mask, other=0.0).to(tl.float32), softcap, APPLY_SOFTCAP)
        x = tl.where(mask, x, -float("inf"))
        bv, bi = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
        take = bv > a_val
        a_idx = tl.where(take, start + bi, a_idx)
        a_val = tl.where(take, bv, a_val)
        t = x * inv_temp
        new_m = tl.maximum(m, tl.max(t, axis=0))
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(t - new_m), 0.0), axis=0)
        m = new_m
        gid = cols + col_offset
        is_cand = cols < 0
        for kk in tl.static_range(TOPK):
            c = tl.load(cid_ptr + cid_base + kk, mask=has_q, other=-1)
            is_cand = is_cand | (gid == c)
        gs = tl.where(mask & ~is_cand, t + _gumbel(seed, j * VOCAB + gid), -float("inf"))
        gv, gi = tl.max(gs, axis=0, return_indices=True, return_indices_tie_break_left=True)
        take_g = gv > g_val
        g_idx = tl.where(take_g, start + gi, g_idx)
        g_val = tl.where(take_g, gv, g_val)
    out_row = out_ptr + (row.to(tl.int64) * SPLITS + sub) * P
    tl.store(out_row + 0, a_val)
    tl.store(out_row + 1, (a_idx + col_offset).to(tl.float32))
    tl.store(out_row + 2, m)
    tl.store(out_row + 3, s)
    tl.store(out_row + 4, g_val)
    tl.store(out_row + 5, (g_idx + col_offset).to(tl.float32))
    # The candidate logits this program's columns own (-inf elsewhere): the combine reads the owning entry.
    k = tl.arange(0, TOPK_P2)
    kmask = (k < TOPK) & has_q
    c = tl.load(cid_ptr + cid_base + k, mask=kmask, other=-1)
    local = c - col_offset
    own = kmask & (local >= lo) & (local < hi)
    cl = _softcap(tl.load(lg_row + local, mask=own, other=0.0).to(tl.float32), softcap, APPLY_SOFTCAP)
    tl.store(out_row + 8 + k, tl.where(own, cl, -float("inf")), mask=k < TOPK)
    # Cross-rank guard fields: the seed bits as loaded, and a hash of this row's candidate ids, q, draft (slot j),
    # coin (slot j + 1's) and temperature.
    tl.store(out_row + 6, tl.load(ufin_ptr + req))
    qb = tl.load(q_ptr + cid_base + k, mask=kmask, other=0.0).to(tl.int32, bitcast=True)
    hk = _fmix(c.to(tl.int32) * 0x2C1B3C6D + k) ^ _fmix(qb + k * 0x297A2D39)
    h = tl.sum(tl.where(kmask, hk, 0), axis=0)
    draft = tl.load(cand_ptr + req * stride_cand + j).to(tl.int32)
    h = _fmix(h ^ _fmix(draft + 0x3B9ACA07))
    coin = tl.load(uni_ptr + req * stride_uni + j, mask=has_q, other=0.0).to(tl.int32, bitcast=True)
    h = _fmix(h ^ _fmix(coin + 0x1F123BB5))
    h = _fmix(h ^ tl.load(temps_ptr + req).to(tl.int32, bitcast=True))
    tl.store(out_row + 7, h.to(tl.float32, bitcast=True))


@triton.jit
def _vp_combine_kernel(
    pay_ptr,
    cand_ptr,
    cid_ptr,
    q_ptr,
    uni_ptr,
    ufin_ptr,
    temps_ptr,
    res_ptr,
    tp_ptr,
    cnt_ptr,
    sync_ptr,
    stride_cand,
    stride_uni,
    stride_rank,
    stride_row,
    shard_width,
    chunk,
    vocab_size,
    sync_n,
    sync_padded,
    W: tl.constexpr,
    SPLITS: tl.constexpr,
    W_P2: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    P: tl.constexpr,
    VOCAB: tl.constexpr,
    COUNT: tl.constexpr,
    SYNC: tl.constexpr,
    SYNC_VALUES: tl.constexpr,
):
    pid = tl.program_id(0)
    # w = virtual rank (rank w // SPLITS, column chunk w % SPLITS), ascending in vocab; payload entry (w, row) at
    # (w // SPLITS) * stride_rank + row * stride_row + (w % SPLITS) * P (world-major or row-major gathers).
    w = tl.arange(0, W_P2)
    s = tl.arange(0, SLOTS_P2)
    wm = w < W * SPLITS
    sm = s < NUM_SLOTS
    m2 = wm[:, None] & sm[None, :]
    woff = (w // SPLITS).to(tl.int64) * stride_rank + (w % SPLITS) * P
    # 1) target argmax per row: the best value, then the lowest global id among equal values (ranks hold
    #    ascending vocab ranges and each reports its lowest local id), as torch.argmax over the gathered row.
    rows = pid * NUM_SLOTS + s
    base = woff[:, None] + rows[None, :].to(tl.int64) * stride_row
    av = tl.load(pay_ptr + base, mask=m2, other=-float("inf"))
    ai = tl.load(pay_ptr + base + 1, mask=m2, other=0.0).to(tl.int32)
    best_v = tl.max(av, axis=0)
    best_i = tl.min(tl.where(m2 & (av == best_v[None, :]), ai, _BIG), axis=0)
    tl.store(tp_ptr + rows, best_i.to(tl.int64), mask=sm)

    # 0) cross-rank guard: every rank's seed bits and input hash vs rank 0's (all ranks read the same bytes).
    if COUNT:
        base0 = rows.to(tl.int64) * stride_row
        sd = tl.load(pay_ptr + base + 6, mask=m2, other=0.0).to(tl.int32, bitcast=True)
        sd0 = tl.load(pay_ptr + base0 + 6, mask=sm, other=0.0).to(tl.int32, bitcast=True)
        hs = tl.load(pay_ptr + base + 7, mask=m2, other=0.0).to(tl.int32, bitcast=True)
        hs0 = tl.load(pay_ptr + base0 + 7, mask=sm, other=0.0).to(tl.int32, bitcast=True)
        seed_bad = tl.max(tl.max(tl.where(m2 & (sd != sd0[None, :]), 1, 0), axis=1), axis=0)
        hash_bad = tl.max(tl.max(tl.where(m2 & (hs != hs0[None, :]), 1, 0), axis=1), axis=0)
        tl.atomic_add(cnt_ptr, seed_bad, mask=seed_bad > 0)
        tl.atomic_add(cnt_ptr + 1, hash_bad, mask=hash_bad > 0)

    # 2) chain walk: slot s (1..gamma) tests its draft on row s - 1.
    inv_temp = 1.0 / tl.load(temps_ptr + pid)
    seed = tl.load(ufin_ptr + pid).to(tl.int32, bitcast=True)
    valid = (s >= 1) & sm
    mv = wm[:, None] & valid[None, :]
    prow = pid * NUM_SLOTS + s - 1
    pbase = woff[:, None] + prow[None, :].to(tl.int64) * stride_row
    mm = tl.load(pay_ptr + pbase + 2, mask=mv, other=-float("inf"))
    ss = tl.load(pay_ptr + pbase + 3, mask=mv, other=0.0)
    big_m = tl.max(mm, axis=0)
    big_m = tl.where(valid, big_m, 0.0)
    z = tl.sum(tl.where(mv, ss * tl.exp(mm - big_m[None, :]), 0.0), axis=0)
    z = tl.where(valid, z, 1.0)
    inv_z = 1.0 / z
    k = tl.arange(0, TOPK_P2)
    lane = valid[:, None] & (k[None, :] < TOPK)
    qoffs = (pid * (NUM_SLOTS - 1) + (s[:, None] - 1)) * TOPK + k[None, :]
    cid = tl.load(cid_ptr + qoffs, mask=lane, other=-1)
    qv = tl.load(q_ptr + qoffs, mask=lane, other=0.0)
    # candidate logits [slot, k]: read from the owning entry only (rank cid // shard_width, chunk of the remainder);
    # the other entries hold -inf there.
    own_r = cid // shard_width
    own_c = (cid - own_r * shard_width) // chunk
    own_ok = lane & (cid >= 0) & (own_r < W) & (own_c < SPLITS)
    caddr = own_r.to(tl.int64) * stride_rank + prow[:, None].to(tl.int64) * stride_row + own_c * P + 8 + k[None, :]
    cl = tl.load(pay_ptr + caddr, mask=own_ok, other=-float("inf"))  # [slot, k]
    draft = tl.load(cand_ptr + pid * stride_cand + s, mask=valid, other=0)
    match = lane & (cid == draft[:, None])
    q = tl.sum(tl.where(match, qv, 0.0), axis=1)
    d_logit = tl.max(tl.where(match, cl, -float("inf")), axis=1)
    p = tl.where(valid, tl.exp(d_logit * inv_temp - big_m) * inv_z, 0.0)
    coin = tl.load(uni_ptr + pid * stride_uni + (s - 1), mask=valid, other=1.0)
    ok = (q > 0.0) & (q <= 1.0) & (coin * q < p)
    first_fail = tl.min(tl.where(valid & ~ok, s, NUM_SLOTS), axis=0)
    num_accept = first_fail - 1

    # 3a) all drafts accepted: Gumbel argmax over every column of row gamma.
    lrow = pid * NUM_SLOTS + NUM_SLOTS - 1
    lbase = woff + lrow.to(tl.int64) * stride_row
    gl = tl.load(pay_ptr + lbase + 4, mask=wm, other=-float("inf"))
    gli = tl.load(pay_ptr + lbase + 5, mask=wm, other=0.0).to(tl.int32)
    gl_best = tl.max(gl, axis=0)
    tok_all = tl.min(tl.where(wm & (gl == gl_best), gli, _BIG), axis=0)

    # 3b) rejected at slot sj = num_accept + 1: row num_accept's residual relu(p - q).
    sel = s == (num_accept + 1)
    gr = tl.load(pay_ptr + pbase + 4, mask=mv, other=-float("inf"))
    gri = tl.load(pay_ptr + pbase + 5, mask=mv, other=0.0).to(tl.int32)
    nb_raw = tl.max(gr, axis=0)
    nb_idx = tl.min(tl.where(mv & (gr == nb_raw[None, :]), gri, _BIG), axis=0)
    log_z = tl.log(z)
    nb = nb_raw - big_m - log_z  # log p + G of the best non-candidate column
    q_res = tl.where((qv >= 0.0) & (qv <= 1.0), qv, 0.0)
    p_c = tl.exp(cl * inv_temp - big_m[:, None]) * inv_z[:, None]
    r = p_c - q_res
    noise = _gumbel(seed, ((s[:, None] - 1) * VOCAB + cid).to(tl.int32))
    c_score = tl.where(lane & (r > 0.0), tl.log(r) + noise, -float("inf"))
    c_best = tl.max(c_score, axis=1)
    c_idx = tl.min(tl.where(lane & (c_score == c_best[:, None]), cid, _BIG), axis=1)
    use_c = c_best > nb
    slot_score = tl.where(use_c, c_best, nb)
    slot_tok = tl.where(use_c, c_idx, nb_idx)
    rej_score = tl.max(tl.where(sel, slot_score, -float("inf")), axis=0)
    tok_rej = tl.max(tl.where(sel, slot_tok, -1), axis=0)
    tok_rej = tl.where(rej_score > -float("inf"), tok_rej, vocab_size - 1)
    tok_all = tl.where(gl_best > -float("inf"), tok_all, vocab_size - 1)
    token = tl.where(num_accept == NUM_SLOTS - 1, tok_all, tok_rej)
    tl.store(res_ptr + pid * 2, num_accept.to(tl.int64))
    tl.store(res_ptr + pid * 2 + 1, token.to(tl.int64))
    if SYNC:
        # SpecTpSync's fp32 all-reduce input: this rank's values (the root, or every rank when the sync is off) or
        # zeros; exact integers below 2**24.
        if SYNC_VALUES:
            tl.store(sync_ptr + pid * 2, num_accept.to(tl.float32))
            tl.store(sync_ptr + pid * 2 + 1, token.to(tl.float32))
        else:
            tl.store(sync_ptr + pid * 2, 0.0)
            tl.store(sync_ptr + pid * 2 + 1, 0.0)
        if pid == 0:
            pad = sync_n + tl.arange(0, 4)
            tl.store(sync_ptr + pad, tl.zeros((4,), dtype=tl.float32), mask=pad < sync_padded)


def vp_local_payload(
    *,
    logits_shard: torch.Tensor,
    temperatures: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    candidate_ids: torch.Tensor,
    rows_per_request: int,
    valid_width: int,
    col_offset: int,
    vocab_size: int,
    softcap: Optional[float],
    candidates: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    splits: int = 1,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """This rank's payload fp32 [rows, splits * (8 + top_k)] (row-major: entry (row, chunk)) from its shard of the raw
    (pre-softcap) verify logits, rows request-major with ``rows_per_request`` = gamma + 1 per request. Column chunk c
    covers shard columns [c * chunk, (c + 1) * chunk); ``splits`` = 1 with ``chunk`` None is the whole shard.
    ``candidates``, ``q_rows`` and ``uniform_samples`` enter only the cross-rank guard hash."""
    rows, shard_width = logits_shard.shape
    bs = rows // rows_per_request
    top_k = candidate_ids.shape[-1]
    # FP32 / BF16 / FP16 shards (16-bit when the LM head emits 16-bit logits; widened exactly in the kernel).
    assert logits_shard.dtype in (torch.float32, torch.bfloat16, torch.float16) and logits_shard.stride(1) == 1
    assert rows == bs * rows_per_request
    assert candidate_ids.is_contiguous() and candidate_ids.shape == (bs, rows_per_request - 1, top_k)
    assert vocab_size < (1 << 24) and (rows_per_request * vocab_size) < (1 << 31)
    if chunk is None:
        assert splits == 1
        chunk = shard_width
    splits, chunk = int(splits), int(chunk)
    assert splits >= 1 and chunk >= 1 and (splits - 1) * chunk < max(shard_width, 1) <= splits * chunk
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    ufin = uniform_samples_final.reshape(bs).contiguous()
    assert ufin.dtype == torch.float32
    assert candidates.shape == (bs, rows_per_request) and candidates.stride(-1) == 1
    assert q_rows.is_contiguous() and q_rows.dtype == torch.float32 and q_rows.shape == candidate_ids.shape
    assert uniform_samples.shape == (bs, rows_per_request - 1) and uniform_samples.stride(-1) == 1
    assert uniform_samples.dtype == torch.float32
    p = payload_width(top_k)
    out = torch.empty((rows, splits * p), dtype=torch.float32, device=logits_shard.device)
    block = min(_BLOCK, triton.next_power_of_2(chunk))
    _vp_local_kernel[(rows, splits)](
        logits_shard, temps, ufin, candidate_ids, candidates, q_rows, uniform_samples, out,
        logits_shard.stride(0), candidates.stride(0), uniform_samples.stride(0), int(valid_width), int(col_offset),
        chunk, float(softcap) if softcap else 1.0,
        NUM_SLOTS=rows_per_request, TOPK=top_k, TOPK_P2=triton.next_power_of_2(top_k), P=p, SPLITS=splits,
        VOCAB=int(vocab_size), BLOCK=block, APPLY_SOFTCAP=bool(softcap), num_warps=8 if block >= 1024 else 4,
    )  # fmt: skip
    return out


def vp_combine(
    *,
    gathered: torch.Tensor,
    candidates: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    temperatures: torch.Tensor,
    vocab_size: int,
    shard_width: int,
    chunk: Optional[int] = None,
    world: Optional[int] = None,
    splits: Optional[int] = None,
    mismatch_counter: Optional[torch.Tensor] = None,
    sync_out: Optional[torch.Tensor] = None,
    sync_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(accept_result int64 [bs, 2] = (num_correct_drafts, bonus), target_predict int64 [bs, gamma + 1]) from the
    all-gathered payloads; every rank gets the same result from the same bytes. ``gathered`` is either world-major
    fp32 [world, rows, splits * P] contiguous (NCCL all_gather_into_tensor; ``world`` / ``splits`` from the shape) or
    row-major fp32 [rows, >= world * splits * P] with unit column stride and any row stride (the multimem all-gather's
    view; pass ``world`` and ``splits``). ``shard_width`` / ``chunk``: every rank's shard width and column chunk
    (vp_split), which locate each candidate id's owning entry. ``mismatch_counter`` (int32 [2], optional) accumulates
    the requests whose seed bits / input hash differ between ranks. ``sync_out`` (fp32 [ceil(2 bs / 4) * 4], optional):
    also write the folded sync's all-reduce input, this rank's values if ``sync_values`` else zeros."""
    bs, num_slots = candidates.shape
    top_k = candidate_ids.shape[-1]
    p = payload_width(top_k)
    rows = bs * num_slots
    if gathered.dim() == 3:
        assert world is None or world == gathered.shape[0]
        world, g_rows, width = gathered.shape
        assert g_rows == rows and width % p == 0 and gathered.is_contiguous()
        assert splits is None or splits == width // p
        splits = width // p
        stride_rank, stride_row = rows * width, width
    else:
        assert gathered.dim() == 2 and world is not None and splits is not None
        assert gathered.shape[0] == rows and gathered.shape[1] >= world * splits * p and gathered.stride(1) == 1
        stride_rank, stride_row = splits * p, gathered.stride(0)
    if chunk is None:
        assert splits == 1
        chunk = shard_width
    assert (splits - 1) * chunk < max(int(shard_width), 1) <= splits * chunk
    assert candidates.stride(-1) == 1 and uniform_samples.stride(-1) == 1
    assert candidate_ids.is_contiguous() and q_rows.is_contiguous() and q_rows.dtype == torch.float32
    assert candidate_ids.shape == (bs, num_slots - 1, top_k) == q_rows.shape
    assert gathered.dtype == torch.float32
    device = candidates.device
    res = torch.empty((bs, 2), dtype=torch.int64, device=device)
    target_predict = torch.empty((bs, num_slots), dtype=torch.int64, device=device)
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    ufin = uniform_samples_final.reshape(bs).contiguous()
    count = mismatch_counter is not None
    if count:
        assert mismatch_counter.dtype == torch.int32 and mismatch_counter.numel() == 2
        assert mismatch_counter.is_contiguous() and mismatch_counter.device == device
    sync = sync_out is not None
    if sync:
        assert sync_out.dtype == torch.float32 and sync_out.is_contiguous()
        assert sync_out.numel() == (2 * bs + 3) // 4 * 4 and vocab_size < (1 << 24)
    _vp_combine_kernel[(bs,)](
        gathered, candidates, candidate_ids, q_rows, uniform_samples, ufin, temps, res, target_predict,
        mismatch_counter if count else res, sync_out if sync else res,
        candidates.stride(0), uniform_samples.stride(0), stride_rank, stride_row, int(shard_width), int(chunk),
        int(vocab_size), 2 * bs, (2 * bs + 3) // 4 * 4,
        W=int(world), SPLITS=int(splits), W_P2=triton.next_power_of_2(int(world) * int(splits)), NUM_SLOTS=num_slots,
        SLOTS_P2=triton.next_power_of_2(num_slots), TOPK=top_k, TOPK_P2=triton.next_power_of_2(top_k),
        P=p, VOCAB=int(vocab_size), COUNT=count, SYNC=sync, SYNC_VALUES=bool(sync_values), num_warps=4,
    )  # fmt: skip
    return res, target_predict


def commit_from_fp32_sync(*, candidates: torch.Tensor, summed: torch.Tensor, prefix_lens: torch.Tensor):
    """commit_selector_block's tuple from the folded sync's fp32 all-reduce output (selector_sampled_accept_synced's
    commit kernel)."""
    bs, block = candidates.shape
    assert candidates.is_contiguous() and summed.dtype == torch.float32 and summed.numel() >= 2 * bs
    device = candidates.device
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    out_tokens = torch.empty((bs, block), dtype=torch.int64, device=device)
    new_seq_lens = torch.empty_like(prefix_lens)
    _commit_block_f32_kernel[(bs,)](
        candidates, summed, prefix_lens, num_correct, commit_lens, bonus,
        out_tokens, new_seq_lens, BLOCK=block, BLOCK_P2=triton.next_power_of_2(block),
    )  # fmt: skip
    return num_correct, commit_lens, bonus, out_tokens, new_seq_lens


class VpPayloadGather:
    """The payload all-gather on the served multimem (NVLS ``multimem.st``) all-gather kernel of
    device_communicators/triton_symm_mem_ag.py (the kernel the served verify-logits gather runs), with the NCCL
    all_gather_into_tensor fallback. The fp32 payload is moved as raw 16-bit halves (a dtype view, no arithmetic), so
    the gathered bytes equal the NCCL form's. The multimem result is a row-major view into the symmetric buffer
    ([rows, world * width] fp32, row stride = the buffer width), valid until the next call; the combine reads it in
    place. The path choice depends only on (rows, width), identical on every rank.

    ``skip_entry_sync`` drops the kernel's entry barrier: only safe when a cross-rank sync orders every rank's read of
    the previous round's view before any rank's next store (as for the served logits gatherer, which runs with it).
    Default False: the entry barrier makes each store wait until every rank has launched this call, i.e. finished the
    previous round's combine (same stream)."""

    def __init__(self, state, group, skip_entry_sync: bool = False):
        self.state = state
        self.group = group
        self.skip_entry_sync = bool(skip_entry_sync)
        self.mode = "multimem" if state is not None else "nccl"
        self._inner = None
        if state is not None:
            from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import all_gather_inner

            self._inner = all_gather_inner

    def fits(self, rows: int, width: int) -> bool:
        st = self.state
        return (
            st is not None
            and 0 < rows <= st.max_token_num
            and (2 * width) % 8 == 0
            and 2 * width * st.world_size <= st.hidden_dim
        )

    def __call__(self, local: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """(gathered, row_major) for this rank's payload fp32 [rows, width]."""
        rows, width = local.shape
        if self.fits(rows, width):
            assert local.dtype == torch.float32 and local.is_contiguous()
            out = self._inner(
                self.state, local.view(torch.bfloat16), tp_hidden_dim=2 * width * self.state.world_size,
                skip_entry_sync=self.skip_entry_sync, safe=False,
            )  # fmt: skip
            return out.view(torch.float32), True
        world = int(self.group.world_size)
        gathered = torch.empty((world, rows, width), dtype=local.dtype, device=local.device)
        self.group.all_gather_into_tensor(gathered.view(-1, width), local)
        return gathered, False


def build_vp_payload_gather(tp_group, *, max_rows: int, max_width: int, skip_entry_sync: bool = False):
    """VpPayloadGather for the served TP group. Collective: call once on every rank, eagerly (outside CUDA-graph
    capture), with identical arguments. Uses the served gatherer's own guards (TP > 1, NVLink clique, multicast
    support) and one eager probe call to allocate + rendezvous the symmetric buffer; any refusal leaves NCCL."""
    state = None
    if tp_group is not None and int(tp_group.world_size) > 1:
        from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as ag

        g = ag.MultimemAllGatherer(int(max_rows), enabled=True, skip_entry_sync=False)
        if g._state is not None:
            probe = torch.zeros((1, 2 * int(max_width)), dtype=torch.bfloat16, device=tp_group.device)
            g(probe)
            if isinstance(g._state, ag.MultimemAllGatherState):
                state = g._state
    return VpPayloadGather(state, tp_group, skip_entry_sync=skip_entry_sync)


def vp_selector_accept(
    *,
    logits_shard: torch.Tensor,
    group,
    vocab_size: int,
    softcap: Optional[float],
    candidates: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_final: torch.Tensor,
    temperatures: torch.Tensor,
    mismatch_counter: Optional[torch.Tensor] = None,
    sync_out: Optional[torch.Tensor] = None,
    sync_values: bool = True,
    gather: Optional[VpPayloadGather] = None,
    split: Optional[int] = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sampled accept and the target argmax from this rank's vocab shard: one local kernel, one all-gather of
    the payloads, one combine kernel. ``group`` is the TP group the LM head shards the vocabulary over. ``gather``
    (optional): a VpPayloadGather (multimem) instead of group.all_gather_into_tensor. ``split``: column chunks per
    shard, 1 = whole shard (the original form), 0 or None = auto (~128 local programs, vp_split), n > 1 = fixed."""
    num_slots = candidates.shape[1]
    world = int(group.world_size) if group is not None else 1
    rows, shard_width = logits_shard.shape
    col_offset = (group.rank_in_group if group is not None else 0) * shard_width
    valid_width = max(0, min(shard_width, vocab_size - col_offset))
    if split == 1:
        splits, chunk = 1, shard_width
    else:
        splits, chunk = vp_split(rows, shard_width, int(split or 0))
    local = vp_local_payload(
        logits_shard=logits_shard, temperatures=temperatures, uniform_samples_final=uniform_samples_final,
        candidate_ids=candidate_ids, rows_per_request=num_slots, valid_width=valid_width,
        col_offset=col_offset, vocab_size=vocab_size, softcap=softcap, candidates=candidates, q_rows=q_rows,
        uniform_samples=uniform_samples, splits=splits, chunk=chunk,
    )  # fmt: skip
    row_major = False
    if world == 1:
        gathered = local.view(1, rows, local.shape[1])
    elif gather is not None:
        gathered, row_major = gather(local)
    else:
        gathered = torch.empty((world,) + tuple(local.shape), dtype=local.dtype, device=local.device)
        group.all_gather_into_tensor(gathered.view(-1, local.shape[1]), local)
    return vp_combine(
        gathered=gathered, candidates=candidates, candidate_ids=candidate_ids, q_rows=q_rows,
        uniform_samples=uniform_samples, uniform_samples_final=uniform_samples_final,
        temperatures=temperatures, vocab_size=vocab_size, shard_width=shard_width, chunk=chunk,
        world=world if row_major else None, splits=splits if row_major else None,
        mismatch_counter=mismatch_counter, sync_out=sync_out, sync_values=sync_values,
    )  # fmt: skip
