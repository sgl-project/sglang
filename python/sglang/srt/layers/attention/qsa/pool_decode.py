"""Pool-reading split-K QSA sparse decode (no packed copy).

Replaces, per QSA layer at verify / draft decode:
    fused_kv_prepare(...)  (write new K/V at loc + pack <=2051 selected tokens/row)
    trtllm_batch_decode_with_kv_cache(...) over the packed rows
with
    kernel 1  _qsa_write_new_kv grid (rows, kv_head): writes k_new/v_new at loc (loc 0 = skip).
    kernel 2  _qsa_pool_splitk  grid (rows, kv_head, splits): every CTA expands its slice of the
              row's block indices (block -> RATIO positions, then the uncompressed tail
              up to the row's query position), maps positions through req_to_token,
              loads K/V straight from the bf16 pool and computes online-softmax partials
              (m, l, acc) for the GQ q heads of that kv head (padded to 16 for tl.dot).
    kernel 3  _qsa_pool_merge   grid (rows * q_heads): merges the split partials.
Default (fastest measured): mode="separate_write" runs a tiny kernel 1
(_qsa_write_new_kv) first and kernel 2 reads every token from the pool; the three kernels
are chained with PDL (griddepcontrol). mode="fresh_select"/"fresh_dual" fuse kernel 1 into
kernel 2 (split-0 CTAs write k/v_new and every CTA reads this call's chain tokens from
k/v_new); fused_merge=True replaces kernel 3 with a last-CTA semaphore merge (measured
much slower: the single merging CTA serialises over the splits).

Semantics mirror fused_kv_prepare(EXPAND, CHAIN, visible_page_size) + trtllm decode:
  n          = min(length // RATIO, IB) * RATIO
  tail_start = (qpos + 1) // RATIO * RATIO
  tail_count = max(0, min(qpos + 1, length) - tail_start)
  count      = n + tail_count                (the trtllm seq_len of the row)
  c < n      : pos = idx[c // RATIO] * RATIO + c % RATIO   (if idx >= 0, else invalid)
  c >= n     : pos = tail_start + (c - n) if c - n < RATIO - 1 and pos < qpos + 1
  valid      = pos >= 0 & pos < length & loc[row] != 0
Counted-but-invalid entries (c < count, not valid) are attended as K = V = 0 (score 0),
exactly what trtllm sees in the zero-filled packed rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _qsa_write_new_kv(K, V, NK, NV, LOC, NK_STRIDE, NV_STRIDE, H: tl.constexpr, D: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    slot = tl.load(LOC + row).to(tl.int64)
    k = tl.load(NK + row.to(tl.int64) * NK_STRIDE + head * D + d).to(K.dtype.element_ty)
    v = tl.load(NV + row.to(tl.int64) * NV_STRIDE + head * D + d).to(V.dtype.element_ty)
    tl.store(K + slot * H * D + head * D + d, k, mask=slot != 0)
    tl.store(V + slot * H * D + head * D + d, v, mask=slot != 0)


@triton.jit
def _qsa_pool_splitk(
    Q,
    K,
    V,
    NK,
    NV,
    LOC,
    MAP,
    REQ,
    IDX,
    LENGTH,
    QPOS,
    M_OUT,
    L_OUT,
    ACC_OUT,
    COUNTER,
    O,
    O_STRIDE,
    qk_scale,
    Q_STRIDE,
    MAP_STRIDE,
    NK_STRIDE,
    NV_STRIDE,
    IDX_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    GP: tl.constexpr,
    IB: tl.constexpr,
    RATIO: tl.constexpr,
    W: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    FRESH_NEW: tl.constexpr,
    WRITE_NEW: tl.constexpr,
    FUSED_MERGE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    d = tl.arange(0, D)
    g = tl.arange(0, GP)

    loc = tl.load(LOC + row)
    active = loc != 0
    if WRITE_NEW:
        if split == 0:
            slot_new = loc.to(tl.int64)
            kn = tl.load(NK + row.to(tl.int64) * NK_STRIDE + head * D + d).to(K.dtype.element_ty)
            vn = tl.load(NV + row.to(tl.int64) * NV_STRIDE + head * D + d).to(V.dtype.element_ty)
            tl.store(K + slot_new * H * D + head * D + d, kn, mask=active)
            tl.store(V + slot_new * H * D + head * D + d, vn, mask=active)

    req = tl.load(REQ + row).to(tl.int64)
    length = tl.load(LENGTH + row)
    qpos = tl.load(QPOS + row)
    n = tl.minimum(length // RATIO, IB) * RATIO
    tail_start = (qpos + 1) // RATIO * RATIO
    tail_count = tl.maximum(0, tl.minimum(qpos + 1, length) - tail_start)
    count = n + tail_count
    first_pos = length - 1 - row % W
    group_row = row // W * W

    q_ptrs = Q + row.to(tl.int64) * Q_STRIDE + (head * GQ + g)[:, None] * D + d[None, :]
    q = tl.load(q_ptrs, mask=(g < GQ)[:, None], other=0.0)

    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)

    start = split * CHUNK
    end = tl.minimum(start + CHUNK, count)
    idx_base = IDX + row.to(tl.int64) * IDX_STRIDE
    map_base = MAP + req * MAP_STRIDE
    for off in range(start, end, BN):
        c = off + tl.arange(0, BN)
        in_cnt = c < count
        blk = tl.load(idx_base + c // RATIO, mask=c < n, other=-1)
        tail = tail_start + (c - n)
        pos = tl.where(
            (c < n) & (blk >= 0),
            blk * RATIO + c % RATIO,
            tl.where((c >= n) & (c - n < RATIO - 1) & (tail < qpos + 1), tail, -1),
        )
        valid = (pos >= 0) & (pos < length) & active & in_cnt
        slot = tl.load(map_base + pos, mask=valid, other=0).to(tl.int64)
        oldoff = slot[:, None] * (H * D) + head * D + d[None, :]
        if FRESH_NEW == 1:
            # pointer select (stock style)
            rel = pos - first_pos
            fresh = valid & (rel >= 0)
            in_row = tl.where(fresh, group_row + rel, 0).to(tl.int64)
            kp = tl.where(fresh[:, None], NK + (in_row[:, None] * NK_STRIDE + head * D + d[None, :]), K + oldoff)
            vp = tl.where(fresh[:, None], NV + (in_row[:, None] * NV_STRIDE + head * D + d[None, :]), V + oldoff)
            k = tl.load(kp, mask=valid[:, None], other=0.0)
            v = tl.load(vp, mask=valid[:, None], other=0.0)
        elif FRESH_NEW == 2:
            # two masked loads: pool for old tokens, k_new/v_new for this call's chain tokens
            rel = pos - first_pos
            fresh = valid & (rel >= 0)
            old = valid & (rel < 0)
            in_row = tl.where(fresh, group_row + rel, 0).to(tl.int64)
            k = tl.load(K + oldoff, mask=old[:, None], other=0.0)
            v = tl.load(V + oldoff, mask=old[:, None], other=0.0)
            kf = tl.load(NK + in_row[:, None] * NK_STRIDE + head * D + d[None, :], mask=fresh[:, None], other=0.0)
            vf = tl.load(NV + in_row[:, None] * NV_STRIDE + head * D + d[None, :], mask=fresh[:, None], other=0.0)
            k = k + kf.to(k.dtype)
            v = v + vf.to(v.dtype)
        else:
            k = tl.load(K + oldoff, mask=valid[:, None], other=0.0)
            v = tl.load(V + oldoff, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(in_cnt[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    part = (row * H + head) * SPLITS + split
    gm = g < GQ
    if FUSED_MERGE:
        if SPLITS == 1:
            out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
            tl.store(
                O + row.to(tl.int64) * O_STRIDE + (head * GQ + g)[:, None] * D + d[None, :],
                out.to(O.dtype.element_ty),
                mask=gm[:, None],
            )
        else:
            tl.store(M_OUT + part * GP + g, m_i, mask=gm)
            tl.store(L_OUT + part * GP + g, l_i, mask=gm)
            tl.store(ACC_OUT + part.to(tl.int64) * (GP * D) + g[:, None] * D + d[None, :], acc, mask=gm[:, None])
            tl.debug_barrier()
            ticket = tl.atomic_add(COUNTER + row * H + head, 1, sem="acq_rel", scope="gpu")
            if ticket == SPLITS - 1:
                # last CTA of this (row, kv head): merge all splits, then re-arm the counter
                base = (row * H + head) * SPLITS
                M = tl.full([GP], float("-inf"), tl.float32)
                for s in range(SPLITS):
                    m_s = tl.load(M_OUT + (base + s) * GP + g, mask=gm, other=float("-inf"), cache_modifier=".cg")
                    M = tl.maximum(M, m_s)
                M = tl.where(M == float("-inf"), 0.0, M)
                L = tl.zeros([GP], tl.float32)
                tot = tl.zeros([GP, D], tl.float32)
                for s in range(SPLITS):
                    m_s = tl.load(M_OUT + (base + s) * GP + g, mask=gm, other=float("-inf"), cache_modifier=".cg")
                    l_s = tl.load(L_OUT + (base + s) * GP + g, mask=gm, other=0.0, cache_modifier=".cg")
                    w = tl.math.exp2(m_s - M)
                    L += w * l_s
                    a = tl.load(
                        ACC_OUT + (base + s).to(tl.int64) * (GP * D) + g[:, None] * D + d[None, :],
                        mask=(gm & (m_s > float("-inf")))[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    tot += a * w[:, None]
                out = tl.where(L[:, None] > 0, tot / L[:, None], 0.0)
                tl.store(
                    O + row.to(tl.int64) * O_STRIDE + (head * GQ + g)[:, None] * D + d[None, :],
                    out.to(O.dtype.element_ty),
                    mask=gm[:, None],
                )
                tl.atomic_xchg(COUNTER + row * H + head, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(M_OUT + part * GP + g, m_i, mask=gm)
        tl.store(L_OUT + part * GP + g, l_i, mask=gm)
        tl.store(ACC_OUT + part.to(tl.int64) * (GP * D) + g[:, None] * D + d[None, :], acc, mask=gm[:, None])


@triton.jit
def _qsa_pool_merge(
    M_IN,
    L_IN,
    ACC_IN,
    O,
    O_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    GP: tl.constexpr,
    SPLITS: tl.constexpr,
    SB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    qh_all = H * GQ
    row = pid // qh_all
    qh = pid % qh_all
    head = qh // GQ
    gi = qh % GQ
    base = (row * H + head) * SPLITS
    s_all = tl.arange(0, SB)
    d = tl.arange(0, D)
    # pass 1: global max over splits
    mx = tl.full([SB], float("-inf"), tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * GP + gi, mask=s < SPLITS, other=float("-inf"))
        mx = tl.maximum(mx, m)
    M = tl.max(mx, 0)
    M = tl.where(M == float("-inf"), 0.0, M)
    L = tl.zeros([SB], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * GP + gi, mask=s < SPLITS, other=float("-inf"))
        l = tl.load(L_IN + (base + s) * GP + gi, mask=s < SPLITS, other=0.0)
        w = tl.math.exp2(m - M)
        L += w * l
        live = (s < SPLITS) & (m > float("-inf"))
        a = tl.load(
            ACC_IN + ((base + s).to(tl.int64) * GP + gi)[:, None] * D + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += tl.sum(a * w[:, None], 0)
    Ls = tl.sum(L, 0)
    out = tl.where(Ls > 0, acc / Ls, 0.0)
    tl.store(O + row.to(tl.int64) * O_STRIDE + qh * D + d, out.to(O.dtype.element_ty))


_SMS = {}
_COUNTERS = {}
LAST_KERNEL = {}


def _num_sms(device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SMS[key]


def _counter(n, device):
    """Zeroed int32 semaphores for the fused merge; the last CTA re-arms its entry,
    so one allocation serves every (CUDA-graph) replay."""
    key = (device.index, n)
    buf = _COUNTERS.get(key)
    if buf is None:
        buf = torch.zeros(max(n, 1), dtype=torch.int32, device=device)
        _COUNTERS[key] = buf
    return buf


def choose_config(rows, kv_heads, topk_tokens, sms, waves=None, bn=None):
    """Splits so rows*kv_heads*splits ~ waves * SMs; CHUNK a multiple of BN.
    Auto (measured, cold L2): >= 32 (row, kv head) pairs -> BN 32, 3 waves;
    fewer -> BN 64, 1 wave."""
    rh = rows * kv_heads
    if bn is None:
        bn = 32 if rh >= 32 else 64
    if waves is None:
        waves = 3.0 if rh >= 32 else 1.0
    splits = max(1, round(waves * sms / rh))
    chunk = triton.cdiv(triton.cdiv(topk_tokens, splits), bn) * bn
    splits = triton.cdiv(topk_tokens, chunk)
    return splits, chunk, bn


_MODES = {"separate_write": 0, "fresh_select": 1, "fresh_dual": 2}


def qsa_pool_sparse_decode(
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    loc,
    req_to_token,
    row_requests,
    block_indices,
    sequence_lens,
    query_positions,
    width,
    scale,
    ratio=4,
    out=None,
    waves=None,
    bn=None,
    num_warps=4,
    num_stages=4,
    mode="separate_write",
    fused_merge=False,
    workspace=None,
    pdl=True,
):
    """q [rows, QH, 256] bf16 (or [rows, QH*256]); k/v_cache [slots, H, 256];
    k/v_new [rows, H, 256] (row stride arbitrary, inner contiguous); loc [rows] int32;
    req_to_token [reqs, max_ctx] int32; row_requests [rows] int32; block_indices
    [rows, IB] int32; sequence_lens / query_positions [rows] int32. Returns [rows, QH*256].

    mode: "separate_write" = kernel 1 writes k/v_new at loc, kernel 2 reads only the pool;
    "fresh_dual" / "fresh_select" = kernel 2 writes k/v_new (split 0) and reads this call's
    chain tokens from k/v_new (fused_kv_prepare CHAIN semantics)."""
    rows, ib = block_indices.shape
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    q3 = q.view(rows, -1, dim)
    qh = q3.shape[1]
    gq = qh // kv_heads
    gp = max(16, triton.next_power_of_2(gq))
    assert q3.stride(2) == 1 and q3.stride(1) == dim
    assert k_new.stride(-1) == 1 and k_new.shape[-1] * k_new.shape[-2] == kv_heads * dim
    topk_tokens = ib * ratio + ratio - 1
    if out is None:
        out = torch.empty(rows, qh * dim, dtype=q.dtype, device=q.device)
    if rows == 0:
        return out
    splits, chunk, bn = choose_config(rows, kv_heads, topk_tokens, _num_sms(q.device), waves, bn)
    nparts = rows * kv_heads * splits
    if workspace is None:
        m_buf = torch.empty(nparts * gp, dtype=torch.float32, device=q.device)
        l_buf = torch.empty(nparts * gp, dtype=torch.float32, device=q.device)
        acc_buf = torch.empty(nparts * gp * dim, dtype=torch.float32, device=q.device)
    else:
        m_buf = workspace[: nparts * gp]
        l_buf = workspace[nparts * gp : 2 * nparts * gp]
        acc_buf = workspace[2 * nparts * gp : 2 * nparts * gp + nparts * gp * dim]
    nk = k_new.reshape(rows, kv_heads * dim) if k_new.dim() == 3 else k_new
    nv = v_new.reshape(rows, kv_heads * dim) if v_new.dim() == 3 else v_new
    mcode = _MODES[mode]
    pdl_kw = {"launch_pdl": True} if pdl else {}
    if mcode == 0:
        _qsa_write_new_kv[(rows, kv_heads)](
            k_cache, v_cache, nk, nv, loc, nk.stride(0), nv.stride(0), kv_heads, dim, pdl, num_warps=1, **pdl_kw
        )
    counter = _counter(rows * kv_heads, q.device) if fused_merge else m_buf
    LAST_KERNEL["splitk"] = _qsa_pool_splitk[(rows, kv_heads, splits)](
        q3,
        k_cache,
        v_cache,
        nk,
        nv,
        loc,
        req_to_token,
        row_requests,
        block_indices,
        sequence_lens,
        query_positions,
        m_buf,
        l_buf,
        acc_buf,
        counter,
        out,
        out.stride(0),
        float(scale) * LOG2E,
        q3.stride(0),
        req_to_token.stride(0),
        nk.stride(0),
        nv.stride(0),
        block_indices.stride(0),
        kv_heads,
        dim,
        gq,
        gp,
        ib,
        ratio,
        width,
        splits,
        chunk,
        bn,
        mcode,
        mcode != 0,
        fused_merge,
        pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kw,
    )
    if not fused_merge:
        sb = min(16, triton.next_power_of_2(splits))
        _qsa_pool_merge[(rows * qh,)](
            m_buf, l_buf, acc_buf, out, out.stride(0), kv_heads, dim, gq, gp, splits, sb, pdl, num_warps=2, **pdl_kw
        )
    return out
