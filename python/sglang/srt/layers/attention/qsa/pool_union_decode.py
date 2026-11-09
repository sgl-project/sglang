"""FP8 (or BF16) pool-read QSA target-verify attention, one program group per request.

Opt-in (``SGLANG_QSA_FP8_POOL_VERIFY=1``, read by qwen_sparse_attn_backend): replaces, per
QSA layer of a TP1 topk-1 target verify on an FP8 (E4M3) KV pool,
    fused_kv_prepare(...)  (write the new K/V + pack each row's <= 2051 selected tokens)
    trtllm_batch_decode_with_kv_cache(...) over the packed rows
with four launches chained by PDL:
    kernel 1  _qsa_write_new_kv (pool_decode.py, unchanged) copies the new K/V into the pool;
    kernel U  _qsa_union_tokens grid (requests,): sorts the W rows' selected blocks
              (key block * 8 + row), keeps one entry per block with a row bitmask, resolves
              each token's pool slot, appends the rows' uncompressed tails and counts each
              row's counted-but-invalid entries (score 0, V 0, as in the packed rows);
    kernel 2  _qsa_pool_union_splitk grid (requests, kv_head, splits): the request's W * GQ
              q heads against a split of its token list, each q row masked to its own row's
              bit; E4M3 -> BF16 is exact and there is no k/v scale (stock bmm2_scale 1.0);
    kernel 3  _qsa_pool_union_merge merges the split partials.
The W rows of a request select mostly the same blocks, so each selected token is read once
per request instead of once per row, and nothing is packed.

Semantics per row are pool_decode.py's header (fused_kv_prepare EXPAND + CHAIN with
visible_page_size, then trtllm decode): the same selection, the same attended positions and
the same zero-score entries. Only the FP32 accumulation order and tiling differ (P is rounded
to BF16 before PV, as the stock BF16 pool-read path does), so outputs differ from the stock
chain at the BF16-ulp level: not bitwise.

Tokens no row attends (row mask 0) load neither K nor V, so a stale pool slot can never
reach the output through 0 * NaN.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.qsa.pool_decode import (
    LAST_KERNEL,
    LOG2E,
    _num_sms,
    _qsa_write_new_kv,
)
from sglang.srt.layers.attention.qsa.pool_decode import FUSED_FP8_KV_WRITE, write_new_kv_e4m3

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_num(name: str, default, cast):
    value = os.environ.get(name, "").strip()
    return cast(value) if value else default


# Read once at import, like the backend's other QSA switches; off by default.
ENABLED = _env_flag("SGLANG_QSA_FP8_POOL_VERIFY")
# Below 64 rows the union kernel's fixed ~15 us (kernel U) loses to the stock chain
# (cae28c66 microbench on VR200: 1.29x at 16 rows, 0.68x at 64, 0.51x at 160, 0.43x at 416).
MIN_ROWS = _env_num("SGLANG_QSA_FP8_POOL_VERIFY_MIN_ROWS", 64, int)
# Arbitrary cap at the grouped verify scorer's 576 rows; nothing in the kernels depends on it.
MAX_ROWS = _env_num("SGLANG_QSA_FP8_POOL_VERIFY_MAX_ROWS", 576, int)
# Best of 24 launch configs measured at 43, 86 and 128 requests on VR200 (212 SMs); waves None
# picks the split count per batch (choose_union_splits).
LAUNCH = dict(
    bn=_env_num("SGLANG_QSA_FP8_POOL_VERIFY_BN", 128, int),
    waves=_env_num("SGLANG_QSA_FP8_POOL_VERIFY_WAVES", None, float),
    num_warps=_env_num("SGLANG_QSA_FP8_POOL_VERIFY_WARPS", 8, int),
    num_stages=_env_num("SGLANG_QSA_FP8_POOL_VERIFY_STAGES", 3, int),
)
# Warps of kernel U (each request's token union); its output does not depend on them. 16 is
# 1-3 us per layer faster than 8 at 44-64 requests on VR200 (H452 microbench); 8 is the default.
UNION_WARPS = _env_num("SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS", 8, int)
_LOGGED_ROWS: set = set()


def admits(rows: int, kv_dtype: torch.dtype, q_dtype: torch.dtype) -> bool:
    """The switch, the verify row range and the pool/query dtypes this path serves."""
    return (
        ENABLED
        and MIN_ROWS <= rows <= MAX_ROWS
        and kv_dtype == torch.float8_e4m3fn
        and q_dtype == torch.bfloat16
    )


def log_engaged(rows: int, width: int, kv_heads: int, device) -> None:
    """One line per verify row count, printed at graph capture (the runner's engagement check)."""
    if rows in _LOGGED_ROWS:
        return
    _LOGGED_ROWS.add(rows)
    logger.info(
        "SGLANG_QSA_FP8_POOL_VERIFY engaged rows=%d width=%d requests=%d splits=%d "
        "bn=%d warps=%d stages=%d rows_range=%d-%d union_warps=%d",
        rows,
        width,
        rows // width,
        choose_union_splits(rows // width, kv_heads, _num_sms(device), LAUNCH["waves"]),
        LAUNCH["bn"],
        LAUNCH["num_warps"],
        LAUNCH["num_stages"],
        MIN_ROWS,
        MAX_ROWS,
        UNION_WARPS,
    )


@triton.jit
def _qsa_union_tokens(
    IDX,
    IDX_STRIDE,
    LENGTH,
    QPOS,
    LOC,
    MAP,
    MAP_STRIDE,
    REQ,
    TS,
    TM,
    NTOK,
    NINV,
    SCR,
    IB: tl.constexpr,
    W: tl.constexpr,
    RATIO: tl.constexpr,
    NB: tl.constexpr,
    NT: tl.constexpr,
    TB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    i = tl.arange(0, NB)
    j = i // IB
    c = i % IB
    in_rows = j < W
    row = r * W + j
    length = tl.load(LENGTH + row, mask=in_rows, other=0)
    active = tl.load(LOC + row, mask=in_rows, other=0) != 0
    take = in_rows & (c < tl.minimum(length // RATIO, IB))
    blk = tl.load(IDX + row.to(tl.int64) * IDX_STRIDE + c, mask=take, other=-1)
    use = take & active & (blk >= 0)
    # invalid tokens of the compressed part: a -1 entry, or a block's tokens at or past the length
    inv = tl.where(
        take,
        tl.where(blk < 0, RATIO, tl.minimum(RATIO, tl.maximum(0, blk * RATIO + RATIO - length))),
        0,
    )
    key = tl.where(use, blk * 8 + j, 1073741824)  # unused entries sort last
    key = tl.sort(key)
    base = SCR + r.to(tl.int64) * NB
    tl.store(base + i, key)
    tl.debug_barrier()
    kb = key // 8
    used = key != 1073741824
    prev = tl.load(base + i - 1, mask=i > 0, other=-8)
    head = used & (kb != prev // 8)
    one = tl.full([NB], 1, tl.int32)
    mask = tl.where(used, one << (key % 8), 0)
    for dlt in tl.static_range(1, W):
        nxt = tl.load(base + i + dlt, mask=i + dlt < NB, other=1073741824)
        same = (nxt != 1073741824) & (nxt // 8 == kb)
        mask = mask | tl.where(same, one << (nxt % 8), 0)
    slot = tl.cumsum(head.to(tl.int32), 0) - 1
    ucnt = tl.sum(head.to(tl.int32), 0)

    req = tl.load(REQ + r * W).to(tl.int64)
    map_base = MAP + req * MAP_STRIDE
    ts_base = TS + r.to(tl.int64) * NT
    tm_base = TM + r.to(tl.int64) * NT
    for t in tl.static_range(RATIO):
        pos = kb * RATIO + t
        below = tl.zeros([NB], tl.int32)
        for jj in tl.static_range(W):
            below = below | tl.where(pos < tl.load(LENGTH + r * W + jj), 1 << jj, 0)
        tok_slot = tl.load(map_base + pos, mask=head, other=0)
        tl.store(ts_base + slot * RATIO + t, tok_slot, mask=head)
        tl.store(tm_base + slot * RATIO + t, mask & below, mask=head)

    # uncompressed tails: row j attends tail_start_j .. min(qpos_j + 1, len_j) - 1 (at most
    # RATIO - 1 tokens); a chain's rows keep them within TB positions of the first tail start
    tbase = 1073741824
    for jj in tl.static_range(W):
        q_j = tl.load(QPOS + r * W + jj)
        act_j = tl.load(LOC + r * W + jj) != 0
        tbase = tl.minimum(tbase, tl.where(act_j, (q_j + 1) // RATIO * RATIO, 1073741824))
    tp = tbase + tl.arange(0, TB)
    tmask = tl.zeros([TB], tl.int32)
    for jj in tl.static_range(W):
        q_j = tl.load(QPOS + r * W + jj)
        len_j = tl.load(LENGTH + r * W + jj)
        act_j = tl.load(LOC + r * W + jj) != 0
        ts_j = (q_j + 1) // RATIO * RATIO
        te_j = tl.minimum(tl.minimum(q_j + 1, len_j), ts_j + RATIO - 1)
        tmask = tmask | tl.where(act_j & (tp >= ts_j) & (tp < te_j), 1 << jj, 0)
    keep = tmask != 0
    tidx = tl.cumsum(keep.to(tl.int32), 0) - 1
    tslot = tl.load(map_base + tp, mask=keep, other=0)
    tl.store(ts_base + ucnt * RATIO + tidx, tslot, mask=keep)
    tl.store(tm_base + ucnt * RATIO + tidx, tmask, mask=keep)
    tl.store(NTOK + r, ucnt * RATIO + tl.sum(keep.to(tl.int32), 0))

    # per row: invalid compressed tokens, or every counted token of an inactive row (loc 0)
    jr = tl.arange(0, NB // IB)
    rrow = r * W + jr
    rin = jr < W
    rlen = tl.load(LENGTH + rrow, mask=rin, other=0)
    rq = tl.load(QPOS + rrow, mask=rin, other=0)
    ract = tl.load(LOC + rrow, mask=rin, other=0) != 0
    n = tl.minimum(rlen // RATIO, IB) * RATIO
    tail_start = (rq + 1) // RATIO * RATIO
    tail_count = tl.maximum(0, tl.minimum(rq + 1, rlen) - tail_start)
    inv_rows = tl.sum(tl.reshape(inv, (NB // IB, IB)), 1)
    tl.store(NINV + rrow, tl.where(ract, inv_rows, n + tail_count), mask=rin)


@triton.jit
def _qsa_pool_union_splitk(
    Q,
    K,
    V,
    TS,
    TM,
    NTOK,
    NINV,
    M_OUT,
    L_OUT,
    ACC_OUT,
    qk_scale,
    Q_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    QP: tl.constexpr,
    W: tl.constexpr,
    NT: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    KV_FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    r = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    d = tl.arange(0, D)
    g = tl.arange(0, QP)
    jq = g // GQ
    qv = jq < W
    row = r * W + jq
    q = tl.load(
        Q + row.to(tl.int64)[:, None] * Q_STRIDE + (head * GQ + g % GQ)[:, None] * D + d[None, :],
        mask=qv[:, None],
        other=0.0,
    )
    ntok = tl.load(NTOK + r)
    chunk = tl.cdiv(tl.cdiv(ntok, SPLITS), BN) * BN
    start = split * chunk
    end = tl.minimum(start + chunk, ntok)
    ts_base = TS + r.to(tl.int64) * NT
    tm_base = TM + r.to(tl.int64) * NT

    m_i = tl.full([QP], float("-inf"), tl.float32)
    l_i = tl.zeros([QP], tl.float32)
    acc = tl.zeros([QP, D], tl.float32)
    for off in range(start, end, BN):
        t = off + tl.arange(0, BN)
        tin = t < end
        tm = tl.load(tm_base + t, mask=tin, other=0)
        # a listed token that no row attends is never read (no stale slot can reach PV)
        live = tin & (tm != 0)
        tok_slot = tl.load(ts_base + t, mask=live, other=0).to(tl.int64)
        kvoff = tok_slot[:, None] * (H * D) + head * D + d[None, :]
        k = tl.load(K + kvoff, mask=live[:, None], other=0.0)
        v = tl.load(V + kvoff, mask=live[:, None], other=0.0)
        if KV_FP8:
            k = k.to(Q.dtype.element_ty)  # E4M3 -> BF16 is exact
            v = v.to(Q.dtype.element_ty)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(((tm[None, :] >> jq[:, None]) & 1) != 0, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if split == 0:
        # counted-but-invalid entries: score q . 0 = 0, V 0
        ninv = tl.load(NINV + row, mask=qv, other=0).to(tl.float32)
        has = ninv > 0
        m_new = tl.where(has, tl.maximum(m_i, 0.0), m_i)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.where(has, ninv * tl.math.exp2(0.0 - m_safe), 0.0)
        acc = acc * alpha[:, None]
        m_i = m_new

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    part = (r * H + head) * SPLITS + split
    tl.store(M_OUT + part * QP + g, m_i, mask=qv)
    tl.store(L_OUT + part * QP + g, l_i, mask=qv)
    live_q = qv & (m_i > float("-inf"))
    tl.store(ACC_OUT + part.to(tl.int64) * (QP * D) + g[:, None] * D + d[None, :], acc, mask=live_q[:, None])


@triton.jit
def _qsa_pool_union_merge(
    M_IN,
    L_IN,
    ACC_IN,
    O,
    O_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    QP: tl.constexpr,
    W: tl.constexpr,
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
    g = (row % W) * GQ + qh % GQ
    base = (row // W * H + head) * SPLITS
    s_all = tl.arange(0, SB)
    d = tl.arange(0, D)
    mx = tl.full([SB], float("-inf"), tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * QP + g, mask=s < SPLITS, other=float("-inf"))
        mx = tl.maximum(mx, m)
    M = tl.max(mx, 0)
    M = tl.where(M == float("-inf"), 0.0, M)
    L = tl.zeros([SB], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * QP + g, mask=s < SPLITS, other=float("-inf"))
        l = tl.load(L_IN + (base + s) * QP + g, mask=s < SPLITS, other=0.0)
        w = tl.math.exp2(m - M)
        L += w * l
        live = (s < SPLITS) & (m > float("-inf"))
        a = tl.load(
            ACC_IN + ((base + s).to(tl.int64) * QP + g)[:, None] * D + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += tl.sum(a * w[:, None], 0)
    Ls = tl.sum(L, 0)
    out = tl.where(Ls > 0, acc / Ls, 0.0)
    tl.store(O + row.to(tl.int64) * O_STRIDE + qh * D + d, out.to(O.dtype.element_ty))


def choose_union_splits(requests, kv_heads, sms, waves=None):
    """Splits of a request's token list. With waves, requests * kv_heads * splits ~ waves * SMs.
    Without: two splits once the (request, head) pairs exceed the SMs; below that, the split
    count with the least wave-quantized work per CTA, each split charged 10% for its merge and
    q load. Picks (VR200, 212 SMs) measured fastest at 43, 64, 86, 104, 128 and 144 requests."""
    pairs = max(1, requests * kv_heads)
    if waves is not None:
        return max(1, min(16, round(waves * sms / pairs)))
    if pairs > sms:
        return 2
    cost = lambda s: -(-pairs * s // sms) / s * (1.0 + 0.1 * s)
    return min(range(1, 17), key=lambda s: (cost(s), s))


def union_buffers(rows, width, ib, ratio):
    """(nb, tb, nt): the sort width, the tail window and one request's token-list capacity."""
    nb = triton.next_power_of_2(width * ib)
    tb = max(16, triton.next_power_of_2(width + 2 * ratio))
    return nb, tb, nb * ratio + tb


def qsa_pool_union_decode(
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
    bn=128,
    num_warps=8,
    num_stages=2,
    pdl=True,
    write_new_kv=True,
):
    """Target-verify sparse attention read from the pool, each request's union of tokens once.

    Rows are requests' groups of ``width`` consecutive rows with chain positions, as a topk-1
    verify lays them out (row_requests constant within a group). q [rows, QH, 256] BF16 (or
    [rows, QH * 256]); k/v_cache [slots, H, 256] BF16 or FP8 (E4M3); an FP8 pool takes
    k/v_new already in its dtype (kernel 1 copies their bytes); write_new_kv=False when the
    new K/V were stored before this call. Returns [rows, QH * 256] in q's dtype.
    """
    rows, ib = block_indices.shape
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    q3 = q.view(rows, -1, dim)
    qh = q3.shape[1]
    gq = qh // kv_heads
    assert rows % width == 0
    assert q3.stride(2) == 1 and q3.stride(1) == dim
    assert k_new.stride(-1) == 1 and k_new.shape[-1] * k_new.shape[-2] == kv_heads * dim
    kv_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if kv_fp8:
        assert k_new.dtype == v_new.dtype == k_cache.dtype == v_cache.dtype or (
            FUSED_FP8_KV_WRITE and k_new.dtype == v_new.dtype == torch.bfloat16 and k_cache.dtype == v_cache.dtype
        )
        assert q.dtype == torch.bfloat16
    if out is None:
        out = torch.empty(rows, qh * dim, dtype=q.dtype, device=q.device)
    if rows == 0:
        return out
    requests = rows // width
    qp = max(16, triton.next_power_of_2(width * gq))
    nb, tb, nt = union_buffers(rows, width, ib, ratio)
    splits = choose_union_splits(requests, kv_heads, _num_sms(q.device), waves)
    nparts = requests * kv_heads * splits
    dev = q.device
    tok_slots = torch.empty(requests * nt, dtype=torch.int32, device=dev)
    tok_masks = torch.empty(requests * nt, dtype=torch.int32, device=dev)
    scratch = torch.empty(requests * nb, dtype=torch.int32, device=dev)
    ntok = torch.empty(requests, dtype=torch.int32, device=dev)
    ninv = torch.empty(rows, dtype=torch.int32, device=dev)
    m_buf = torch.empty(nparts * qp, dtype=torch.float32, device=dev)
    l_buf = torch.empty(nparts * qp, dtype=torch.float32, device=dev)
    acc_buf = torch.empty(nparts * qp * dim, dtype=torch.float32, device=dev)
    pdl_kw = {"launch_pdl": True} if pdl else {}
    if write_new_kv:
        nk = k_new.reshape(rows, kv_heads * dim) if k_new.dim() == 3 else k_new
        nv = v_new.reshape(rows, kv_heads * dim) if v_new.dim() == 3 else v_new
        if nk.dtype != k_cache.dtype:  # SGLANG_QSA_FUSED_FP8_KV_WRITE: BF16 rows, cast in the write
            write_new_kv_e4m3(k_cache, v_cache, nk, nv, loc, rows, kv_heads, dim, pdl, pdl_kw)
        else:
            _qsa_write_new_kv[(rows, kv_heads)](
                k_cache, v_cache, nk, nv, loc, nk.stride(0), nv.stride(0), kv_heads, dim, pdl, num_warps=1, **pdl_kw
            )
    _qsa_union_tokens[(requests,)](
        block_indices,
        block_indices.stride(0),
        sequence_lens,
        query_positions,
        loc,
        req_to_token,
        req_to_token.stride(0),
        row_requests,
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        scratch,
        ib,
        width,
        ratio,
        nb,
        nt,
        tb,
        pdl,
        num_warps=UNION_WARPS,
        **pdl_kw,
    )
    LAST_KERNEL["union"] = _qsa_pool_union_splitk[(requests, kv_heads, splits)](
        q3,
        k_cache,
        v_cache,
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        m_buf,
        l_buf,
        acc_buf,
        float(scale) * LOG2E,
        q3.stride(0),
        kv_heads,
        dim,
        gq,
        qp,
        width,
        nt,
        splits,
        bn,
        kv_fp8,
        pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kw,
    )
    sb = min(16, triton.next_power_of_2(splits))
    _qsa_pool_union_merge[(rows * qh,)](
        m_buf, l_buf, acc_buf, out, out.stride(0), kv_heads, dim, gq, qp, width, splits, sb, pdl, num_warps=2,
        **pdl_kw,
    )
    return out
