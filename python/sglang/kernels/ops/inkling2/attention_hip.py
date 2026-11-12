"""Inkling2 extend (prefill) attention with the relative bias for ROCm.

Logits are qk * scale + rel_bias[t, h, q_pos - kv_pos] for distances below the bias
extent and qk * scale beyond it, so key tiles that every query of the block sees from
farther than the extent run as plain attention: no bias, no masks. Prefix keys come
from the paged cache through kv_indices (relative positions, as the Triton backend
lays them out; window layers list only their window), the chunk's own keys from the
contiguous K/V, causally. Returns the FP32 output and the natural-log LSE;
rel_verify_attention serves short extends (MTP verify) split-KV."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.inkling2.attention import _sigmoid

LOG2E = 1.4426950408889634
LOG2E_C = tl.constexpr(LOG2E)


@triton.jit
def _tile_update(qk, v, acc, m_i, l_i):
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_ref)
    p = tl.exp2(qk - m_ref[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, m_new, l_i


@triton.jit
def _prefix_tile(
    q,
    K_Buf,
    V_Buf,
    kv_indices,
    kv_start,
    start_n,
    kvh,
    d,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    n_mask,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BN)
    loc = tl.load(kv_indices + kv_start + offs_n, n_mask, other=0)
    page, slot = loc // PAGE, loc % PAGE
    k = tl.load(
        K_Buf
        + page[None, :] * stride_kp
        + slot[None, :] * stride_kt
        + kvh * stride_kh
        + d[:, None],
        n_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        V_Buf
        + page[:, None] * stride_vp
        + slot[:, None] * stride_vt
        + kvh * stride_vh
        + d[None, :],
        n_mask[:, None],
        other=0.0,
    )
    return tl.dot(q, k), v


@triton.jit
def _bias(Bias, q_rows, h, dist, mask, stride_bt, stride_bh, EXTENT: tl.constexpr):
    keep = mask & (dist >= 0) & (dist < EXTENT)
    return tl.load(
        Bias + q_rows[:, None] * stride_bt + h * stride_bh + dist,
        keep,
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _rel_extend_fwd(
    Q,
    K_Ext,
    V_Ext,
    K_Buf,
    V_Buf,
    Bias,
    O,
    LSE,
    qo_indptr,
    kv_indptr,
    kv_indices,
    scale_log2,
    stride_qt,
    stride_qh,
    stride_kt_ext,
    stride_kh_ext,
    stride_vt_ext,
    stride_vh_ext,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_bt,
    stride_bh,
    stride_ot,
    stride_oh,
    stride_lt,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    h = tl.program_id(0)
    mblk = tl.program_id(1)
    b = tl.program_id(2)
    kvh = h // GROUP
    q_start = tl.load(qo_indptr + b)
    ext_len = tl.load(qo_indptr + b + 1) - q_start
    if mblk * BM >= ext_len:
        return
    kv_start = tl.load(kv_indptr + b)
    prefix = tl.load(kv_indptr + b + 1) - kv_start
    offs_m = mblk * BM + tl.arange(0, BM)
    m_mask = offs_m < ext_len
    q_rows = (q_start + offs_m).to(tl.int64)
    d = tl.arange(0, D)
    q = tl.load(
        Q + q_rows[:, None] * stride_qt + h * stride_qh + d[None, :],
        m_mask[:, None],
        other=0.0,
    )
    q_pos = prefix + offs_m
    acc = tl.zeros((BM, D), tl.float32)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)

    # Prefix tiles farther than the extent from every query of the block (whole tiles
    # only, so no masks); window layers have none.
    far_end = 0
    if WINDOW < 0:
        far_end = tl.maximum(prefix + mblk * BM - EXTENT + 1, 0) // BN * BN
        far_end = tl.minimum(far_end, prefix // BN * BN)
    for start_n in range(0, far_end, BN):
        qk, v = _prefix_tile(
            q, K_Buf, V_Buf, kv_indices, kv_start, start_n, kvh, d,
            stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
            tl.full((BN,), True, tl.int1), PAGE, BN,
        )  # fmt: skip
        acc, m_i, l_i = _tile_update(qk * scale_log2, v, acc, m_i, l_i)

    near_start = far_end
    if WINDOW >= 0:
        near_start = tl.maximum(prefix + mblk * BM - WINDOW, 0) // BN * BN
    for start_n in range(near_start, prefix, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_mask = offs_n < prefix
        qk, v = _prefix_tile(
            q, K_Buf, V_Buf, kv_indices, kv_start, start_n, kvh, d,
            stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
            n_mask, PAGE, BN,
        )  # fmt: skip
        dist = q_pos[:, None] - offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]
        if WINDOW >= 0:
            mask &= dist <= WINDOW
        qk = (
            qk * scale_log2
            + _bias(Bias, q_rows, h, dist, mask, stride_bt, stride_bh, EXTENT) * LOG2E_C
        )
        acc, m_i, l_i = _tile_update(
            tl.where(mask, qk, float("-inf")), v, acc, m_i, l_i
        )

    ext_start = 0
    if WINDOW >= 0:
        ext_start = tl.maximum(mblk * BM - WINDOW, 0) // BN * BN
    for start_n in range(ext_start, tl.minimum((mblk + 1) * BM, ext_len), BN):
        offs_n = start_n + tl.arange(0, BN)
        n_mask = offs_n < ext_len
        rows = (q_start + offs_n).to(tl.int64)
        k = tl.load(
            K_Ext + rows[None, :] * stride_kt_ext + kvh * stride_kh_ext + d[:, None],
            n_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            V_Ext + rows[:, None] * stride_vt_ext + kvh * stride_vh_ext + d[None, :],
            n_mask[:, None],
            other=0.0,
        )
        dist = offs_m[:, None] - offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :] & (dist >= 0)
        if WINDOW >= 0:
            mask &= dist <= WINDOW
        qk = (
            tl.dot(q, k) * scale_log2
            + _bias(Bias, q_rows, h, dist, mask, stride_bt, stride_bh, EXTENT) * LOG2E_C
        )
        acc, m_i, l_i = _tile_update(
            tl.where(mask, qk, float("-inf")), v, acc, m_i, l_i
        )

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        O + q_rows[:, None] * stride_ot + h * stride_oh + d[None, :],
        out.to(O.dtype.element_ty),
        m_mask[:, None],
    )
    lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(l_safe)) / LOG2E_C)
    tl.store(LSE + q_rows * stride_lt + h, lse, m_mask)


def rel_extend_attention(
    q,
    k_ext,
    v_ext,
    k_buf,
    v_buf,
    rel_bias,
    o,
    lse,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_extend_len,
    sm_scale,
    *,
    window=-1,
    page_size=1,
):
    """q (tokens, heads, d); k_ext / v_ext (tokens, kv_heads, d); k_buf / v_buf the
    paged cache (slots, kv_heads, d), viewed as (pages, page_size, kv_heads, d);
    rel_bias (tokens, heads, extent); o (tokens, heads, d); lse (tokens, heads), FP32."""
    tokens, heads, d = q.shape
    kv_heads = k_ext.shape[1]
    # Measured on MI350X at 8k-token chunks over 256k keys; short extends waste most
    # of a 256-row tile.
    block_m = 256 if max_extend_len > 128 else 64
    assert lse.stride(1) == 1 and lse.shape == (tokens, heads)
    kb = k_buf.view(-1, page_size, kv_heads, d)
    vb = v_buf.view(-1, page_size, kv_heads, d)
    batch = qo_indptr.shape[0] - 1
    # Requests outermost: a full-graph prefill pads the batch with empty request
    # slots, and with the request innermost the real programs of one request would
    # sit at every batch-th workgroup and pile onto a few CUs.
    grid = (heads, triton.cdiv(max_extend_len, block_m), batch)
    _rel_extend_fwd[grid](
        q, k_ext, v_ext, kb, vb, rel_bias, o, lse,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale * LOG2E,
        q.stride(0), q.stride(1),
        k_ext.stride(0), k_ext.stride(1), v_ext.stride(0), v_ext.stride(1),
        kb.stride(0), kb.stride(1), kb.stride(2), vb.stride(0), vb.stride(1), vb.stride(2),
        rel_bias.stride(0), rel_bias.stride(1),
        o.stride(0), o.stride(1), lse.stride(0),
        heads // kv_heads, d, page_size, rel_bias.shape[-1], window,
        block_m, 32,
        num_warps=8,
        num_stages=1,
    )  # fmt: skip
    return o, lse


@triton.jit
def _rel_verify_split(
    Q,
    K_Ext,
    V_Ext,
    K_Buf,
    V_Buf,
    Bias,
    Mid_O,
    Mid_L,
    qo_indptr,
    kv_indptr,
    kv_indices,
    scale_log2,
    stride_qt,
    stride_qh,
    stride_kt_ext,
    stride_kh_ext,
    stride_vt_ext,
    stride_vh_ext,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_bt,
    stride_bh,
    GROUP: tl.constexpr,
    LP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """One split of the prefix keys (split 0 also takes the extend's own keys) for all
    query heads of one KV head: rows are (head, query) pairs, row = g * LP + i."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    q_start = tl.load(qo_indptr + b).to(tl.int32)
    ext_len = tl.load(qo_indptr + b + 1).to(tl.int32) - q_start
    kv_start = tl.load(kv_indptr + b).to(tl.int32)
    prefix = tl.load(kv_indptr + b + 1).to(tl.int32) - kv_start
    chunk = tl.maximum(
        tl.cdiv(tl.cdiv(prefix, SPLITS), MIN_CHUNK) * MIN_CHUNK, MIN_CHUNK
    )
    s0 = split * chunk
    s1 = tl.minimum(s0 + chunk, prefix)

    r = tl.arange(0, BM)
    g = r // LP
    i = r % LP
    m_mask = (g < GROUP) & (i < ext_len)
    h = kvh * GROUP + g
    q_rows = (q_start + i).to(tl.int64)
    d = tl.arange(0, D)
    q = tl.load(
        Q + q_rows[:, None] * stride_qt + h[:, None] * stride_qh + d[None, :],
        m_mask[:, None],
        other=0.0,
    )
    q_pos = prefix + i
    acc = tl.zeros((BM, D), tl.float32)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)

    # Prefix tiles farther than the extent from every query: no bias, no masks.
    far_end = 0
    if WINDOW < 0:
        far_end = tl.maximum(prefix - EXTENT + 1, 0) // BN * BN
    for start_n in range(s0, tl.minimum(s1, far_end), BN):
        qk, v = _prefix_tile(
            q, K_Buf, V_Buf, kv_indices, kv_start, start_n, kvh, d,
            stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
            tl.full((BN,), True, tl.int1), PAGE, BN,
        )  # fmt: skip
        acc, m_i, l_i = _tile_update(qk * scale_log2, v, acc, m_i, l_i)

    near_start = tl.maximum(s0, far_end)
    if WINDOW >= 0:
        near_start = tl.maximum(near_start, tl.maximum(prefix - WINDOW, 0) // BN * BN)
    for start_n in range(near_start, s1, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_mask = offs_n < s1
        qk, v = _prefix_tile(
            q, K_Buf, V_Buf, kv_indices, kv_start, start_n, kvh, d,
            stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
            n_mask, PAGE, BN,
        )  # fmt: skip
        dist = q_pos[:, None] - offs_n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]
        if WINDOW >= 0:
            mask &= dist <= WINDOW
        bias = tl.load(
            Bias + q_rows[:, None] * stride_bt + h[:, None] * stride_bh + dist,
            mask & (dist < EXTENT),
            other=0.0,
        ).to(tl.float32)
        qk = qk * scale_log2 + bias * LOG2E_C
        acc, m_i, l_i = _tile_update(
            tl.where(mask, qk, float("-inf")), v, acc, m_i, l_i
        )

    if split == 0:
        for start_e in range(0, ext_len, BN):
            offs_n = start_e + tl.arange(0, BN)
            n_mask = offs_n < ext_len
            rows = (q_start + offs_n).to(tl.int64)
            k = tl.load(
                K_Ext
                + rows[None, :] * stride_kt_ext
                + kvh * stride_kh_ext
                + d[:, None],
                n_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                V_Ext
                + rows[:, None] * stride_vt_ext
                + kvh * stride_vh_ext
                + d[None, :],
                n_mask[:, None],
                other=0.0,
            )
            dist = i[:, None] - offs_n[None, :]
            mask = m_mask[:, None] & n_mask[None, :]
            if CAUSAL:
                mask &= dist >= 0
            if WINDOW >= 0:
                mask &= dist <= WINDOW
            bias = tl.load(
                Bias + q_rows[:, None] * stride_bt + h[:, None] * stride_bh + dist,
                mask & (dist >= 0) & (dist < EXTENT),
                other=0.0,
            ).to(tl.float32)
            qk = tl.dot(q, k) * scale_log2 + bias * LOG2E_C
            acc, m_i, l_i = _tile_update(
                tl.where(mask, qk, float("-inf")), v, acc, m_i, l_i
            )

    base = ((b * tl.num_programs(1) + kvh) * SPLITS + split).to(tl.int64) * BM
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(Mid_O + (base + r)[:, None] * D + d[None, :], acc / l_safe[:, None])
    tl.store(
        Mid_L + base + r, tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(l_safe))
    )


@triton.jit
def _rel_verify_combine(
    Mid_O,
    Mid_L,
    O,
    LSE,
    qo_indptr,
    stride_ot,
    stride_oh,
    stride_lt,
    GROUP: tl.constexpr,
    LP: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    Gate=None,
    Sink=None,
    Y=None,
    stride_gt=0,
    stride_st=0,
    stride_yt=0,
    GATE_SINK: tl.constexpr = False,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    r = tl.program_id(2)
    g = r // LP
    i = r % LP
    q_start = tl.load(qo_indptr + b).to(tl.int32)
    ext_len = tl.load(qo_indptr + b + 1).to(tl.int32) - q_start
    if (g >= GROUP) | (i >= ext_len):
        return
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    base = ((b * tl.num_programs(1) + kvh) * SPLITS + s).to(tl.int64) * BM + r
    lse = tl.load(Mid_L + base)
    m = tl.max(lse, 0)
    w = tl.exp2(lse - m)
    o = tl.sum(
        tl.load(Mid_O + base[:, None] * D + d[None, :]) * w[:, None], 0
    ) / tl.sum(w, 0)
    row = (q_start + i).to(tl.int64)
    h = kvh * GROUP + g
    lse_out = (m + tl.log2(tl.sum(w, 0))) / LOG2E_C
    tl.store(O + row * stride_ot + h * stride_oh + d, o.to(O.dtype.element_ty))
    tl.store(LSE + row * stride_lt + h, lse_out)
    if GATE_SINK:
        # gate_sink's _gate_sink_kernel on the stored output: o * sigmoid(gate) *
        # sigmoid(lse - sink), from o rounded to O's dtype.
        x = o.to(O.dtype.element_ty).to(tl.float32)
        gate = tl.load(Gate + row * stride_gt + h * D + d).to(tl.float32)
        sink = tl.load(Sink + row * stride_st + h).to(tl.float32)
        result = x * _sigmoid(gate)
        result = result * _sigmoid(lse_out - sink)
        tl.store(Y + row * stride_yt + h * D + d, result.to(Y.dtype.element_ty))


VERIFY_MIN_CHUNK = 64
# Extends up to this many tokens per request take rel_verify_attention.
REL_VERIFY_MAX_EXTEND = 16


def rel_verify_attention(
    q,
    k_ext,
    v_ext,
    k_buf,
    v_buf,
    rel_bias,
    o,
    lse,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_extend_len,
    sm_scale,
    *,
    window=-1,
    page_size=1,
    causal=True,
    gate_sink=None,
):
    """rel_extend_attention for short extends (MTP verify): split-KV over the prefix,
    all query heads of a KV head in one tile. Same arguments and results. The grid is
    fixed by the batch and max_extend_len (graph-safe); splits beyond a request's
    prefix (MIN_CHUNK keys each at least) exit after writing an empty partial.
    causal=False makes the extend's own keys bidirectional (MTP draft blocks).
    gate_sink=(gate, sinks, y) also writes gate_sink(o, gate, lse, sinks) into y
    (tokens, heads * d), bitwise, from the same reduction."""
    tokens, heads, d = q.shape
    kv_heads = k_ext.shape[1]
    group = heads // kv_heads
    lp = triton.next_power_of_2(max_extend_len)
    bm = max(16, triton.next_power_of_2(group * lp))
    batch = qo_indptr.shape[0] - 1
    # Swept on MI350X (8-token verify, 8k-128k prefixes): long full-attention
    # prefixes at small batch want more splits and wider key tiles.
    small = window < 0 and batch * kv_heads <= 4
    splits = 128 if small else 32
    block_n = 64 if small else 32
    kb = k_buf.view(-1, page_size, kv_heads, d)
    vb = v_buf.view(-1, page_size, kv_heads, d)
    mid_o = torch.empty(
        (batch, kv_heads, splits, bm, d), dtype=torch.float32, device=q.device
    )
    mid_l = torch.empty(
        (batch, kv_heads, splits, bm), dtype=torch.float32, device=q.device
    )
    _rel_verify_split[(batch, kv_heads, splits)](
        q, k_ext, v_ext, kb, vb, rel_bias, mid_o, mid_l,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale * LOG2E,
        q.stride(0), q.stride(1),
        k_ext.stride(0), k_ext.stride(1), v_ext.stride(0), v_ext.stride(1),
        kb.stride(0), kb.stride(1), kb.stride(2), vb.stride(0), vb.stride(1), vb.stride(2),
        rel_bias.stride(0), rel_bias.stride(1),
        group, lp, d, page_size, rel_bias.shape[-1], window,
        splits, VERIFY_MIN_CHUNK, causal, bm, block_n,
        num_warps=4,
    )  # fmt: skip
    epilogue = {}
    if gate_sink is not None:
        gate, sinks, y = gate_sink
        assert gate.stride(-1) == 1 and sinks.stride(-1) == 1 and y.stride(-1) == 1
        epilogue = dict(
            Gate=gate, Sink=sinks, Y=y, stride_gt=gate.stride(0),
            stride_st=sinks.stride(0), stride_yt=y.stride(0), GATE_SINK=True,
        )  # fmt: skip
    _rel_verify_combine[(batch, kv_heads, group * lp)](
        mid_o, mid_l, o, lse, qo_indptr,
        o.stride(0), o.stride(1), lse.stride(0),
        group, lp, d, splits, bm, **epilogue,
    )  # fmt: skip
    return o, lse
