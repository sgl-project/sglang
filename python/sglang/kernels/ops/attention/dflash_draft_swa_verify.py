"""Split-KV, GQA-packed sliding-window attention for the DFlash draft block.

All q heads sharing a KV head are packed into one tile, and the window is split
across programs whose partials are merged with a log-sum-exp combine (which
also runs the causal block-block part). Masking matches ``extend_attention_fwd``
with ``SLIDING_WINDOW_SIZE = W`` and ``IS_CAUSAL``: query row l of a request
with P window keys sees prefix key n iff ``P + l <= n + W``, block key j iff
``j <= l``. Results equal it up to fp32 summation order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_IS_HIP = is_hip()
_AMD_KW = {"waves_per_eu": 2, "matrix_instr_nonkdim": 16} if _IS_HIP else {}


@triton.jit
def _draft_swa_stage1(
    Q,  # [T, Hq, D] (strided)
    K_Ext,  # [T, Hkv, D] (strided) block keys
    V_Ext,
    stride_kt,
    stride_kht,
    stride_vt,
    stride_vht,
    K_Buffer,  # [slots, Hkv, D]
    V_Buffer,  # [slots, Hkv, Dv]
    qo_indptr,
    kv_indptr,
    kv_indices,
    Att_Out,  # [BS, Hq, N_SPLITS, L_PAD, Dv] fp32
    Att_Lse,  # [BS, Hq, N_SPLITS, L_PAD] fp32
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    SLIDING_WINDOW: tl.constexpr,
    G: tl.constexpr,
    L_PAD: tl.constexpr,
    N_SPLITS: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LB: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    R: tl.constexpr = G * L_PAD

    offs_r = tl.arange(0, R)
    row_l = offs_r % L_PAD
    row_h = kvh * G + offs_r // L_PAD
    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)

    q_start = tl.load(qo_indptr + b)
    l_ext = tl.load(qo_indptr + b + 1) - q_start
    mask_r = row_l < l_ext
    kv_start = tl.load(kv_indptr + b)
    P = tl.load(kv_indptr + b + 1) - kv_start

    per_split = tl.cdiv(tl.cdiv(P, N_SPLITS), BLOCK_N) * BLOCK_N
    n_lo = per_split * s
    n_hi = tl.minimum(n_lo + per_split, P)
    if SLIDING_WINDOW > 0:
        # Keys below P - W are invisible to every row (row 0 sees n >= P - W).
        n_lo = tl.maximum(n_lo, (P - SLIDING_WINDOW) // BLOCK_N * BLOCK_N)

    m_i = tl.zeros([R], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([R], dtype=tl.float32)
    acc = tl.zeros([R, DV], dtype=tl.float32)

    if s == N_SPLITS:
        # Extra split: causal block-block attention (keys = the L block tokens).
        q = tl.load(
            Q
            + (q_start + row_l)[:, None] * stride_qt
            + row_h[:, None] * stride_qh
            + offs_d[None, :],
            mask=mask_r[:, None],
            other=0.0,
        )
        offs_j = tl.arange(0, LB)
        mask_j = offs_j < l_ext
        ke = tl.load(
            K_Ext
            + (q_start + offs_j)[None, :] * stride_kt
            + kvh * stride_kht
            + offs_d[:, None],
            mask=mask_j[None, :],
            other=0.0,
        )
        qk = tl.dot(q, ke.to(q.dtype)) * sm_scale  # [R, L_PAD]
        vis = mask_r[:, None] & mask_j[None, :] & (offs_j[None, :] <= row_l[:, None])
        qk = tl.where(vis, qk, float("-inf"))
        ve = tl.load(
            V_Ext
            + (q_start + offs_j)[:, None] * stride_vt
            + kvh * stride_vht
            + offs_dv[None, :],
            mask=mask_j[:, None],
            other=0.0,
        )
        m_i = tl.max(qk, 1)
        m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
        p = tl.exp(qk - m_safe[:, None])
        l_i = tl.sum(p, 1)
        acc = tl.dot(p.to(ve.dtype), ve)
    elif n_hi > n_lo:
        q = tl.load(
            Q
            + (q_start + row_l)[:, None] * stride_qt
            + row_h[:, None] * stride_qh
            + offs_d[None, :],
            mask=mask_r[:, None],
            other=0.0,
        )
        q = q.to(K_Buffer.dtype.element_ty)
        for start_n in tl.range(n_lo, n_hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_hi
            loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)
            k = tl.load(
                K_Buffer
                + loc[None, :] * stride_kbs
                + kvh * stride_kh
                + offs_d[:, None],
                mask=mask_n[None, :],
                other=0.0,
            )
            qk = tl.dot(q, k) * sm_scale  # [R, BLOCK_N]
            vis = mask_n[None, :] & mask_r[:, None]
            if SLIDING_WINDOW > 0:
                vis = vis & ((P + row_l)[:, None] <= offs_n[None, :] + SLIDING_WINDOW)
            qk = tl.where(vis, qk, float("-inf"))
            v = tl.load(
                V_Buffer
                + loc[:, None] * stride_vbs
                + kvh * stride_vh
                + offs_dv[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    has = l_i > 0.0
    out = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    tl.store(
        Att_Out
        + b * stride_ob
        + row_h[:, None] * stride_oh
        + s * stride_os
        + row_l[:, None] * stride_ol
        + offs_dv[None, :],
        out,
        mask=mask_r[:, None],
    )
    tl.store(
        Att_Lse + b * stride_lb + row_h * stride_lh + s * stride_ls + row_l,
        lse,
        mask=mask_r,
    )


@triton.jit
def _draft_swa_stage2(
    Att_Out,
    Att_Lse,
    O,
    qo_indptr,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_ot,
    stride_oh_out,
    L_PAD: tl.constexpr,
    NS: tl.constexpr,  # N_SPLITS + 1 (last = block-block part)
    NS_PAD: tl.constexpr,
    DV: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_l = tl.arange(0, L_PAD)
    offs_s = tl.arange(0, NS_PAD)
    offs_dv = tl.arange(0, DV)
    q_start = tl.load(qo_indptr + b)
    l_ext = tl.load(qo_indptr + b + 1) - q_start
    mask_l = offs_l < l_ext
    lse = tl.load(
        Att_Lse
        + b * stride_lb
        + h * stride_lh
        + offs_s[:, None] * stride_ls
        + offs_l[None, :],
        mask=(offs_s[:, None] < NS) & mask_l[None, :],
        other=float("-inf"),
    )
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    den = tl.sum(tl.exp(lse - m_safe[None, :]), 0)
    o = tl.zeros([L_PAD, DV], dtype=tl.float32)
    base_lse = Att_Lse + b * stride_lb + h * stride_lh + offs_l
    base_ao = (
        Att_Out
        + b * stride_ob
        + h * stride_oh
        + offs_l[:, None] * stride_ol
        + offs_dv[None, :]
    )
    for si in range(NS):
        lse_s = tl.load(base_lse + si * stride_ls, mask=mask_l, other=float("-inf"))
        w_s = tl.exp(lse_s - m_safe)
        ao_s = tl.load(
            base_ao + si * stride_os,
            mask=mask_l[:, None] & (lse_s > float("-inf"))[:, None],
            other=0.0,
        )
        o += ao_s * w_s[:, None]
    o = o / tl.where(den > 0, den, 1.0)[:, None]
    tl.store(
        O
        + (q_start + offs_l)[:, None] * stride_ot
        + h * stride_oh_out
        + offs_dv[None, :],
        o.to(O.dtype.element_ty),
        mask=mask_l[:, None],
    )


_SCRATCH = {}


def _scratch(max_bs, hq, n_splits, l_pad, dv, device):
    key = (hq, n_splits, l_pad, dv, str(device))
    cur = _SCRATCH.get(key)
    if cur is None or cur[0].shape[0] < max_bs:
        out = torch.empty(
            (max_bs, hq, n_splits, l_pad, dv), dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (max_bs, hq, n_splits, l_pad), dtype=torch.float32, device=device
        )
        cur = (out, lse)
        _SCRATCH[key] = cur
    return cur


def can_handle(
    q,
    k,
    v,
    k_buffer,
    v_buffer,
    qo_indptr,
    max_len_extend,
    *,
    causal,
    sinks,
    logit_cap,
    xai_temperature_len,
    score_mod,
) -> bool:
    if not causal or sinks is not None or score_mod is not None:
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return False
    hq, hkv, d, dv = q.shape[1], k.shape[1], q.shape[2], v.shape[2]
    if hkv == 0 or hq % hkv:
        return False
    if k.shape[2] != d or k_buffer.shape[2] != d or v_buffer.shape[2] != dv:
        return False
    if k_buffer.shape[1] != hkv or v_buffer.shape[1] != hkv:
        return False
    if d not in (64, 128, 256) or dv not in (64, 128, 256):
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if k_buffer.stride(-1) != 1 or v_buffer.stride(-1) != 1:
        return False
    if k_buffer.dtype not in (torch.bfloat16, torch.float16):
        return False
    try:
        mle = int(max_len_extend)
    except (TypeError, ValueError):
        return False
    bs = qo_indptr.shape[0] - 1
    if bs < 1 or mle < 1 or mle > 16 or q.shape[0] != bs * mle:
        return False
    return True


def draft_swa_verify_fwd(
    q,
    k,
    v,
    o,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
    sm_scale,
    sliding_window_size,
    max_bs,
    n_splits=None,
    block_n=64,
    num_warps=4,
):
    """q: [T, Hq, D]; k/v: [T, Hkv, D] (may be strided column slices);
    o: [T, Hq, Dv] output. kv_indptr/kv_indices describe the (window) prefix.
    """
    bs = qo_indptr.shape[0] - 1
    hq, hkv, d, dv = q.shape[1], k.shape[1], q.shape[2], v.shape[2]
    g = hq // hkv
    l_pad = triton.next_power_of_2(int(max_len_extend))
    if n_splits is None:
        n_splits = 16 if bs <= 8 else 8
    max_bs = max(int(max_bs or bs), bs)
    att_out, att_lse = _scratch(max_bs, hq, n_splits + 1, l_pad, dv, q.device)
    sw = (
        int(sliding_window_size)
        if sliding_window_size and sliding_window_size > 0
        else 0
    )
    _draft_swa_stage1[(bs, hkv, n_splits + 1)](
        q,
        k,
        v,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_out.stride(3),
        att_lse.stride(0),
        att_lse.stride(1),
        att_lse.stride(2),
        SLIDING_WINDOW=sw,
        G=g,
        L_PAD=l_pad,
        N_SPLITS=n_splits,
        D=d,
        DV=dv,
        BLOCK_N=block_n,
        LB=max(16, l_pad),
        num_warps=num_warps,
        num_stages=1,
        **_AMD_KW,
    )
    _draft_swa_stage2[(bs, hq)](
        att_out,
        att_lse,
        o,
        qo_indptr,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_out.stride(3),
        att_lse.stride(0),
        att_lse.stride(1),
        att_lse.stride(2),
        o.stride(0),
        o.stride(1),
        L_PAD=l_pad,
        NS=n_splits + 1,
        NS_PAD=triton.next_power_of_2(n_splits + 1),
        DV=dv,
        num_warps=2,
        num_stages=1,
    )
    return o
