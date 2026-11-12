# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Single-pass split-KV extend attention for the DFLASH MLA draft call (absorbed MLA, Lq 576 / Lv 512).

Two launches replace the serial prefix-tile chain: kernel A scores every 64-key rail tile and the draft block's own keys in
parallel (one CTA per (head, tile, sequence)) and writes each tile's (row max, row sum, p.V) to an fp32 workspace; kernel B
combines the tiles in one fixed reduction tree (deterministic, no atomics). The masks, scales and operand dtypes are the
stock serial kernel's (extend_attention.py on CUDA): only the softmax is reassociated (tile-local maxima), so this is a
reorder of the stock arithmetic, not an approximation.

hp=True takes the bf16 operand form for the rail dots (fp8 K / V upcast to bf16, q and p kept bf16) instead of the CUDA
form (q and p cast to fp8): fewer roundings than stock.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _p1_softmax_pv_store(qk, final_mask, v, v_scale, WS_ACC, WS_ML, ws_row, offs_m, offs_dv, mask_m, mask_dv,
                         BLOCK_DV: tl.constexpr, EXACT: tl.constexpr, HP: tl.constexpr):
    qk = tl.where(final_mask, qk, float("-inf"))
    row_max = tl.max(qk, 1)
    m_t = tl.where(row_max == float("-inf"), -1e20, row_max)
    p = tl.exp(qk - m_t[:, None])
    l_t = tl.sum(p, 1)
    if EXACT:
        acc = tl.dot(p, v.to(tl.float32), input_precision="ieee")
    elif HP:
        acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
    else:
        acc = tl.dot(p.to(v.dtype), v)
    acc = acc * v_scale
    tl.store(WS_ACC + (ws_row + offs_m[:, None]) * BLOCK_DV + offs_dv[None, :], acc,
             mask=mask_m[:, None] & mask_dv[None, :])
    tl.store(WS_ML + (ws_row + offs_m) * 2, m_t, mask=mask_m)
    tl.store(WS_ML + (ws_row + offs_m) * 2 + 1, l_t, mask=mask_m)


@triton.jit
def _p1_partial_kernel(
    Q, K_Ext, V_Ext, K_Buf, V_Buf, qo_indptr, kv_indptr, kv_indices, WS_ACC, WS_ML,
    sm_scale, k_scale, v_scale, kv_group_num, num_heads, NTS,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh,
    stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh,
    SLIDING_WINDOW_SIZE: tl.constexpr, Lq: tl.constexpr, Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, EXACT: tl.constexpr, HP: tl.constexpr, GDC_TRIGGER: tl.constexpr,
    GDC_WAIT: tl.constexpr,
):
    cur_head = tl.program_id(0)
    tile = tl.program_id(1)
    cur_seq = tl.program_id(2)
    if GDC_WAIT:
        tl.extra.cuda.gdc_wait()  # launched with PDL: the predecessor's q / k / KV rows are visible past this point
    if GDC_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    cur_kv_head = cur_head // kv_group_num
    q_start = tl.load(qo_indptr + cur_seq)
    q_len = tl.load(qo_indptr + cur_seq + 1) - q_start
    kv_start = tl.load(kv_indptr + cur_seq)
    L = tl.load(kv_indptr + cur_seq + 1) - kv_start
    n_rail = (L + BLOCK_N - 1) // BLOCK_N
    if (tile >= n_rail) & (tile < NTS - 1):
        return

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv
    q_row = (q_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh
    q = tl.load(Q + q_row + offs_d[None, :], mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    qpe = tl.load(Q + q_row + offs_dpe[None, :], mask=mask_m[:, None], other=0.0)
    ws_row = ((cur_seq * num_heads + cur_head) * NTS + tile) * BLOCK_M

    if tile == NTS - 1:
        # The block's own keys (bf16 k_ext / v_ext), causal inside the block. These names differ from the rail
        # branch's: a name assigned in both branches of a Triton if must have one type (bf16 / i32 here, fp8 / i64 there).
        mask_n = offs_n < q_len
        final_mask = mask_m[:, None] & mask_n[None, :]
        final_mask &= offs_m[:, None] >= offs_n[None, :]  # causal (the only mode this module accepts)
        if SLIDING_WINDOW_SIZE > 0:
            final_mask &= offs_m[:, None] <= offs_n[None, :] + SLIDING_WINDOW_SIZE
        kx_col = (q_start + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh
        kx = tl.load(K_Ext + kx_col + offs_d[:, None], mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        kxpe = tl.load(K_Ext + kx_col + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
        if EXACT:
            qk = tl.dot(q.to(tl.float32), kx.to(tl.float32), input_precision="ieee")
            qk += tl.dot(qpe.to(tl.float32), kxpe.to(tl.float32), input_precision="ieee")
        else:
            qk = tl.dot(q, kx, out_dtype=tl.float32)
            qk += tl.dot(qpe, kxpe)
        qk *= sm_scale
        vx = tl.load(V_Ext + (q_start + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_dv[None, :],
                     mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        _p1_softmax_pv_store(qk, final_mask, vx, 1.0, WS_ACC, WS_ML, ws_row, offs_m, offs_dv, mask_m, mask_dv,
                             BLOCK_DV, EXACT, HP)
    else:
        # one 64-key tile of the rail (fp8 pool rows through kv_indices)
        start_n = tile * BLOCK_N
        mask_n = (start_n + offs_n) < L
        final_mask = mask_m[:, None] & mask_n[None, :]
        if SLIDING_WINDOW_SIZE > 0:
            final_mask &= (L + offs_m[:, None]) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
        kv_loc = tl.load(kv_indices + kv_start + start_n + offs_n, mask=mask_n, other=0).to(tl.int64)
        k_col = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh
        k = tl.load(K_Buf + k_col + offs_d[:, None], mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        kpe = tl.load(K_Buf + k_col + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
        if EXACT:
            qk = tl.dot(q.to(tl.float32), k.to(tl.float32), input_precision="ieee")
            qk += tl.dot(qpe.to(tl.float32), kpe.to(tl.float32), input_precision="ieee")
        elif HP:
            qk = tl.dot(q, k.to(q.dtype))
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
        else:
            qk = tl.dot(q.to(k.dtype), k)
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale * k_scale
        v = tl.load(V_Buf + kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :],
                    mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        _p1_softmax_pv_store(qk, final_mask, v, v_scale, WS_ACC, WS_ML, ws_row, offs_m, offs_dv, mask_m, mask_dv,
                             BLOCK_DV, EXACT, HP)


@triton.jit
def _p1h_partial_kernel(
    Q, K_Ext, V_Ext, K_Buf, V_Buf, qo_indptr, kv_indptr, kv_indices, WS_ACC, WS_ML,
    sm_scale, k_scale, v_scale, num_heads, NTS,
    stride_qbs, stride_qh, stride_kbs, stride_vbs, stride_buf_kbs, stride_buf_vbs,
    SLIDING_WINDOW_SIZE: tl.constexpr, Lq: tl.constexpr, Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr, BLOCK_DV: tl.constexpr, WS_M: tl.constexpr,
    MPH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TPS: tl.constexpr, DV_SPLIT: tl.constexpr, HPG: tl.constexpr,
    GDC_TRIGGER: tl.constexpr, GDC_WAIT: tl.constexpr,
):
    """Head-packed partial pass (one KV head, MQA): one CTA per (slot, sequence, value chunk) holds every query head's
    rows (row r = head r // MPH, block row r % MPH), so each K / V tile is read once for all heads. Slot `split` covers
    rail tiles [split * TPS, split * TPS + TPS) (online softmax across them); the last slot is the block's own keys.
    Same operand dtypes as _p1_partial_kernel with EXACT = HP = False (q / p cast to fp8 for the rail dots); writes the
    same workspace layout (WS_M rows per (sequence, head, slot)), so _p1_combine_kernel reads it unchanged."""
    split = tl.program_id(0)
    cur_seq = tl.program_id(1)
    chunk = tl.program_id(2) % DV_SPLIT
    hgrp = tl.program_id(2) // DV_SPLIT
    if GDC_WAIT:
        tl.extra.cuda.gdc_wait()
    if GDC_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    q_start = tl.load(qo_indptr + cur_seq)
    q_len = tl.load(qo_indptr + cur_seq + 1) - q_start
    kv_start = tl.load(kv_indptr + cur_seq)
    L = tl.load(kv_indptr + cur_seq + 1) - kv_start
    n_rail = (L + BLOCK_N - 1) // BLOCK_N
    n_split = (n_rail + TPS - 1) // TPS
    if (split >= n_split) & (split < NTS - 1):
        return

    offs_r = tl.arange(0, BLOCK_M)
    head = hgrp * HPG + offs_r // MPH
    row = offs_r % MPH
    valid_r = (offs_r // MPH < HPG) & (head < num_heads) & (row < q_len)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    offs_dv = chunk * (BLOCK_DV // DV_SPLIT) + tl.arange(0, BLOCK_DV // DV_SPLIT)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv
    q_row = (q_start + row[:, None]) * stride_qbs + head[:, None] * stride_qh
    q = tl.load(Q + q_row + offs_d[None, :], mask=valid_r[:, None] & mask_d[None, :], other=0.0)
    qpe = tl.load(Q + q_row + offs_dpe[None, :], mask=valid_r[:, None], other=0.0)
    ws_row = ((cur_seq * num_heads + head) * NTS + split) * WS_M + row

    if split == NTS - 1:
        mask_n = offs_n < q_len
        final_mask = valid_r[:, None] & mask_n[None, :]
        final_mask &= row[:, None] >= offs_n[None, :]
        if SLIDING_WINDOW_SIZE > 0:
            final_mask &= row[:, None] <= offs_n[None, :] + SLIDING_WINDOW_SIZE
        kx_col = (q_start + offs_n[None, :]) * stride_kbs
        kx = tl.load(K_Ext + kx_col + offs_d[:, None], mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        kxpe = tl.load(K_Ext + kx_col + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, kx, out_dtype=tl.float32)
        qk += tl.dot(qpe, kxpe)
        qk *= sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))
        row_max = tl.max(qk, 1)
        m_i = tl.where(row_max == float("-inf"), -1e20, row_max)
        p = tl.exp(qk - m_i[:, None])
        l_i = tl.sum(p, 1)
        vx = tl.load(V_Ext + (q_start + offs_n[:, None]) * stride_vbs + offs_dv[None, :],
                     mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        acc = tl.dot(p.to(vx.dtype), vx)
    else:
        m_i = tl.full([BLOCK_M], -1e20, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DV // DV_SPLIT], dtype=tl.float32)
        t_end = tl.minimum(split * TPS + TPS, n_rail)
        for t in range(split * TPS, t_end):
            start_n = t * BLOCK_N
            mask_n = (start_n + offs_n) < L
            final_mask = valid_r[:, None] & mask_n[None, :]
            if SLIDING_WINDOW_SIZE > 0:
                final_mask &= (L + row[:, None]) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            kv_loc = tl.load(kv_indices + kv_start + start_n + offs_n, mask=mask_n, other=0).to(tl.int64)
            k_col = kv_loc[None, :] * stride_buf_kbs
            k = tl.load(K_Buf + k_col + offs_d[:, None], mask=mask_n[None, :] & mask_d[:, None], other=0.0)
            kpe = tl.load(K_Buf + k_col + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q.to(k.dtype), k)
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
            qk *= sm_scale * k_scale
            qk = tl.where(final_mask, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V_Buf + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :],
                        mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
            pv = tl.dot(p.to(v.dtype), v)
            acc = acc * alpha[:, None] + pv
            m_i = m_new
        acc = acc * v_scale
    tl.store(WS_ACC + ws_row[:, None] * BLOCK_DV + offs_dv[None, :], acc, mask=valid_r[:, None] & mask_dv[None, :])
    if chunk == 0:
        tl.store(WS_ML + ws_row * 2, m_i, mask=valid_r)
        tl.store(WS_ML + ws_row * 2 + 1, l_i, mask=valid_r)


@triton.jit
def _p1_combine_kernel(
    WS_ACC, WS_ML, O, qo_indptr, kv_indptr, num_heads, NTS, stride_obs, stride_oh,
    Lv: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DV: tl.constexpr,
    DV_CHUNK: tl.constexpr, NT_PAD: tl.constexpr, USE_GDC: tl.constexpr, TPS: tl.constexpr = 1, GDC_TRIGGER: tl.constexpr = False,
):
    # TPS (head-packed pass only): rail tiles per workspace slot; slot t < ceil(n_rail / TPS) holds tiles [t * TPS, t * TPS + TPS).
    cur_head = tl.program_id(0) // (BLOCK_DV // DV_CHUNK)
    chunk = tl.program_id(0) % (BLOCK_DV // DV_CHUNK)
    row = tl.program_id(1)
    cur_seq = tl.program_id(2)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()  # kernel A's workspace writes are complete and visible past this point
        if GDC_TRIGGER:
            # combine_trigger: release the PDL successor now; it reads O only after its own griddepcontrol.wait,
            # which waits for this whole grid and its memory operations.
            tl.extra.cuda.gdc_launch_dependents()
    q_start = tl.load(qo_indptr + cur_seq)
    q_len = tl.load(qo_indptr + cur_seq + 1) - q_start
    if row >= q_len:
        return
    L = tl.load(kv_indptr + cur_seq + 1) - tl.load(kv_indptr + cur_seq)
    n_rail = (L + BLOCK_N - 1) // BLOCK_N
    n_slot = (n_rail + TPS - 1) // TPS
    offs_dv = chunk * DV_CHUNK + tl.arange(0, DV_CHUNK)
    o_ptrs = O + (q_start + row) * stride_obs + cur_head * stride_oh + offs_dv
    if n_slot > NTS - 1:
        # The rail is longer than the grid sized from max_kv_len: fail loud (NaN rows), never a silently cut rail.
        tl.store(o_ptrs, tl.full([DV_CHUNK], float("nan"), tl.float32), mask=offs_dv < Lv)
        return
    offs_t = tl.arange(0, NT_PAD)
    valid_t = (offs_t < n_slot) | (offs_t == NTS - 1)
    ws_rows = ((cur_seq * num_heads + cur_head) * NTS + offs_t) * BLOCK_M + row
    m_t = tl.load(WS_ML + ws_rows * 2, mask=valid_t, other=float("-inf"))
    l_t = tl.load(WS_ML + ws_rows * 2 + 1, mask=valid_t, other=0.0)
    big_m = tl.max(m_t, 0)
    w = tl.exp(m_t - big_m)
    deno = tl.sum(w * l_t, 0)
    acc_t = tl.load(WS_ACC + ws_rows[:, None] * BLOCK_DV + offs_dv[None, :], mask=valid_t[:, None], other=0.0)
    acc = tl.sum(acc_t * w[:, None], 0)
    deno_safe = tl.where(deno == 0.0, 1.0, deno)
    tl.store(o_ptrs, acc / deno_safe, mask=offs_dv < Lv)


def _p1h_launch(num_heads: int, mph: int):
    """(num_warps, head groups) of the head-packed partial pass: 16 rows per CTA (two heads of an 8-row block) and 8 warps, the
    per-head pass's tile shapes and launch width, so every dot and row reduction runs in the same layout and order (bitwise
    equal outputs). Wider packings were faster on the target GPU but not bitwise: 4 warps at 16 rows differed by 1 ulp,
    and at 64 rows Triton 3.9's tcgen05 P.V dot at N = 512 faults."""
    return 8, triton.cdiv(num_heads * mph, 16)


def _p1h_fwd(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
             max_len_extend, k_scale, v_scale, sm_scale, nts, sliding_window_size, pdl, pdl_wait_first, tps=1, launch=None):
    """Head-packed partial pass (SGLANG_OPT_DFLASH_DRAFT_P1_HPACK) + the shared combine. Same workspace layout, operand dtypes and
    reduction order as the per-head pass: outputs are bitwise equal to it. launch = (heads per CTA, V split, num_warps,
    num_stages) of a P1_WIDE table entry (same workspace layout and combine; not bitwise to the served launch); None = served."""
    bs = qo_indptr.shape[0] - 1
    num_heads = q_extend.shape[1]
    mph = triton.next_power_of_2(int(max_len_extend))
    nw, hg = _p1h_launch(num_heads, mph)
    hpg = triton.cdiv(num_heads, hg)
    dvs, ns = 1, 1
    if launch is not None:
        hpg, dvs, nw, ns = (int(x) for x in launch)
        hg = triton.cdiv(num_heads, hpg)
    block_m = max(16, triton.next_power_of_2(hpg * mph))
    dev = q_extend.device
    ws_acc = torch.empty(bs * num_heads * nts * 16 * 512, dtype=torch.float32, device=dev)
    ws_ml = torch.empty(bs * num_heads * nts * 16 * 2, dtype=torch.float32, device=dev)
    _p1h_partial_kernel[(nts, bs, hg * dvs)](
        q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, ws_acc, ws_ml,
        sm_scale, k_scale, v_scale, num_heads, nts,
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), v_extend.stride(0), k_buffer.stride(0),
        v_buffer.stride(0),
        SLIDING_WINDOW_SIZE=sliding_window_size, Lq=576, Lv=512, BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=512, WS_M=16,
        MPH=mph, BLOCK_M=block_m, BLOCK_N=64, TPS=int(tps), DV_SPLIT=dvs, HPG=hpg, GDC_TRIGGER=bool(pdl),
        GDC_WAIT=bool(pdl_wait_first), num_warps=nw, num_stages=ns,
        **({"launch_pdl": True} if pdl_wait_first else {}),
    )
    return ws_acc, ws_ml


# tps per draft batch (requests) for the head-packed pass: the fastest tps of {1, 2, 3, 4, 6, 9} at each measured batch on the
# target GPU (6 query heads per rank, rail <= 1,087 rows); a batch between two keys takes the lower key's value.
P1_TPS_AUTO = ((1, 1), (2, 1), (4, 2), (8, 3), (12, 2), (16, 6), (24, 4), (32, 3), (48, 4), (64, 9))


def resolve_p1_tps(setting: str, bs: int) -> int:
    """tps for draft batch bs under SGLANG_OPT_DFLASH_DRAFT_P1_TPS ('1', an integer >= 1, or 'auto')."""
    setting = str(setting).strip().lower()
    if setting == "auto":
        tps = 1
        for key, val in P1_TPS_AUTO:
            if bs >= key:
                tps = val
        return tps
    try:
        tps = int(setting)
    except ValueError:
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_P1_TPS must be 'auto' or an integer >= 1, got {setting!r}") from None
    if tps < 1:
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_P1_TPS must be 'auto' or an integer >= 1, got {setting!r}")
    return tps


# Head-packed pass launch per draft batch for SGLANG_OPT_DFLASH_DRAFT_P1_WIDE=auto: ((lowest bs, highest bs), (heads per CTA, V split,
# num_warps, num_stages, tps, combine DV chunk or 0 = the default rule)). Tuned on the target GPU (6 query heads per rank,
# rail <= 1,087 rows): all 6 heads of a request per CTA, V in two 256-wide halves, 8 warps and 3 stages were the fastest launch
# at bs 4-16; FP64 error within 1.3x of the served launch; 4 stages time the same as 3. A batch outside every range keeps the
# served launch.
P1_WIDE_AUTO: Tuple[Tuple[Tuple[int, int], Tuple[int, int, int, int, int, int]], ...] = (
    ((4, 5), (6, 2, 8, 3, 1, 0)),
    ((6, 8), (6, 2, 8, 3, 2, 0)),
    ((9, 10), (6, 2, 8, 3, 2, 256)),
    ((11, 15), (6, 2, 8, 3, 3, 256)),
    ((16, 16), (6, 2, 8, 3, 4, 256)),
)


def resolve_p1_wide(setting: str, bs: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Launch of the head-packed pass at draft batch bs under SGLANG_OPT_DFLASH_DRAFT_P1_WIDE ('0' = served, 'auto' = table)."""
    setting = str(setting).strip().lower()
    if setting == "0":
        return None
    if setting != "auto":
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_P1_WIDE must be '0' or 'auto', got {setting!r}")
    for (lo, hi), launch in P1_WIDE_AUTO:
        if lo <= bs <= hi:
            return launch
    return None


def dflash_draft_splitkv_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr,
                                       kv_indptr, kv_indices, max_len_extend, k_scale, v_scale, sm_scale, *,
                                       max_kv_len: int, sliding_window_size: int = -1, hp: bool = False,
                                       pdl: bool = False, pdl_wait_first: bool = False, exact: bool = False,
                                       hpack: bool = False, tps: int = 1,
                                       wide: Optional[Tuple[int, int, int, int, int, int]] = None,
                                       combine_trigger: bool = False):
    """Causal extend attention of the absorbed-MLA draft block (q / k_ext rows = the block, rail = fp8 pool rows).

    max_kv_len bounds every sequence's rail and sizes the grid (CUDA-graph safe: the rail length itself is read on
    device); a longer rail writes NaN rows instead of attending a cut rail. Workspace: torch.empty per call (the graph
    pool under capture). pdl: the combine launches with PDL behind the partial pass; pdl_wait_first (needs pdl): the
    partial pass also launches with PDL and waits before its first read. hpack (SGLANG_OPT_DFLASH_DRAFT_P1_HPACK): at draft
    batch > 1 with one KV head and the stock operand dtypes, one CTA scores a rail tile for a group of query heads, so each
    K / V tile is read once per group instead of once per head (bitwise equal outputs). tps (hpack only, wide draft batches): each
    partial-pass slot scores tps consecutive rail tiles with an online softmax, so the grid and the fp32 workspace shrink by ~tps
    (fewer, longer CTAs once bs x head groups fill the SMs); tps > 1 changes the fp32 reduction order (not bitwise to tps = 1).
    combine_trigger (SGLANG_OPT_DFLASH_DRAFT_P1_COMBINE_TRIGGER, needs pdl): the combine releases its PDL dependents right after
    its wait instead of at exit (bitwise equal outputs)."""
    if pdl_wait_first and not pdl:
        raise ValueError("pdl_wait_first needs pdl")
    if combine_trigger and not pdl:
        raise ValueError("combine_trigger needs pdl")
    Lq, Lv = q_extend.shape[-1], v_extend.shape[-1]
    if Lq != 576 or Lv != 512 or max_len_extend > 16 or k_buffer.dim() != 3 or v_buffer.dim() != 3:
        raise NotImplementedError("dflash_draft_splitkv covers the absorbed-MLA DFLASH draft call only")
    bs = qo_indptr.shape[0] - 1
    num_heads = q_extend.shape[1]
    kv_group_num = num_heads // k_extend.shape[1]
    BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV = 16, 64, 512, 64, 512
    nts = triton.cdiv(int(max_kv_len), BLOCK_N) + 1
    dev = q_extend.device
    use_hpack = hpack and bs > 1 and k_extend.shape[1] == 1 and k_buffer.shape[1] == 1 and not exact and not hp
    wide = wide if use_hpack else None
    tps = int(wide[4]) if wide is not None else (int(tps) if use_hpack else 1)
    if tps < 1:
        raise ValueError(f"tps must be >= 1, got {tps}")
    if tps > 1:
        nts = triton.cdiv(triton.cdiv(int(max_kv_len), BLOCK_N), tps) + 1
    if use_hpack:
        ws_acc, ws_ml = _p1h_fwd(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
                                 max_len_extend, k_scale, v_scale, sm_scale, nts, sliding_window_size, pdl,
                                 pdl_wait_first, tps=tps, launch=None if wide is None else wide[:4])
    else:
        ws_acc = torch.empty(bs * num_heads * nts * BLOCK_M * BLOCK_DV, dtype=torch.float32, device=dev)
        ws_ml = torch.empty(bs * num_heads * nts * BLOCK_M * 2, dtype=torch.float32, device=dev)
        _p1_partial_kernel[(num_heads, nts, bs)](
            q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, ws_acc, ws_ml,
            sm_scale, k_scale, v_scale, kv_group_num, num_heads, nts,
            q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1), v_extend.stride(0),
            v_extend.stride(1), k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
            SLIDING_WINDOW_SIZE=sliding_window_size, Lq=Lq, Lv=Lv, BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, EXACT=bool(exact), HP=bool(hp),
            GDC_TRIGGER=bool(pdl), GDC_WAIT=bool(pdl_wait_first), num_warps=8, num_stages=1,
            **({"launch_pdl": True} if pdl_wait_first else {}),
        )
    nt_pad = triton.next_power_of_2(nts)
    dv_chunk = 128 if nt_pad <= 32 else 32
    if wide is not None and int(wide[5]):
        dv_chunk = int(wide[5])
    _p1_combine_kernel[(num_heads * (BLOCK_DV // dv_chunk), max_len_extend, bs)](
        ws_acc, ws_ml, o_extend, qo_indptr, kv_indptr, num_heads, nts, o_extend.stride(0), o_extend.stride(1),
        Lv=Lv, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DV=BLOCK_DV, DV_CHUNK=dv_chunk, NT_PAD=nt_pad,
        USE_GDC=bool(pdl), TPS=tps, GDC_TRIGGER=bool(combine_trigger), num_warps=4, **({"launch_pdl": True} if pdl else {}),
    )
