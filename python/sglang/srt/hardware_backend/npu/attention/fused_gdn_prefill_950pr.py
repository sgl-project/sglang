"""Custom Ascend 950PR (arch35) fused Triton GDN Prefill kernels.

Designed specifically for Ascend 950PR (28 AIC / 56 AIV, 1.43 TB/s HBM, 128 MB L2,
512 KB L1 CBUF, 254 KB UB):
  1. `_fused_qk_l2norm_gating_row_kernel`: Persistent 56-AIV fusion of Q/K L2Norm
     (4-row vectorized tiles) and GDN gating (`g = -exp(A_log) * softplus(a + dt_bias)`,
     `beta = sigmoid(b)`), reading Q/K directly from `mixed_qkv` without copying V,
     and using 2D UB block-transpose (`R_GB=32`) so `g_T [NV, T]` / `beta_T [NV, T]`
     are written as contiguous bursts (eliminating 128B L2 sector write amplification).
  2. GQA-interleaved Prepare kernels (`_fused_gdn_prepare_gqa3_950pr_kernel`,
     `_fused_gdn_prepare_gqa2x2_950pr_kernel`, and generic fallback
     `_fused_gdn_prepare_950pr_kernel`): Single-kernel fusion of local cumsum,
     GQA-deduplicated `K @ K^T` and `Q @ K^T`, single-pass `exp(g_diff)` reused for
     both pre-scaled intra-chunk attention `ATTN [NT, NV, 64, 64]` and `L`, exact
     2-level 8x8 block Euler triangular inversion `(I + L)^-1` in FP16 (`out_dtype=fp16`),
     and WY representation `(W, U)` reading V directly from `mixed_qkv` (zero V copy).
     Interleaving 3 V-heads (`GQA=3`) or 4 V-heads (`2x GQA=2`) hides cross-core
     AIC<->AIV sync latency and deduplicates `K@K^T` and `Q@K^T` across GQA siblings.
  3. `_fused_gdn_fwd_ho_950pr_kernel`: Slim 4-dot `(B * NV)` fully-parallel fusion of
     `ChunkFwdH + ChunkFwdO` (zero inter-sequence SyncAll, zero `Q@K^T` or 64x64 `exp`
     in the recurrent loop), issuing `W @ H^T` and `Q @ H^T` back-to-back, keeping the
     state `h [DV, DK]` on-chip in native CANN layout, reading/writing `ssm_states` pool
     slots in-place, and writing single-point Radix Cache checkpoints in-kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35


@triton.jit(do_not_specialize=["T", "rows_per_prog"])
def _fused_qk_l2norm_gating_row_kernel(
    MIXED_QKV_ptr,  # [T, QKV_DIM] bf16
    A_ptr,  # [T, NV] bf16/fp32
    B_ptr,  # [T, NV] bf16/fp32
    A_LOG_ptr,  # [NV] fp32
    DT_BIAS_ptr,  # [NV] bf16/fp32
    Q_OUT_ptr,  # [T, NK, D] bf16
    K_OUT_ptr,  # [T, NK, D] bf16
    G_T_ptr,  # [NV, T] fp32
    BETA_T_ptr,  # [NV, T] bf16
    T,
    rows_per_prog,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    NV_PAD: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R_GB: tl.constexpr,
    R_QK: tl.constexpr,
):
    pid = tl.program_id(0)
    r0 = pid * rows_per_prog
    r1 = tl.minimum(r0 + rows_per_prog, T)
    offs_nv = tl.arange(0, NV_PAD)
    mask_nv = offs_nv < NV
    a_log = tl.load(A_LOG_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
    neg_exp_a = -tl.exp(a_log)
    dt_b = tl.load(DT_BIAS_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)

    # Pass 1: 2D UB block-transpose for contiguous bursts to G_T [NV, T] and BETA_T [NV, T]
    offs_r = tl.arange(0, R_GB)
    for rb in range(r0, r1, R_GB):
        rows = rb + offs_r
        mask_r = rows < r1
        mask_2d = mask_r[:, None] & mask_nv[None, :]
        ab_offs = rows[:, None] * NV + offs_nv[None, :]
        a_val = tl.load(A_ptr + ab_offs, mask=mask_2d, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + ab_offs, mask=mask_2d, other=0.0).to(tl.float32)
        sp_in = a_val + dt_b[None, :]
        sp = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
        g_val = neg_exp_a[None, :] * sp
        beta_val = tl.sigmoid(b_val).to(BETA_T_ptr.dtype.element_ty)

        mask_t = mask_nv[:, None] & mask_r[None, :]
        gt_offs = offs_nv[:, None] * T + rows[None, :]
        tl.store(G_T_ptr + gt_offs, tl.trans(g_val), mask=mask_t)
        tl.store(BETA_T_ptr + gt_offs, tl.trans(beta_val), mask=mask_t)

    # Pass 2: Multi-row vectorized Q/K L2Norm reading directly from mixed_qkv (zero V copy)
    offs_rqk = tl.arange(0, R_QK)
    offs_hd = tl.arange(0, NK)[:, None] * D + tl.arange(0, D)[None, :]
    for rb in range(r0, r1, R_QK):
        rows_qk = rb + offs_rqk
        mask_qk = rows_qk < r1
        q = tl.load(
            MIXED_QKV_ptr + rows_qk[:, None, None] * QKV_DIM + offs_hd[None, :, :],
            mask=mask_qk[:, None, None],
            other=0.0,
        ).to(tl.float32)
        q_inv = tl.rsqrt(tl.sum(q * q, axis=2, keep_dims=True) + EPS)
        tl.store(
            Q_OUT_ptr + rows_qk[:, None, None] * Q_DIM + offs_hd[None, :, :],
            (q * q_inv).to(Q_OUT_ptr.dtype.element_ty),
            mask=mask_qk[:, None, None],
        )

        k = tl.load(
            MIXED_QKV_ptr
            + rows_qk[:, None, None] * QKV_DIM
            + Q_DIM
            + offs_hd[None, :, :],
            mask=mask_qk[:, None, None],
            other=0.0,
        ).to(tl.float32)
        k_inv = tl.rsqrt(tl.sum(k * k, axis=2, keep_dims=True) + EPS)
        tl.store(
            K_OUT_ptr + rows_qk[:, None, None] * K_DIM + offs_hd[None, :, :],
            (k * k_inv).to(K_OUT_ptr.dtype.element_ty),
            mask=mask_qk[:, None, None],
        )


@triton.jit(do_not_specialize=["T_total"])
def _fused_gdn_prepare_gqa3_950pr_kernel(
    Q_ptr,  # [T_total, NK, D] bf16
    K_ptr,  # [T_total, NK, D] bf16
    MIXED_QKV_ptr,  # [T_total, QKV_DIM] bf16 — zero-copy V read at + Q_DIM + K_DIM
    G_T_ptr,  # [NV, T_total] fp32
    BETA_T_ptr,  # [NV, T_total] bf16
    CU_ptr,  # [B+1] int32
    CHUNK_IDX_ptr,  # [NT, 2] int32
    W_ptr,  # [NT, NV, BT, D] bf16
    U_ptr,  # [NT, NV, BT, D] bf16
    ATTN_ptr,  # [NT, NV, BT, BT] bf16 — unscaled lower-tri attention (scaled in FP32 in Stage 2)
    GCUM_ptr,  # [NT, NV, BT] fp32
    T_total,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """3-way GQA-interleaved Prepare kernel for GQA=3 (e.g. NK=16, NV=48)."""
    pid_t = tl.program_id(0)
    i_kh = tl.program_id(1)
    v0 = i_kh * 3
    v1 = v0 + 1
    v2 = v0 + 2

    i_n = tl.load(CHUNK_IDX_ptr + pid_t * 2).to(tl.int32)
    i_t = tl.load(CHUNK_IDX_ptr + pid_t * 2 + 1).to(tl.int32)
    bos = tl.load(CU_ptr + i_n).to(tl.int32)
    eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
    seq_len = eos - bos

    offs_bt = tl.arange(0, BT)
    offs_t = i_t * BT + offs_bt
    mask_t = offs_t < seq_len
    offs_d = tl.arange(0, D)
    offs_btd = offs_bt[:, None] * D + offs_d[None, :]
    offs_bt2 = offs_bt[:, None] * BT + offs_bt[None, :]
    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    strict_lower = rows > cols
    lower_eq = rows >= cols
    same_blk8 = (rows // 8) == (cols // 8)
    mask_d8 = strict_lower & same_blk8
    mask_off8 = strict_lower & (~same_blk8)
    eye16 = tl.where(rows == cols, 1.0, 0.0).to(tl.float16)
    zero16 = tl.full((BT, BT), 0.0, dtype=tl.float16)

    # Deduplicate K@K^T and Q@K^T across all 3 GQA V-heads!
    t_nk = offs_t[:, None] * (NK * D) + offs_d[None, :]
    b_k = tl.load(K_ptr + (bos * NK + i_kh) * D + t_nk, mask=mask_t[:, None], other=0.0)
    b_q = tl.load(Q_ptr + (bos * NK + i_kh) * D + t_nk, mask=mask_t[:, None], other=0.0)
    b_k_t = tl.trans(b_k)
    b_kkt = tl.dot(b_k, b_k_t)
    b_qk = tl.dot(b_q, b_k_t)

    # Sequential per-head exp(g_diff) keeps only one 16 KB eg matrix live in UB at a time.
    g0 = tl.load(G_T_ptr + v0 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    beta0 = tl.load(
        BETA_T_ptr + v0 * T_total + bos + offs_t, mask=mask_t, other=0.0
    ).to(tl.float32)
    gc0 = tl.cumsum(g0, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v0) * BT + offs_bt, gc0)
    eg0 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc0[:, None] - gc0[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v0) * (BT * BT) + offs_bt2,
        (b_qk * eg0).to(ATTN_ptr.dtype.element_ty),
    )
    l0 = tl.where(strict_lower, b_kkt * beta0[:, None] * eg0, 0.0)
    p0 = tl.where(mask_d8, l0, 0.0).to(tl.float16)
    loff0 = tl.where(mask_off8, l0, 0.0).to(tl.float16)

    g1 = tl.load(G_T_ptr + v1 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    beta1 = tl.load(
        BETA_T_ptr + v1 * T_total + bos + offs_t, mask=mask_t, other=0.0
    ).to(tl.float32)
    gc1 = tl.cumsum(g1, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v1) * BT + offs_bt, gc1)
    eg1 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc1[:, None] - gc1[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v1) * (BT * BT) + offs_bt2,
        (b_qk * eg1).to(ATTN_ptr.dtype.element_ty),
    )
    l1 = tl.where(strict_lower, b_kkt * beta1[:, None] * eg1, 0.0)
    p1 = tl.where(mask_d8, l1, 0.0).to(tl.float16)
    loff1 = tl.where(mask_off8, l1, 0.0).to(tl.float16)

    g2 = tl.load(G_T_ptr + v2 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    beta2 = tl.load(
        BETA_T_ptr + v2 * T_total + bos + offs_t, mask=mask_t, other=0.0
    ).to(tl.float32)
    gc2 = tl.cumsum(g2, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v2) * BT + offs_bt, gc2)
    eg2 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc2[:, None] - gc2[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v2) * (BT * BT) + offs_bt2,
        (b_qk * eg2).to(ATTN_ptr.dtype.element_ty),
    )
    l2 = tl.where(strict_lower, b_kkt * beta2[:, None] * eg2, 0.0)
    p2 = tl.where(mask_d8, l2, 0.0).to(tl.float16)
    loff2 = tl.where(mask_off8, l2, 0.0).to(tl.float16)

    # 3-way interleaved exact 2-level 8x8 Euler inversion in FP16 (hides AIC<->AIV sync latency).
    a0 = eye16 - p0
    a1 = eye16 - p1
    a2 = eye16 - p2
    for _ in range(2):
        p0 = tl.dot(p0, p0, out_dtype=tl.float16)
        p1 = tl.dot(p1, p1, out_dtype=tl.float16)
        p2 = tl.dot(p2, p2, out_dtype=tl.float16)
        a0 = tl.dot(eye16 + p0, a0, out_dtype=tl.float16)
        a1 = tl.dot(eye16 + p1, a1, out_dtype=tl.float16)
        a2 = tl.dot(eye16 + p2, a2, out_dtype=tl.float16)

    p0 = tl.where(mask_off8, tl.dot(a0, loff0, out_dtype=tl.float16), zero16)
    p1 = tl.where(mask_off8, tl.dot(a1, loff1, out_dtype=tl.float16), zero16)
    p2 = tl.where(mask_off8, tl.dot(a2, loff2, out_dtype=tl.float16), zero16)
    a0 = tl.dot(eye16 - p0, a0, out_dtype=tl.float16)
    a1 = tl.dot(eye16 - p1, a1, out_dtype=tl.float16)
    a2 = tl.dot(eye16 - p2, a2, out_dtype=tl.float16)
    for _ in range(2):
        p0 = tl.dot(p0, p0, out_dtype=tl.float16)
        p1 = tl.dot(p1, p1, out_dtype=tl.float16)
        p2 = tl.dot(p2, p2, out_dtype=tl.float16)
        a0 = tl.dot(eye16 + p0, a0, out_dtype=tl.float16)
        a1 = tl.dot(eye16 + p1, a1, out_dtype=tl.float16)
        a2 = tl.dot(eye16 + p2, a2, out_dtype=tl.float16)

    a0_bf = tl.where(lower_eq, a0, zero16).to(tl.bfloat16)
    a1_bf = tl.where(lower_eq, a1, zero16).to(tl.bfloat16)
    a2_bf = tl.where(lower_eq, a2, zero16).to(tl.bfloat16)

    bk_f32 = b_k.to(tl.float32)
    w0 = tl.dot(a0_bf, (bk_f32 * (beta0 * tl.exp(gc0))[:, None]).to(tl.bfloat16))
    w1 = tl.dot(a1_bf, (bk_f32 * (beta1 * tl.exp(gc1))[:, None]).to(tl.bfloat16))
    w2 = tl.dot(a2_bf, (bk_f32 * (beta2 * tl.exp(gc2))[:, None]).to(tl.bfloat16))
    tl.store(
        W_ptr + (pid_t * NV + v0) * (BT * D) + offs_btd, w0.to(W_ptr.dtype.element_ty)
    )
    tl.store(
        W_ptr + (pid_t * NV + v1) * (BT * D) + offs_btd, w1.to(W_ptr.dtype.element_ty)
    )
    tl.store(
        W_ptr + (pid_t * NV + v2) * (BT * D) + offs_btd, w2.to(W_ptr.dtype.element_ty)
    )

    v_base = MIXED_QKV_ptr + bos * QKV_DIM + Q_DIM + K_DIM
    t_qkv = offs_t[:, None] * QKV_DIM + offs_d[None, :]
    bv0 = tl.load(v_base + v0 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    bv1 = tl.load(v_base + v1 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    bv2 = tl.load(v_base + v2 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    u0 = tl.dot(a0_bf, (bv0 * beta0[:, None]).to(tl.bfloat16))
    u1 = tl.dot(a1_bf, (bv1 * beta1[:, None]).to(tl.bfloat16))
    u2 = tl.dot(a2_bf, (bv2 * beta2[:, None]).to(tl.bfloat16))
    tl.store(
        U_ptr + (pid_t * NV + v0) * (BT * D) + offs_btd, u0.to(U_ptr.dtype.element_ty)
    )
    tl.store(
        U_ptr + (pid_t * NV + v1) * (BT * D) + offs_btd, u1.to(U_ptr.dtype.element_ty)
    )
    tl.store(
        U_ptr + (pid_t * NV + v2) * (BT * D) + offs_btd, u2.to(U_ptr.dtype.element_ty)
    )


@triton.jit(do_not_specialize=["T_total"])
def _fused_gdn_prepare_gqa2x2_950pr_kernel(
    Q_ptr,  # [T_total, NK, D] bf16
    K_ptr,  # [T_total, NK, D] bf16
    MIXED_QKV_ptr,  # [T_total, QKV_DIM] bf16
    G_T_ptr,  # [NV, T_total] fp32
    BETA_T_ptr,  # [NV, T_total] bf16
    CU_ptr,  # [B+1] int32
    CHUNK_IDX_ptr,  # [NT, 2] int32
    W_ptr,  # [NT, NV, BT, D] bf16
    U_ptr,  # [NT, NV, BT, D] bf16
    ATTN_ptr,  # [NT, NV, BT, BT] bf16
    GCUM_ptr,  # [NT, NV, BT] fp32
    T_total,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """4-way interleaved Prepare kernel for GQA=2, even NK (e.g. NK=16, NV=32)."""
    pid_t = tl.program_id(0)
    pid_kh2 = tl.program_id(1)
    kh0 = pid_kh2 * 2
    kh1 = kh0 + 1
    v0 = kh0 * 2
    v1 = v0 + 1
    v2 = kh1 * 2
    v3 = v2 + 1

    i_n = tl.load(CHUNK_IDX_ptr + pid_t * 2).to(tl.int32)
    i_t = tl.load(CHUNK_IDX_ptr + pid_t * 2 + 1).to(tl.int32)
    bos = tl.load(CU_ptr + i_n).to(tl.int32)
    eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
    seq_len = eos - bos

    offs_bt = tl.arange(0, BT)
    offs_t = i_t * BT + offs_bt
    mask_t = offs_t < seq_len
    offs_d = tl.arange(0, D)
    offs_btd = offs_bt[:, None] * D + offs_d[None, :]
    offs_bt2 = offs_bt[:, None] * BT + offs_bt[None, :]
    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    strict_lower = rows > cols
    lower_eq = rows >= cols
    same_blk8 = (rows // 8) == (cols // 8)
    mask_d8 = strict_lower & same_blk8
    mask_off8 = strict_lower & (~same_blk8)
    eye16 = tl.where(rows == cols, 1.0, 0.0).to(tl.float16)
    zero16 = tl.full((BT, BT), 0.0, dtype=tl.float16)

    t_nk = offs_t[:, None] * (NK * D) + offs_d[None, :]
    bk0 = tl.load(K_ptr + (bos * NK + kh0) * D + t_nk, mask=mask_t[:, None], other=0.0)
    bk1 = tl.load(K_ptr + (bos * NK + kh1) * D + t_nk, mask=mask_t[:, None], other=0.0)
    bq0 = tl.load(Q_ptr + (bos * NK + kh0) * D + t_nk, mask=mask_t[:, None], other=0.0)
    bq1 = tl.load(Q_ptr + (bos * NK + kh1) * D + t_nk, mask=mask_t[:, None], other=0.0)
    bk0_t = tl.trans(bk0)
    bk1_t = tl.trans(bk1)
    kkt0 = tl.dot(bk0, bk0_t)
    kkt1 = tl.dot(bk1, bk1_t)
    qk0 = tl.dot(bq0, bk0_t)
    qk1 = tl.dot(bq1, bk1_t)

    g0 = tl.load(G_T_ptr + v0 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    b0 = tl.load(BETA_T_ptr + v0 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    gc0 = tl.cumsum(g0, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v0) * BT + offs_bt, gc0)
    eg0 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc0[:, None] - gc0[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v0) * (BT * BT) + offs_bt2,
        (qk0 * eg0).to(ATTN_ptr.dtype.element_ty),
    )
    l0 = tl.where(strict_lower, kkt0 * b0[:, None] * eg0, 0.0)
    p0 = tl.where(mask_d8, l0, 0.0).to(tl.float16)
    loff0 = tl.where(mask_off8, l0, 0.0).to(tl.float16)

    g1 = tl.load(G_T_ptr + v1 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    b1 = tl.load(BETA_T_ptr + v1 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    gc1 = tl.cumsum(g1, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v1) * BT + offs_bt, gc1)
    eg1 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc1[:, None] - gc1[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v1) * (BT * BT) + offs_bt2,
        (qk0 * eg1).to(ATTN_ptr.dtype.element_ty),
    )
    l1 = tl.where(strict_lower, kkt0 * b1[:, None] * eg1, 0.0)
    p1 = tl.where(mask_d8, l1, 0.0).to(tl.float16)
    loff1 = tl.where(mask_off8, l1, 0.0).to(tl.float16)

    g2 = tl.load(G_T_ptr + v2 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    b2 = tl.load(BETA_T_ptr + v2 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    gc2 = tl.cumsum(g2, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v2) * BT + offs_bt, gc2)
    eg2 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc2[:, None] - gc2[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v2) * (BT * BT) + offs_bt2,
        (qk1 * eg2).to(ATTN_ptr.dtype.element_ty),
    )
    l2 = tl.where(strict_lower, kkt1 * b2[:, None] * eg2, 0.0)
    p2 = tl.where(mask_d8, l2, 0.0).to(tl.float16)
    loff2 = tl.where(mask_off8, l2, 0.0).to(tl.float16)

    g3 = tl.load(G_T_ptr + v3 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    b3 = tl.load(BETA_T_ptr + v3 * T_total + bos + offs_t, mask=mask_t, other=0.0).to(
        tl.float32
    )
    gc3 = tl.cumsum(g3, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + v3) * BT + offs_bt, gc3)
    eg3 = tl.where(
        lower_eq, tl.exp(tl.where(lower_eq, gc3[:, None] - gc3[None, :], -1e9)), 0.0
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + v3) * (BT * BT) + offs_bt2,
        (qk1 * eg3).to(ATTN_ptr.dtype.element_ty),
    )
    l3 = tl.where(strict_lower, kkt1 * b3[:, None] * eg3, 0.0)
    p3 = tl.where(mask_d8, l3, 0.0).to(tl.float16)
    loff3 = tl.where(mask_off8, l3, 0.0).to(tl.float16)

    # 4-way interleaved exact 2-level 8x8 Euler inversion in FP16.
    a0 = eye16 - p0
    a1 = eye16 - p1
    a2 = eye16 - p2
    a3 = eye16 - p3
    for _ in range(2):
        p0 = tl.dot(p0, p0, out_dtype=tl.float16)
        p1 = tl.dot(p1, p1, out_dtype=tl.float16)
        p2 = tl.dot(p2, p2, out_dtype=tl.float16)
        p3 = tl.dot(p3, p3, out_dtype=tl.float16)
        a0 = tl.dot(eye16 + p0, a0, out_dtype=tl.float16)
        a1 = tl.dot(eye16 + p1, a1, out_dtype=tl.float16)
        a2 = tl.dot(eye16 + p2, a2, out_dtype=tl.float16)
        a3 = tl.dot(eye16 + p3, a3, out_dtype=tl.float16)

    p0 = tl.where(mask_off8, tl.dot(a0, loff0, out_dtype=tl.float16), zero16)
    p1 = tl.where(mask_off8, tl.dot(a1, loff1, out_dtype=tl.float16), zero16)
    p2 = tl.where(mask_off8, tl.dot(a2, loff2, out_dtype=tl.float16), zero16)
    p3 = tl.where(mask_off8, tl.dot(a3, loff3, out_dtype=tl.float16), zero16)
    a0 = tl.dot(eye16 - p0, a0, out_dtype=tl.float16)
    a1 = tl.dot(eye16 - p1, a1, out_dtype=tl.float16)
    a2 = tl.dot(eye16 - p2, a2, out_dtype=tl.float16)
    a3 = tl.dot(eye16 - p3, a3, out_dtype=tl.float16)
    for _ in range(2):
        p0 = tl.dot(p0, p0, out_dtype=tl.float16)
        p1 = tl.dot(p1, p1, out_dtype=tl.float16)
        p2 = tl.dot(p2, p2, out_dtype=tl.float16)
        p3 = tl.dot(p3, p3, out_dtype=tl.float16)
        a0 = tl.dot(eye16 + p0, a0, out_dtype=tl.float16)
        a1 = tl.dot(eye16 + p1, a1, out_dtype=tl.float16)
        a2 = tl.dot(eye16 + p2, a2, out_dtype=tl.float16)
        a3 = tl.dot(eye16 + p3, a3, out_dtype=tl.float16)

    a0_bf = tl.where(lower_eq, a0, zero16).to(tl.bfloat16)
    a1_bf = tl.where(lower_eq, a1, zero16).to(tl.bfloat16)
    a2_bf = tl.where(lower_eq, a2, zero16).to(tl.bfloat16)
    a3_bf = tl.where(lower_eq, a3, zero16).to(tl.bfloat16)

    bk0_f32 = bk0.to(tl.float32)
    bk1_f32 = bk1.to(tl.float32)
    w0 = tl.dot(a0_bf, (bk0_f32 * (b0 * tl.exp(gc0))[:, None]).to(tl.bfloat16))
    w1 = tl.dot(a1_bf, (bk0_f32 * (b1 * tl.exp(gc1))[:, None]).to(tl.bfloat16))
    w2 = tl.dot(a2_bf, (bk1_f32 * (b2 * tl.exp(gc2))[:, None]).to(tl.bfloat16))
    w3 = tl.dot(a3_bf, (bk1_f32 * (b3 * tl.exp(gc3))[:, None]).to(tl.bfloat16))
    tl.store(
        W_ptr + (pid_t * NV + v0) * (BT * D) + offs_btd, w0.to(W_ptr.dtype.element_ty)
    )
    tl.store(
        W_ptr + (pid_t * NV + v1) * (BT * D) + offs_btd, w1.to(W_ptr.dtype.element_ty)
    )
    tl.store(
        W_ptr + (pid_t * NV + v2) * (BT * D) + offs_btd, w2.to(W_ptr.dtype.element_ty)
    )
    tl.store(
        W_ptr + (pid_t * NV + v3) * (BT * D) + offs_btd, w3.to(W_ptr.dtype.element_ty)
    )

    v_base = MIXED_QKV_ptr + bos * QKV_DIM + Q_DIM + K_DIM
    t_qkv = offs_t[:, None] * QKV_DIM + offs_d[None, :]
    bv0 = tl.load(v_base + v0 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    bv1 = tl.load(v_base + v1 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    bv2 = tl.load(v_base + v2 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    bv3 = tl.load(v_base + v3 * D + t_qkv, mask=mask_t[:, None], other=0.0).to(
        tl.float32
    )
    u0 = tl.dot(a0_bf, (bv0 * b0[:, None]).to(tl.bfloat16))
    u1 = tl.dot(a1_bf, (bv1 * b1[:, None]).to(tl.bfloat16))
    u2 = tl.dot(a2_bf, (bv2 * b2[:, None]).to(tl.bfloat16))
    u3 = tl.dot(a3_bf, (bv3 * b3[:, None]).to(tl.bfloat16))
    tl.store(
        U_ptr + (pid_t * NV + v0) * (BT * D) + offs_btd, u0.to(U_ptr.dtype.element_ty)
    )
    tl.store(
        U_ptr + (pid_t * NV + v1) * (BT * D) + offs_btd, u1.to(U_ptr.dtype.element_ty)
    )
    tl.store(
        U_ptr + (pid_t * NV + v2) * (BT * D) + offs_btd, u2.to(U_ptr.dtype.element_ty)
    )
    tl.store(
        U_ptr + (pid_t * NV + v3) * (BT * D) + offs_btd, u3.to(U_ptr.dtype.element_ty)
    )


@triton.jit(do_not_specialize=["T_total"])
def _fused_gdn_prepare_950pr_kernel(
    Q_ptr,  # [T_total, NK, D] bf16
    K_ptr,  # [T_total, NK, D] bf16
    MIXED_QKV_ptr,  # [T_total, QKV_DIM] bf16 — zero-copy V read at + Q_DIM + K_DIM
    G_T_ptr,  # [NV, T_total] fp32
    BETA_T_ptr,  # [NV, T_total] bf16
    CU_ptr,  # [B+1] int32
    CHUNK_IDX_ptr,  # [NT, 2] int32
    W_ptr,  # [NT, NV, BT, D] bf16
    U_ptr,  # [NT, NV, BT, D] bf16
    ATTN_ptr,  # [NT, NV, BT, BT] bf16
    GCUM_ptr,  # [NT, NV, BT] fp32
    T_total,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    """Generic single-head Prepare fallback for arbitrary (NK, NV) with NV % NK == 0."""
    pid_t = tl.program_id(0)
    i_vh = tl.program_id(1)
    i_kh = i_vh // GQA

    i_n = tl.load(CHUNK_IDX_ptr + pid_t * 2).to(tl.int32)
    i_t = tl.load(CHUNK_IDX_ptr + pid_t * 2 + 1).to(tl.int32)
    bos = tl.load(CU_ptr + i_n).to(tl.int32)
    eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
    seq_len = eos - bos

    offs_bt = tl.arange(0, BT)
    offs_t = i_t * BT + offs_bt
    mask_t = offs_t < seq_len
    offs_d = tl.arange(0, D)
    offs_btd = offs_bt[:, None] * D + offs_d[None, :]
    offs_bt2 = offs_bt[:, None] * BT + offs_bt[None, :]

    t_nk = offs_t[:, None] * (NK * D) + offs_d[None, :]
    b_k = tl.load(K_ptr + (bos * NK + i_kh) * D + t_nk, mask=mask_t[:, None], other=0.0)
    b_q = tl.load(Q_ptr + (bos * NK + i_kh) * D + t_nk, mask=mask_t[:, None], other=0.0)
    b_k_t = tl.trans(b_k)
    b_kkt = tl.dot(b_k, b_k_t)
    b_qk = tl.dot(b_q, b_k_t)

    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    strict_lower = rows > cols
    lower_eq = rows >= cols
    same_blk8 = (rows // 8) == (cols // 8)
    mask_d8 = strict_lower & same_blk8
    mask_off8 = strict_lower & (~same_blk8)
    eye16 = tl.where(rows == cols, 1.0, 0.0).to(tl.float16)
    zero16 = tl.full((BT, BT), 0.0, dtype=tl.float16)

    g_off = i_vh * T_total + bos + offs_t
    b_g = tl.load(G_T_ptr + g_off, mask=mask_t, other=0.0).to(tl.float32)
    b_beta = tl.load(BETA_T_ptr + g_off, mask=mask_t, other=0.0).to(tl.float32)
    b_g_cum = tl.cumsum(b_g, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + i_vh) * BT + offs_bt, b_g_cum)

    eg = tl.where(
        lower_eq,
        tl.exp(tl.where(lower_eq, b_g_cum[:, None] - b_g_cum[None, :], -1e9)),
        0.0,
    )
    tl.store(
        ATTN_ptr + (pid_t * NV + i_vh) * (BT * BT) + offs_bt2,
        (b_qk * eg).to(ATTN_ptr.dtype.element_ty),
    )

    b_L = tl.where(strict_lower, b_kkt * b_beta[:, None] * eg, 0.0)
    p_mat = tl.where(mask_d8, b_L, 0.0).to(tl.float16)
    a_inv = eye16 - p_mat
    for _ in range(2):
        p_mat = tl.dot(p_mat, p_mat, out_dtype=tl.float16)
        a_inv = tl.dot(eye16 + p_mat, a_inv, out_dtype=tl.float16)

    l_off = tl.where(mask_off8, b_L, 0.0).to(tl.float16)
    p_mat = tl.where(mask_off8, tl.dot(a_inv, l_off, out_dtype=tl.float16), zero16)
    a_inv = tl.dot(eye16 - p_mat, a_inv, out_dtype=tl.float16)
    for _ in range(2):
        p_mat = tl.dot(p_mat, p_mat, out_dtype=tl.float16)
        a_inv = tl.dot(eye16 + p_mat, a_inv, out_dtype=tl.float16)

    a_inv_bf = tl.where(lower_eq, a_inv, zero16).to(tl.bfloat16)

    k_scaled = (b_k.to(tl.float32) * (b_beta * tl.exp(b_g_cum))[:, None]).to(
        tl.bfloat16
    )
    b_w = tl.dot(a_inv_bf, k_scaled)
    tl.store(
        W_ptr + (pid_t * NV + i_vh) * (BT * D) + offs_btd,
        b_w.to(W_ptr.dtype.element_ty),
    )

    v_base = MIXED_QKV_ptr + bos * QKV_DIM + Q_DIM + K_DIM + i_vh * D
    b_v = tl.load(
        v_base + offs_t[:, None] * QKV_DIM + offs_d[None, :],
        mask=mask_t[:, None],
        other=0.0,
    ).to(tl.float32)
    v_scaled = (b_v * b_beta[:, None]).to(tl.bfloat16)
    b_u = tl.dot(a_inv_bf, v_scaled)
    tl.store(
        U_ptr + (pid_t * NV + i_vh) * (BT * D) + offs_btd,
        b_u.to(U_ptr.dtype.element_ty),
    )


@triton.jit(do_not_specialize=["scale"])
def _fused_gdn_fwd_ho_950pr_kernel(
    Q_ptr,  # [T_total, NK, D] bf16
    K_ptr,  # [T_total, NK, D] bf16
    W_ptr,  # [NT, NV, BT, D] bf16 — contiguous!
    U_ptr,  # [NT, NV, BT, D] bf16 — contiguous!
    ATTN_ptr,  # [NT, NV, BT, BT] bf16 — unscaled intra-chunk attention from Stage 1
    GCUM_ptr,  # [NT, NV, BT] fp32    — contiguous!
    H0_ptr,  # [NUM_SLOTS, NV, DV, DK] bf16/fp32 (ssm_states pool!)
    HT_ptr,  # [NUM_SLOTS, NV, DV, DK] bf16/fp32 (ssm_states pool!)
    H_CKPT_ptr,  # [NUM_SLOTS, NV, DV, DK] bf16/fp32 (ssm_states pool!)
    CACHE_IDX_ptr,  # [B] int32
    O_ptr,  # [T_total, NV, D] bf16
    CKPT_DST_ptr,  # [B] int32
    CU_ptr,  # [B+1] int32
    CHUNK_OFF_ptr,  # [B+1] int32
    CKPT_IDX_ptr,  # [B] int32
    scale,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    SAVE_CKPT: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = pid // NV
    i_vh = pid % NV
    i_kh = i_vh // GQA

    bos = tl.load(CU_ptr + i_n).to(tl.int32)
    eos = tl.load(CU_ptr + i_n + 1).to(tl.int32)
    seq_len = eos - bos
    boh = tl.load(CHUNK_OFF_ptr + i_n).to(tl.int32)
    NT = (seq_len + BT - 1) // BT

    slot_idx = tl.load(CACHE_IDX_ptr + i_n).to(tl.int64)
    ckpt_chunk = -1
    ckpt_slot = tl.zeros((), dtype=tl.int64)
    if SAVE_CKPT:
        ckpt_chunk = tl.load(CKPT_IDX_ptr + i_n).to(tl.int32)
        ckpt_slot = tl.load(CKPT_DST_ptr + i_n).to(tl.int64)
        if ckpt_slot < 0:
            ckpt_chunk = -1

    offs_d = tl.arange(0, D)
    offs_dd = (offs_d[:, None] * D + offs_d[None, :]).to(tl.int64)
    state_off = (slot_idx * NV + i_vh) * (D * D) + offs_dd
    b_h = tl.load(H0_ptr + state_off).to(tl.float32)

    offs_bt = tl.arange(0, BT)
    offs_btd = offs_bt[:, None] * D + offs_d[None, :]
    offs_bt2 = offs_bt[:, None] * BT + offs_bt[None, :]

    q_base = Q_ptr + (bos * NK + i_kh) * D
    k_base = K_ptr + (bos * NK + i_kh) * D
    o_base = O_ptr + (bos * NV + i_vh) * D

    for i_t in range(NT):
        if SAVE_CKPT:
            if i_t == ckpt_chunk:
                ckpt_off = (ckpt_slot * NV + i_vh) * (D * D) + offs_dd
                tl.store(H_CKPT_ptr + ckpt_off, b_h.to(H_CKPT_ptr.dtype.element_ty))

        c_id = boh + i_t
        offs_t = i_t * BT + offs_bt
        mask_t = offs_t < seq_len
        t_nd_k = offs_t[:, None] * (NK * D) + offs_d[None, :]
        t_nd_v = offs_t[:, None] * (NV * D) + offs_d[None, :]

        b_w = tl.load(W_ptr + (c_id * NV + i_vh) * (BT * D) + offs_btd)
        b_u = tl.load(U_ptr + (c_id * NV + i_vh) * (BT * D) + offs_btd)
        b_attn = tl.load(ATTN_ptr + (c_id * NV + i_vh) * (BT * BT) + offs_bt2)
        b_g = tl.load(GCUM_ptr + (c_id * NV + i_vh) * BT + offs_bt).to(tl.float32)
        b_k = tl.load(k_base + t_nd_k, mask=mask_t[:, None], other=0.0)
        b_q = tl.load(q_base + t_nd_k, mask=mask_t[:, None], other=0.0)

        # Issue both H^T dots back-to-back for 2-way Cube overlap sharing b_h_bf_t.
        b_h_bf_t = tl.trans(b_h.to(tl.bfloat16))
        b_wh = tl.dot(b_w, b_h_bf_t)
        b_qh = tl.dot(b_q, b_h_bf_t)

        b_v_new = tl.where(mask_t[:, None], b_u.to(tl.float32) - b_wh, 0.0)
        b_o = (
            b_qh * tl.exp(b_g)[:, None] + tl.dot(b_attn, b_v_new.to(tl.bfloat16))
        ) * scale
        tl.store(o_base + t_nd_v, b_o.to(O_ptr.dtype.element_ty), mask=mask_t[:, None])

        last_in_chunk = tl.minimum(seq_len - i_t * BT, BT) - 1
        b_g_last = tl.load(GCUM_ptr + (c_id * NV + i_vh) * BT + last_in_chunk).to(
            tl.float32
        )
        b_decay = tl.where(mask_t, tl.exp(b_g_last - b_g), 0.0)
        b_v_gated = (b_v_new * b_decay[:, None]).to(tl.bfloat16)
        b_h = b_h * tl.exp(b_g_last) + tl.dot(tl.trans(b_v_gated), b_k)

    tl.store(HT_ptr + state_off, b_h.to(HT_ptr.dtype.element_ty))


def build_gdn_950pr_chunk_meta(
    seq_lens_cpu,
    device: torch.device,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build (chunk_indices [NT, 2], chunk_offsets [B+1], NT, total_valid_tokens) on CPU once per step."""
    pairs = []
    offsets = [0]
    total_valid = 0
    for i_n, raw_len in enumerate(seq_lens_cpu):
        L = int(raw_len)
        total_valid += L
        nt = (L + chunk_size - 1) // chunk_size
        for i_t in range(nt):
            pairs.append((i_n, i_t))
        offsets.append(offsets[-1] + nt)
    nt_total = len(pairs)
    chunk_indices = torch.tensor(pairs, dtype=torch.int32).to(device, non_blocking=True)
    chunk_offsets = torch.tensor(offsets, dtype=torch.int32).to(
        device, non_blocking=True
    )
    return chunk_indices, chunk_offsets, nt_total, total_valid


def can_use_fused_gdn_prefill_950pr(
    layer,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    chunk_meta,
) -> bool:
    if (
        not envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL.get()
        or not is_npu_arch35()
        or chunk_meta is None
    ):
        return False
    nk = layer.num_k_heads
    nv = layer.num_v_heads
    dk = layer.head_k_dim
    return (
        dk == 128
        and layer.head_q_dim == 128
        and layer.head_v_dim == 128
        and layer.num_q_heads == nk
        and nv > 0
        and nk > 0
        and nv % nk == 0
        and (nk & (nk - 1)) == 0
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
        and ssm_states.is_contiguous()
        and layer.q_dim == nk * 128
        and layer.k_dim == nk * 128
        and layer.v_dim == nv * 128
        and mixed_qkv.shape[-1] == layer.q_dim + layer.k_dim + layer.v_dim
    )


def fused_gdn_prefill_950pr(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_meta: Tuple[torch.Tensor, torch.Tensor, int, int],
    *,
    ckpt_chunk_idx: Optional[torch.Tensor] = None,
    ckpt_dst_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run full Ascend 950PR fused GDN prefill (L2Norm + Gating + Prepare + FwdH+O + In-Place Pool/Ckpt)."""
    chunk_indices, chunk_offsets, nt_total, total_valid = chunk_meta
    T = mixed_qkv.shape[0]
    NK = layer.num_k_heads
    NV = layer.num_v_heads
    D = 128
    GQA = NV // NK
    BT = 64
    B = cu_seqlens.shape[0] - 1
    Q_DIM = layer.q_dim
    K_DIM = layer.k_dim
    QKV_DIM = mixed_qkv.shape[-1]
    dev = mixed_qkv.device
    scale = D**-0.5

    if nt_total == 0:
        return torch.zeros((T, NV, D), dtype=mixed_qkv.dtype, device=dev)

    # Stage 0: Persistent 56-AIV Q/K L2Norm (4-row) + 2D UB block-transposed Gating (zero V copy)
    q_norm = torch.empty((T, NK, D), dtype=torch.bfloat16, device=dev)
    k_norm = torch.empty((T, NK, D), dtype=torch.bfloat16, device=dev)
    g_T = torch.empty((NV, T), dtype=torch.float32, device=dev)
    beta_T = torch.empty((NV, T), dtype=torch.bfloat16, device=dev)
    grid_row = min(56, max(1, T))
    rpp = (T + grid_row - 1) // grid_row
    _fused_qk_l2norm_gating_row_kernel[(grid_row,)](
        mixed_qkv,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        q_norm,
        k_norm,
        g_T,
        beta_T,
        T,
        rpp,
        Q_DIM=Q_DIM,
        K_DIM=K_DIM,
        QKV_DIM=QKV_DIM,
        NK=NK,
        NV=NV,
        NV_PAD=triton.next_power_of_2(NV),
        D=D,
        EPS=1e-6,
        R_GB=32,
        R_QK=4,
    )

    # Stage 1: GQA-interleaved Prepare (K@K^T + Q@K^T + cumsum + ATTN + FP16 Euler solve_tril + W,U)
    w = torch.empty((nt_total, NV, BT, D), dtype=torch.bfloat16, device=dev)
    u = torch.empty((nt_total, NV, BT, D), dtype=torch.bfloat16, device=dev)
    attn = torch.empty((nt_total, NV, BT, BT), dtype=torch.bfloat16, device=dev)
    g_cum = torch.empty((nt_total, NV, BT), dtype=torch.float32, device=dev)
    if GQA == 3:
        _fused_gdn_prepare_gqa3_950pr_kernel[(nt_total, NK)](
            q_norm,
            k_norm,
            mixed_qkv,
            g_T,
            beta_T,
            cu_seqlens,
            chunk_indices,
            w,
            u,
            attn,
            g_cum,
            T,
            Q_DIM=Q_DIM,
            K_DIM=K_DIM,
            QKV_DIM=QKV_DIM,
            NK=NK,
            NV=NV,
            GQA=GQA,
            D=D,
            BT=BT,
            hfusion_enable_multiple_consumer_fusion=True,
            multibuffer=False,
        )
    elif GQA == 2 and NK % 2 == 0:
        _fused_gdn_prepare_gqa2x2_950pr_kernel[(nt_total, NK // 2)](
            q_norm,
            k_norm,
            mixed_qkv,
            g_T,
            beta_T,
            cu_seqlens,
            chunk_indices,
            w,
            u,
            attn,
            g_cum,
            T,
            Q_DIM=Q_DIM,
            K_DIM=K_DIM,
            QKV_DIM=QKV_DIM,
            NK=NK,
            NV=NV,
            GQA=GQA,
            D=D,
            BT=BT,
            hfusion_enable_multiple_consumer_fusion=True,
            multibuffer=False,
        )
    else:
        _fused_gdn_prepare_950pr_kernel[(nt_total, NV)](
            q_norm,
            k_norm,
            mixed_qkv,
            g_T,
            beta_T,
            cu_seqlens,
            chunk_indices,
            w,
            u,
            attn,
            g_cum,
            T,
            Q_DIM=Q_DIM,
            K_DIM=K_DIM,
            QKV_DIM=QKV_DIM,
            NK=NK,
            NV=NV,
            GQA=GQA,
            D=D,
            BT=BT,
            hfusion_enable_multiple_consumer_fusion=True,
        )

    # Stage 2: Slim 4-dot Fused ChunkFwdH + ChunkFwdO + In-Place Pool + In-Kernel Checkpoint
    o = torch.empty((T, NV, D), dtype=torch.bfloat16, device=dev)
    save_ckpt = ckpt_chunk_idx is not None and ckpt_dst_slots is not None
    ckpt_idx_arg = ckpt_chunk_idx if save_ckpt else cache_indices
    ckpt_dst_arg = ckpt_dst_slots if save_ckpt else cache_indices

    _fused_gdn_fwd_ho_950pr_kernel[(B * NV,)](
        q_norm,
        k_norm,
        w,
        u,
        attn,
        g_cum,
        ssm_states,
        ssm_states,
        ssm_states,
        cache_indices,
        o,
        ckpt_dst_arg,
        cu_seqlens,
        chunk_offsets,
        ckpt_idx_arg,
        scale,
        NK=NK,
        NV=NV,
        GQA=GQA,
        D=D,
        BT=BT,
        SAVE_CKPT=save_ckpt,
        multibuffer=False,
    )
    if T > total_valid:
        o[total_valid:].zero_()
    return o
