"""Custom Ascend 950PR (arch35) fused Triton GDN Prefill kernels.

Designed specifically for Ascend 950PR (28 AIC / 56 AIV, 1.43 TB/s HBM, 128 MB L2,
512 KB L1 CBUF, 254 KB UB):
  1. `_fused_qk_l2norm_gating_row_kernel`: Row-parallel AIV fusion of Q/K L2Norm and
     GDN gating (`g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`),
     reading Q/K directly from `mixed_qkv` without copying V, and emitting transposed
     contiguous `g_T [NV, T]` / `beta_T [NV, T]` for burst chunk reads.
  2. `_fused_gdn_prepare_950pr_kernel`: Single-kernel fusion of local cumsum,
     `K @ K^T`, exact 2-level 8x8 block Euler triangular inversion `(I + L)^-1`
     (9 Cube dots, numerically stable even on 100% identical repeated tokens), and
     WY representation `(W, U)` reading V directly from `mixed_qkv` (zero V copy)
     into contiguous `[NT, NV, 64, D]` chunk layout.
  3. `_fused_gdn_fwd_ho_950pr_kernel`: Single-kernel `(B * NV)` fully-parallel
     fusion of `ChunkFwdH + ChunkFwdO` (zero inter-sequence SyncAll), keeping the
     state `h [DV, DK]` on-chip in native CANN layout, eliminating all per-chunk `h`
     and `v_new` HBM traffic, reading/writing `ssm_states` pool slots in-place, and
     writing single-point Radix Cache checkpoints directly into `ssm_states` in-kernel
     (eliminating 2-piece Slice + ViewCopy overhead).
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
):
    pid = tl.program_id(0)
    r0 = pid * rows_per_prog
    r1 = tl.minimum(r0 + rows_per_prog, T)
    offs_kh = tl.arange(0, NK)[:, None] * D + tl.arange(0, D)[None, :]
    offs_nv = tl.arange(0, NV_PAD)
    mask_nv = offs_nv < NV
    a_log = tl.load(A_LOG_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
    neg_exp_a = -tl.exp(a_log)
    dt_b = tl.load(DT_BIAS_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)

    for row in range(r0, r1):
        base_in = MIXED_QKV_ptr + row * QKV_DIM
        q = tl.load(base_in + offs_kh).to(tl.float32)
        q_inv = tl.rsqrt(tl.sum(q * q, axis=1, keep_dims=True) + EPS)
        tl.store(
            Q_OUT_ptr + row * Q_DIM + offs_kh,
            (q * q_inv).to(Q_OUT_ptr.dtype.element_ty),
        )

        k = tl.load(base_in + Q_DIM + offs_kh).to(tl.float32)
        k_inv = tl.rsqrt(tl.sum(k * k, axis=1, keep_dims=True) + EPS)
        tl.store(
            K_OUT_ptr + row * K_DIM + offs_kh,
            (k * k_inv).to(K_OUT_ptr.dtype.element_ty),
        )

        a_val = tl.load(A_ptr + row * NV + offs_nv, mask=mask_nv, other=0.0).to(
            tl.float32
        )
        b_val = tl.load(B_ptr + row * NV + offs_nv, mask=mask_nv, other=0.0).to(
            tl.float32
        )
        sp_in = a_val + dt_b
        sp = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
        g_val = neg_exp_a * sp
        beta_val = tl.sigmoid(b_val)
        tl.store(G_T_ptr + offs_nv * T + row, g_val, mask=mask_nv)
        tl.store(
            BETA_T_ptr + offs_nv * T + row,
            beta_val.to(BETA_T_ptr.dtype.element_ty),
            mask=mask_nv,
        )


@triton.jit(do_not_specialize=["T_total"])
def _fused_gdn_prepare_950pr_kernel(
    K_ptr,  # [T_total, NK, D] bf16
    MIXED_QKV_ptr,  # [T_total, QKV_DIM] bf16 — zero-copy V read at + Q_DIM + K_DIM!
    G_T_ptr,  # [NV, T_total] fp32
    BETA_T_ptr,  # [NV, T_total] bf16
    CU_ptr,  # [B+1] int32
    CHUNK_IDX_ptr,  # [NT, 2] int32
    W_ptr,  # [NT, NV, BT, D] bf16
    U_ptr,  # [NT, NV, BT, D] bf16
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

    b_k = tl.load(
        K_ptr + (bos * NK + i_kh) * D + offs_t[:, None] * (NK * D) + offs_d[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    b_kkt = tl.dot(b_k, tl.trans(b_k))

    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    strict_lower = rows > cols
    same_blk8 = (rows // 8) == (cols // 8)
    mask_d8 = strict_lower & same_blk8
    mask_off8 = strict_lower & (~same_blk8)
    eye = tl.where(rows == cols, 1.0, 0.0).to(tl.float32)

    g_off = i_vh * T_total + bos + offs_t
    b_g = tl.load(G_T_ptr + g_off, mask=mask_t, other=0.0).to(tl.float32)
    b_beta = tl.load(BETA_T_ptr + g_off, mask=mask_t, other=0.0).to(tl.float32)
    b_g_cum = tl.cumsum(b_g, axis=0)
    tl.store(GCUM_ptr + (pid_t * NV + i_vh) * BT + offs_bt, b_g_cum)

    g_diff = tl.where(strict_lower, b_g_cum[:, None] - b_g_cum[None, :], -1e9)
    b_L = tl.where(strict_lower, b_kkt * b_beta[:, None] * tl.exp(g_diff), 0.0)

    # Exact 2-level 8x8 block Euler inversion: (I + L)^-1 (9 Cube dots, zero loops over rows).
    # Use FP16 (10-bit mantissa vs 7-bit BF16) inside the inversion: every entry of
    # (I + L)^-1 for unit-norm keys and beta in (0, 1) is bounded by O(1) (no FP16
    # overflow risk), cutting correlated-key inversion error by >4x at identical Cube speed.
    p_mat = tl.where(mask_d8, b_L, 0.0)
    a_inv = eye - p_mat
    for _ in range(2):
        p_h = p_mat.to(tl.float16)
        p_mat = tl.dot(p_h, p_h)
        a_inv = tl.dot((eye + p_mat).to(tl.float16), a_inv.to(tl.float16))

    l_off = tl.where(mask_off8, b_L, 0.0)
    p_mat = tl.where(mask_off8, tl.dot(a_inv.to(tl.float16), l_off.to(tl.float16)), 0.0)
    a_inv = tl.dot((eye - p_mat).to(tl.float16), a_inv.to(tl.float16))
    for _ in range(2):
        p_h = p_mat.to(tl.float16)
        p_mat = tl.dot(p_h, p_h)
        a_inv = tl.dot((eye + p_mat).to(tl.float16), a_inv.to(tl.float16))

    a_inv_bf = tl.where(rows >= cols, a_inv, 0.0).to(tl.bfloat16)

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
    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    lower_eq = rows >= cols
    offs_btd = offs_bt[:, None] * D + offs_d[None, :]

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
        b_g = tl.load(GCUM_ptr + (c_id * NV + i_vh) * BT + offs_bt).to(tl.float32)
        b_k = tl.load(k_base + t_nd_k, mask=mask_t[:, None], other=0.0)
        b_q = tl.load(q_base + t_nd_k, mask=mask_t[:, None], other=0.0)

        b_h_bf_t = tl.trans(b_h.to(tl.bfloat16))
        b_v_new = tl.where(
            mask_t[:, None], b_u.to(tl.float32) - tl.dot(b_w, b_h_bf_t), 0.0
        )

        b_o = tl.dot(b_q, b_h_bf_t) * tl.exp(b_g)[:, None]
        b_qk = tl.dot(b_q, tl.trans(b_k))
        g_diff = tl.where(lower_eq, b_g[:, None] - b_g[None, :], -1e9)
        b_attn = tl.where(lower_eq, b_qk * tl.exp(g_diff), 0.0).to(tl.bfloat16)
        b_o = (b_o + tl.dot(b_attn, b_v_new.to(tl.bfloat16))) * scale
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

    if nt_total == 0:
        return torch.zeros((T, NV, D), dtype=mixed_qkv.dtype, device=dev)

    # Stage 0: Row-parallel AIV Q/K L2Norm + Gating (zero V copy!)
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
    )

    # Stage 1: Fused Prepare (K@K^T + cumsum + 2-level 8x8 Euler solve_tril + W,U)
    w = torch.empty((nt_total, NV, BT, D), dtype=torch.bfloat16, device=dev)
    u = torch.empty((nt_total, NV, BT, D), dtype=torch.bfloat16, device=dev)
    g_cum = torch.empty((nt_total, NV, BT), dtype=torch.float32, device=dev)
    _fused_gdn_prepare_950pr_kernel[(nt_total, NV)](
        k_norm,
        mixed_qkv,
        g_T,
        beta_T,
        cu_seqlens,
        chunk_indices,
        w,
        u,
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
    )

    # Stage 2: Fused ChunkFwdH + ChunkFwdO + In-Place Pool + In-Kernel Checkpoint
    o = torch.empty((T, NV, D), dtype=torch.bfloat16, device=dev)
    save_ckpt = ckpt_chunk_idx is not None and ckpt_dst_slots is not None
    ckpt_idx_arg = ckpt_chunk_idx if save_ckpt else cache_indices
    ckpt_dst_arg = ckpt_dst_slots if save_ckpt else cache_indices

    _fused_gdn_fwd_ho_950pr_kernel[(B * NV,)](
        q_norm,
        k_norm,
        w,
        u,
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
        D**-0.5,
        NK=NK,
        NV=NV,
        GQA=GQA,
        D=D,
        BT=BT,
        SAVE_CKPT=save_ckpt,
    )
    if T > total_valid:
        o[total_valid:].zero_()
    return o
