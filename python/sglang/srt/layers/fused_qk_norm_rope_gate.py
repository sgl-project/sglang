"""Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave kernel.

Single Triton kernel replacing 4 separate operations for models with
attn_output_gate (e.g. Qwen3.5):
  1. Q/Gate deinterleave from interleaved buffer
  2. GemmaRMSNorm on Q (per-head)
  3. GemmaRMSNorm on K (per-head)
  4. NeoX RoPE on Q and K (partial rotation)

Grid = q_rows (seq * num_q_heads).  Each block handles one (token, q_head).
The first k_rows blocks also handle the corresponding K heads.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_gemma_rmsnorm_rope_gate_kernel(
    QG_ptr,
    K_ptr,
    Q_out_ptr,
    K_out_ptr,
    Gate_out_ptr,
    QW_ptr,
    KW_ptr,
    COS_SIN_ptr,
    POS_ptr,
    qg_token_stride,
    qg_head_stride,
    k_token_stride,
    k_head_stride,
    num_heads,
    num_kv_heads,
    k_rows,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HEAD_DIM
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    token_idx = pid // num_heads
    head_idx = pid % num_heads

    # --- Q: RMSNorm + RoPE ---
    base = token_idx * qg_token_stride + head_idx * qg_head_stride
    q = tl.load(QG_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w_q = tl.load(QW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=0) / HEAD_DIM
    q_inv_rms = tl.rsqrt(q_var + EPS)
    q_normed = (q * q_inv_rms * (w_q + 1.0)).to(out_dtype).to(tl.float32)
    out_off = pid * HEAD_DIM

    # Store passthrough tail [rotary_dim, head_dim)
    if HAS_PASS:
        pass_mask = mask & (cols >= ROTARY_DIM)
        tl.store(Q_out_ptr + out_off + cols, q_normed, mask=pass_mask)

    # RoPE on [0, rotary_dim): reload from input, re-normalize (L1-cached)
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    qr1 = tl.load(QG_ptr + base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    qr2 = tl.load(QG_ptr + base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wq1 = tl.load(QW_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wq2 = tl.load(QW_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    qr1 = (qr1 * q_inv_rms * (wq1 + 1.0)).to(out_dtype).to(tl.float32)
    qr2 = (qr2 * q_inv_rms * (wq2 + 1.0)).to(out_dtype).to(tl.float32)

    pos = tl.load(POS_ptr + token_idx).to(tl.int64)
    cache_off = pos * ROTARY_DIM
    cos = tl.load(COS_SIN_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    sin = tl.load(
        COS_SIN_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)

    tl.store(Q_out_ptr + out_off + rot_offs, (qr1 * cos - qr2 * sin), mask=rot_mask)
    tl.store(
        Q_out_ptr + out_off + HALF_ROTARY + rot_offs,
        (qr2 * cos + qr1 * sin),
        mask=rot_mask,
    )

    # --- Gate copy ---
    gate = tl.load(QG_ptr + base + HEAD_DIM + cols, mask=mask, other=0.0)
    tl.store(Gate_out_ptr + out_off + cols, gate, mask=mask)

    # --- K: RMSNorm + RoPE (first k_rows blocks only) ---
    if pid < k_rows:
        token_idx_k = pid // num_kv_heads
        head_idx_k = pid % num_kv_heads
        k_off = token_idx_k * k_token_stride + head_idx_k * k_head_stride
        k = tl.load(K_ptr + k_off + cols, mask=mask, other=0.0).to(tl.float32)
        w_k = tl.load(KW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM
        k_inv_rms = tl.rsqrt(k_var + EPS)
        k_normed = (k * k_inv_rms * (w_k + 1.0)).to(out_dtype).to(tl.float32)
        k_out_off = pid * HEAD_DIM

        if HAS_PASS:
            tl.store(K_out_ptr + k_out_off + cols, k_normed, mask=pass_mask)

        kr1 = tl.load(K_ptr + k_off + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        kr2 = tl.load(
            K_ptr + k_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
        ).to(tl.float32)
        wk1 = tl.load(KW_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
        wk2 = tl.load(KW_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
            tl.float32
        )
        kr1 = (kr1 * k_inv_rms * (wk1 + 1.0)).to(out_dtype).to(tl.float32)
        kr2 = (kr2 * k_inv_rms * (wk2 + 1.0)).to(out_dtype).to(tl.float32)

        k_pos = tl.load(POS_ptr + token_idx_k).to(tl.int64)
        k_cache_off = k_pos * ROTARY_DIM
        k_cos = tl.load(
            COS_SIN_ptr + k_cache_off + rot_offs, mask=rot_mask, other=0.0
        ).to(tl.float32)
        k_sin = tl.load(
            COS_SIN_ptr + k_cache_off + HALF_ROTARY + rot_offs,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(
            K_out_ptr + k_out_off + rot_offs,
            (kr1 * k_cos - kr2 * k_sin),
            mask=rot_mask,
        )
        tl.store(
            K_out_ptr + k_out_off + HALF_ROTARY + rot_offs,
            (kr2 * k_cos + kr1 * k_sin),
            mask=rot_mask,
        )


def fused_qk_gemma_rmsnorm_rope_with_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    num_heads: int,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK GemmaRMSNorm + NeoX RoPE + gate extraction.

    Combines deinterleave, per-head RMSNorm, partial-rotary NeoX RoPE,
    and gate copy into a single Triton kernel launch.

    q_gate: (seq, q_size*2) interleaved [q_h0, gate_h0, q_h1, gate_h1, ...]
    k: (seq, kv_size)
    cos_sin_cache: (max_pos, rotary_dim) with [cos | sin] layout
    positions: (seq,) token positions

    Returns (q_out, k_out, gate_out) all contiguous with shape
    (seq*num_heads, head_dim), (seq*num_kv_heads, head_dim), (seq*num_heads, head_dim).
    """
    seq_len = q_gate.shape[0]
    qg_3d = q_gate.view(seq_len, num_heads, 2 * head_dim)
    num_kv_heads = k.shape[-1] // head_dim
    k_3d = k.view(seq_len, num_kv_heads, head_dim)

    q_rows = seq_len * num_heads
    k_rows = seq_len * num_kv_heads

    q_out = torch.empty(q_rows, head_dim, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(k_rows, head_dim, dtype=k.dtype, device=k.device)
    gate_out = torch.empty(q_rows, head_dim, dtype=q_gate.dtype, device=q_gate.device)

    half_rotary = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)
    ROT_HALF_BLOCK = triton.next_power_of_2(half_rotary)

    _fused_qk_gemma_rmsnorm_rope_gate_kernel[(q_rows,)](
        qg_3d,
        k_3d,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qg_3d.stride(0),
        qg_3d.stride(1),
        k_3d.stride(0),
        k_3d.stride(1),
        num_heads,
        num_kv_heads,
        k_rows,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        BLOCK_HD=BLOCK_HD,
        ROT_HALF_BLOCK=ROT_HALF_BLOCK,
        EPS=eps,
        FP16=(q_gate.dtype == torch.float16),
        HAS_PASS=rotary_dim < head_dim,
    )

    return q_out, k_out, gate_out
