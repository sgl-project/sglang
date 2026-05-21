"""Fused post-gather compressor kernel for DeepSeek V4 on HIP.

Fuses APE-add + overlap-transform + softmax-pool + RMSNorm + RoPE into a
single Triton kernel, operating on the already-gathered KVAndScore tensor.

Grid = (bs,): one program per sequence in the batch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ape_pool_norm_rope_kernel(
    kv_score_ptr,
    kv_score_stride_b,
    kv_score_stride_k,
    ape_ptr,
    ape_stride_r,
    rms_weight_ptr,
    rms_eps,
    freqs_ptr,
    freqs_stride_b,
    out_ptr,
    out_stride_b,
    head_dim,
    rope_head_dim,
    half_dim,
    RATIO: tl.constexpr,
    K_POOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    bid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim

    m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
    kv_acc = tl.zeros([BLOCK_D], tl.float32)
    w_acc = tl.zeros([BLOCK_D], tl.float32)

    batch_base = bid * kv_score_stride_b

    # --- APE-add + overlap-transform + online softmax pool ---
    for k in tl.range(0, K_POOL):
        if OVERLAP:
            is_b = k >= RATIO
            col_off = tl.where(is_b, head_dim, 0)
        else:
            col_off = 0

        row_off = batch_base + k * kv_score_stride_k

        kv_val = tl.load(
            kv_score_ptr + row_off + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        sc_val = tl.load(
            kv_score_ptr + row_off + half_dim + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)

        ape_val = tl.load(
            ape_ptr + (k % RATIO) * ape_stride_r + col_off + d, mask=d_mask, other=0.0
        ).to(tl.float32)
        score_k = sc_val + ape_val

        m_new = tl.maximum(m_prev, score_k)
        exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
        exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
        kv_acc = kv_acc * exp_old + exp_cur * kv_val
        w_acc = w_acc * exp_old + exp_cur
        m_prev = m_new

    compressed = kv_acc / w_acc

    # --- RMSNorm ---
    rms_w = tl.load(rms_weight_ptr + d, mask=d_mask, other=0.0)
    c_sq = tl.where(d_mask, compressed * compressed, 0.0)
    var = tl.sum(c_sq, axis=0) / head_dim
    normed = compressed * tl.rsqrt(var + rms_eps) * rms_w

    # --- Store full normed result ---
    out_base = out_ptr + bid * out_stride_b
    tl.store(out_base + d, normed.to(out_ptr.dtype.element_ty), mask=d_mask)

    # --- RoPE in-place on the last rope_head_dim elements ---
    # Load back pairs, apply complex multiplication, store back.
    rope_start = head_dim - rope_head_dim
    p = tl.arange(0, HALF_ROPE)
    pmask = p < (rope_head_dim // 2)

    xr = tl.load(out_base + rope_start + 2 * p, mask=pmask, other=0.0).to(tl.float32)
    xi = tl.load(out_base + rope_start + 2 * p + 1, mask=pmask, other=0.0).to(
        tl.float32
    )

    freq_base = bid * freqs_stride_b
    fr = tl.load(freqs_ptr + freq_base + 2 * p, mask=pmask, other=1.0).to(tl.float32)
    fi = tl.load(freqs_ptr + freq_base + 2 * p + 1, mask=pmask, other=0.0).to(
        tl.float32
    )

    tl.store(
        out_base + rope_start + 2 * p,
        (xr * fr - xi * fi).to(out_ptr.dtype.element_ty),
        mask=pmask,
    )
    tl.store(
        out_base + rope_start + 2 * p + 1,
        (xr * fi + xi * fr).to(out_ptr.dtype.element_ty),
        mask=pmask,
    )


def fused_ape_pool_norm_rope(
    kv_score_gathered: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    freqs_cis_real: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
) -> torch.Tensor:
    """Fused APE-add + overlap-transform + softmax-pool + RMSNorm + RoPE.

    Args:
        kv_score_gathered: [bs, K_IN, last_dim] fp32.
        ape: [ratio, coff * head_dim] fp32.
        rms_weight: [head_dim] fp32.
        rms_eps: float.
        freqs_cis_real: [bs, rope_dim] fp32.

    Returns:
        [bs, head_dim] fp32.
    """
    coff = 2 if overlap else 1
    bs = kv_score_gathered.shape[0]
    K_IN = kv_score_gathered.shape[1]
    last_dim = kv_score_gathered.shape[2]
    half_dim = last_dim // 2

    assert K_IN == ratio * coff, f"K_IN={K_IN} != ratio*coff={ratio}*{coff}"

    out = torch.empty(
        bs, head_dim, dtype=torch.float32, device=kv_score_gathered.device
    )
    if bs == 0:
        return out

    BLOCK_D = triton.next_power_of_2(head_dim)
    HALF_ROPE = triton.next_power_of_2(rope_head_dim // 2)
    num_warps = 4 if head_dim <= 256 else 8

    _fused_ape_pool_norm_rope_kernel[(bs,)](
        kv_score_gathered,
        kv_score_gathered.stride(0),
        kv_score_gathered.stride(1),
        ape,
        ape.stride(0),
        rms_weight,
        rms_eps,
        freqs_cis_real,
        freqs_cis_real.stride(0),
        out,
        out.stride(0),
        head_dim,
        rope_head_dim,
        half_dim,
        RATIO=ratio,
        K_POOL=K_IN,
        BLOCK_D=BLOCK_D,
        HALF_ROPE=HALF_ROPE,
        OVERLAP=int(overlap),
        num_warps=num_warps,
    )
    return out
