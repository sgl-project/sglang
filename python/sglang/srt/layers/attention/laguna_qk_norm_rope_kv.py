"""Fused decode-time attention prep for Laguna: per-head q/k RMSNorm + neox RoPE + fp8 KV-cache store.

Replaces, per layer, the qkv split copies, two RMSNorms, the RoPE kernel, the k/v scale divisions,
the fp8 casts and the KV store (about ten small kernels) with one Triton launch. q is written to a
fresh contiguous buffer; k and v go straight into the fp8 KV cache, so the attention call must use
save_kv_cache=False.

Numerics: rounds to the activation dtype at the same points as the unfused ROCm path (norm output, each
RoPE product and sum, the k/v scale division), so q and the fp8 KV bytes match it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kv_kernel(
    qkv_ptr,
    qkv_stride,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    q_out_ptr,
    k_buf_ptr,
    v_buf_ptr,
    kv_stride_slot,
    kv_stride_head,
    loc_ptr,
    eps,
    inv_k_scale,
    inv_v_scale,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    HALF: tl.constexpr,
    PAGE: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    D: tl.constexpr = 2 * HALF
    offs = tl.arange(0, HALF)
    row = qkv_ptr + m.to(tl.int64) * qkv_stride
    pos = tl.load(pos_ptr + m).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * cos_sin_stride + offs).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * cos_sin_stride + HALF + offs).to(tl.float32)
    if h < NQ:
        src = row + h * D
        w_ptr = qw_ptr
    else:
        src = row + (NQ + (h - NQ)) * D
        w_ptr = kw_ptr
    lo = tl.load(src + offs).to(tl.float32)
    hi = tl.load(src + HALF + offs).to(tl.float32)
    # One reduction over the whole head, as the aiter rmsnorm kernel does (matching rounding).
    full = tl.load(src + tl.arange(0, D)).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(full * full, 0) / D + eps)
    act = q_out_ptr.dtype.element_ty
    lo = (lo * rstd * tl.load(w_ptr + offs).to(tl.float32)).to(act).to(tl.float32)
    hi = (
        (hi * rstd * tl.load(w_ptr + HALF + offs).to(tl.float32)).to(act).to(tl.float32)
    )
    # Same per-op rounding as sglang's HIP rotary kernel (bf16 cos/sin, bf16 products and sum).
    cos = cos.to(act).to(tl.float32)
    sin = sin.to(act).to(tl.float32)
    out_lo = ((lo * cos).to(act).to(tl.float32) - (hi * sin).to(act).to(tl.float32)).to(
        act
    )
    out_hi = ((hi * cos).to(act).to(tl.float32) + (lo * sin).to(act).to(tl.float32)).to(
        act
    )
    if h < NQ:
        dst = q_out_ptr + m.to(tl.int64) * (NQ * D) + h * D
        tl.store(dst + offs, out_lo)
        tl.store(dst + HALF + offs, out_hi)
    else:
        kh = h - NQ
        slot = tl.load(loc_ptr + m).to(tl.int64)
        if PAGE == 0:
            kdst = k_buf_ptr + slot * kv_stride_slot + kh * kv_stride_head
            k_off_lo = offs
            k_off_hi = HALF + offs
        else:
            # SHUFFLE 5D K [blocks, NK, D // 16, PAGE, 16] (fp8, 16 elements per vector).
            blk = slot // PAGE
            pin = slot % PAGE
            kdst = k_buf_ptr + blk * (NK * D * PAGE) + kh * (D * PAGE) + pin * 16
            k_off_lo = (offs // 16) * (PAGE * 16) + offs % 16
            k_off_hi = ((HALF + offs) // 16) * (PAGE * 16) + (HALF + offs) % 16
        k_lo = (out_lo.to(tl.float32) * inv_k_scale).to(act).to(tl.float32)
        k_hi = (out_hi.to(tl.float32) * inv_k_scale).to(act).to(tl.float32)
        tl.store(kdst + k_off_lo, k_lo.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
        tl.store(kdst + k_off_hi, k_hi.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
        vsrc = row + (NQ + NK + kh) * D
        v_lo = (
            (tl.load(vsrc + offs).to(tl.float32) * inv_v_scale).to(act).to(tl.float32)
        )
        v_hi = (
            (tl.load(vsrc + HALF + offs).to(tl.float32) * inv_v_scale)
            .to(act)
            .to(tl.float32)
        )
        if PAGE == 0:
            vdst = v_buf_ptr + slot * kv_stride_slot + kh * kv_stride_head
            v_off_lo = offs
            v_off_hi = HALF + offs
        else:
            # SHUFFLE 5D V [blocks, NK, PAGE // 16, D, 16].
            vdst = (
                v_buf_ptr
                + blk * (NK * PAGE * D)
                + kh * (PAGE * D)
                + (pin // 16) * (D * 16)
                + pin % 16
            )
            v_off_lo = offs * 16
            v_off_hi = (HALF + offs) * 16
        tl.store(vdst + v_off_lo, v_lo.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
        tl.store(vdst + v_off_hi, v_hi.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


def qk_norm_rope_kv_store(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    loc: torch.Tensor,
    k_scale: float,
    v_scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """qkv [M, (NQ + 2 NK) * D] bf16 (row stride may exceed the width); k_buf/v_buf uint8 holding fp8
    e4m3fn bytes, either NHD [slots, NK, D] or SHUFFLE 5D (K [blocks, NK, D // 16, page, 16], V [blocks, NK,
    page // 16, D, 16]). Returns q [M, NQ * D] (normed + rotated)."""
    m = qkv.shape[0]
    q_out = torch.empty((m, num_q_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    if m == 0:
        return q_out
    page = k_buf.shape[3] if k_buf.dim() == 5 else 0
    _qk_norm_rope_kv_kernel[(m, num_q_heads + num_kv_heads)](
        qkv,
        qkv.stride(0),
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_out,
        k_buf,
        v_buf,
        0 if page else k_buf.stride(0),
        0 if page else k_buf.stride(1),
        loc,
        eps,
        1.0 / k_scale,
        1.0 / v_scale,
        NQ=num_q_heads,
        NK=num_kv_heads,
        HALF=head_dim // 2,
        PAGE=page,
        num_warps=4,  # same reduction tree as the aiter rmsnorm kernel (default 4 warps)
    )
    return q_out
