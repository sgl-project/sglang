"""Kimi-K3 (NoPE MLA) absorbed-q concat + latent KV-cache write, one launch (ROCm).

K3 has no rotary embedding, so ``fused_qk_rope_cat_and_cache_mla`` does not
apply. Bit-identical to torch.cat + set_kv_buffer (same RNE bf16 -> e4m3fn cast,
no scale).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_mla_cat_cache_kernel(
    qn_ptr,  # [M, H, DN] (any row strides, unit last stride)
    qp_ptr,  # [M, H, DP]
    kn_ptr,  # [M, DN]
    kp_ptr,  # [M, DP]
    loc_ptr,  # [M] int32/int64
    kv_ptr,  # [rows, DN + DP] cache (bf16 or fp8)
    qo_ptr,  # [M, H, DN + DP] out
    s_qn_m,
    s_qn_h,
    s_qp_m,
    s_qp_h,
    s_kn_m,
    s_kp_m,
    s_kv,
    s_qo_m,
    s_qo_h,
    H: tl.constexpr,
    DN: tl.constexpr,
    DP: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = tl.arange(0, DN)
    offs_p = tl.arange(0, DP)
    if h < H:
        xn = tl.load(qn_ptr + m * s_qn_m + h * s_qn_h + offs_n)
        xp = tl.load(qp_ptr + m * s_qp_m + h * s_qp_h + offs_p)
        base = qo_ptr + m * s_qo_m + h * s_qo_h
        tl.store(base + offs_n, xn)
        tl.store(base + DN + offs_p, xp)
    else:
        loc = tl.load(loc_ptr + m).to(tl.int64)
        kn = tl.load(kn_ptr + m * s_kn_m + offs_n)
        kp = tl.load(kp_ptr + m * s_kp_m + offs_p)
        dst = kv_ptr + loc * s_kv
        tl.store(dst + offs_n, kn.to(kv_ptr.dtype.element_ty))
        tl.store(dst + DN + offs_p, kp.to(kv_ptr.dtype.element_ty))


def covered(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> bool:
    """Shape/dtype gate; uncovered calls keep the cat + set_kv_buffer path."""
    if q_nope.dim() != 3 or q_pe.dim() != 3:
        return False
    m, h, dn = q_nope.shape
    dp = q_pe.shape[-1]
    if m == 0 or q_pe.shape[:2] != (m, h):
        return False
    if k_nope.shape[0] != m or k_pe.shape[0] != m or loc.shape[0] != m:
        return False
    if k_nope.numel() != m * dn or k_pe.numel() != m * dp:
        return False
    if kv_buffer.shape[-1] != dn + dp or kv_buffer.numel() != kv_buffer.shape[0] * (
        dn + dp
    ):
        return False
    if dn & (dn - 1) or dp & (dp - 1):
        return False
    if q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
        return False
    if k_nope.dtype != torch.bfloat16 or k_pe.dtype != torch.bfloat16:
        return False
    if kv_buffer.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    if (
        loc.dtype not in (torch.int32, torch.int64)
        or loc.dim() != 1
        or loc.stride(0) != 1
    ):
        return False
    if any(t.stride(-1) != 1 for t in (q_nope, q_pe, k_nope, k_pe, kv_buffer)):
        return False
    if not kv_buffer.is_contiguous():
        return False
    return True


def k3_mla_cat_cache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> torch.Tensor:
    """Write kv_buffer[loc[m]] = cast([k_nope[m] | k_pe[m]]) and return
    q = [q_nope | q_pe] as a new contiguous [M, H, DN + DP] tensor.

    q_nope [M, H, DN], q_pe [M, H, DP] (strided views are fine),
    k_nope [M, (1,) DN], k_pe [M, (1,) DP], kv_buffer [rows, (1,) DN + DP]
    (the layer's key buffer in the cache dtype), loc [M]. Caller checks
    covered()."""
    m, h, dn = q_nope.shape
    dp = q_pe.shape[-1]
    kn = k_nope.reshape(m, dn)
    kp = k_pe.reshape(m, dp)
    kv = kv_buffer.view(kv_buffer.shape[0], dn + dp)
    q_out = torch.empty((m, h, dn + dp), dtype=q_nope.dtype, device=q_nope.device)
    _k3_mla_cat_cache_kernel[(m, h + 1)](
        q_nope,
        q_pe,
        kn,
        kp,
        loc,
        kv,
        q_out,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kn.stride(0),
        kp.stride(0),
        kv.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        H=h,
        DN=dn,
        DP=dp,
        num_warps=2,
    )
    return q_out
