"""Fused MLA decode/verify query+key prep for aiter's head-padded asm kernel.

One launch for what the absorbed-MLA path otherwise runs as seven small ops:
    q_pad[t, h] = cat(q_nope[t, h], q_rope[t, h]) for h < H, zeros for H <= h < Hp
    kv[loc[t]]  = fp8(cat(k_nope[t], k_rope[t]))   (zeros when loc[t] == 0)
    kv[0]       = 0
Row 0 is where CUDA-graph padding tokens write their KV, and the asm kernel
reads it (weighted by p = 0) for masked positions, so it must stay finite.
The fp8 write is a plain cast, as MLATokenToKVPool.set_kv_buffer does.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_qk_prep_kernel(
    q_nope_ptr,
    q_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    loc_ptr,
    kv_ptr,
    q_out_ptr,
    stride_qn_t,
    stride_qn_h,
    stride_qr_t,
    stride_qr_h,
    stride_kn_t,
    stride_kr_t,
    stride_kv_row,
    stride_qo_t,
    stride_qo_h,
    H: tl.constexpr,
    HP: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    Q_FP8: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = tl.arange(0, DN)
    offs_r = tl.arange(0, DR)
    dst = q_out_ptr + t * stride_qo_t + h * stride_qo_h
    if h < H:
        qn = tl.load(q_nope_ptr + t * stride_qn_t + h * stride_qn_h + offs_n)
        qr = tl.load(q_rope_ptr + t * stride_qr_t + h * stride_qr_h + offs_r)
        if Q_FP8:
            # Unit-scale e4m3 query, saturated rather than overflowing to NaN.
            qn = tl.clamp(qn.to(tl.float32), -448.0, 448.0)
            qr = tl.clamp(qr.to(tl.float32), -448.0, 448.0)
        tl.store(dst + offs_n, qn.to(q_out_ptr.dtype.element_ty))
        tl.store(dst + DN + offs_r, qr.to(q_out_ptr.dtype.element_ty))
    else:
        tl.store(dst + offs_n, tl.zeros([DN], dtype=q_out_ptr.dtype.element_ty))
        tl.store(dst + DN + offs_r, tl.zeros([DR], dtype=q_out_ptr.dtype.element_ty))

    if h == 0:
        loc = tl.load(loc_ptr + t)
        keep = loc != 0
        kn = tl.load(k_nope_ptr + t * stride_kn_t + offs_n).to(tl.float32)
        kr = tl.load(k_rope_ptr + t * stride_kr_t + offs_r).to(tl.float32)
        kn = tl.where(keep, kn, 0.0)
        kr = tl.where(keep, kr, 0.0)
        row = kv_ptr + loc * stride_kv_row
        tl.store(row + offs_n, kn.to(kv_ptr.dtype.element_ty))
        tl.store(row + DN + offs_r, kr.to(kv_ptr.dtype.element_ty))
    if t == 0 and h == HP - 1:
        # Same zeros a padding token (loc 0) writes, so the overlap is benign.
        tl.store(kv_ptr + offs_n, tl.zeros([DN], dtype=kv_ptr.dtype.element_ty))
        tl.store(kv_ptr + DN + offs_r, tl.zeros([DR], dtype=kv_ptr.dtype.element_ty))


def mla_qk_prep(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cache_loc: torch.Tensor,
    kv_buffer: torch.Tensor,
    num_head_padded: int,
    q_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """q_nope [T, H, Dn], q_rope [T, H, Dr], k_nope / k_rope [T, (1,) Dn / Dr]
    (any row/head strides, unit inner stride), cache_loc [T], kv_buffer
    [S, 1, Dn + Dr] fp8. Writes the KV rows and returns q_pad [T, Hp, Dn + Dr]
    in q_dtype (default q_nope's; float8_e4m3fn gives a unit-scale fp8 query)."""
    T, H, DN = q_nope.shape
    DR = q_rope.shape[-1]
    assert q_rope.shape[:2] == (T, H) and kv_buffer.shape[-1] == DN + DR
    assert k_nope.shape[-1] == DN and k_rope.shape[-1] == DR
    assert kv_buffer.dtype == torch.float8_e4m3fn, "the KV write is a plain fp8 cast"
    assert all(x.stride(-1) == 1 for x in (q_nope, q_rope, k_nope, k_rope))
    q_pad = q_nope.new_empty(
        (T, num_head_padded, DN + DR), dtype=q_dtype or q_nope.dtype
    )
    if T == 0:
        return q_pad
    kv = kv_buffer.view(kv_buffer.shape[0], -1)
    k_nope2 = k_nope.squeeze(1) if k_nope.dim() == 3 else k_nope
    k_rope2 = k_rope.squeeze(1) if k_rope.dim() == 3 else k_rope
    _mla_qk_prep_kernel[(T, num_head_padded)](
        q_nope,
        q_rope,
        k_nope2,
        k_rope2,
        cache_loc,
        kv,
        q_pad,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope2.stride(0),
        k_rope2.stride(0),
        kv.stride(0),
        q_pad.stride(0),
        q_pad.stride(1),
        H=H,
        HP=num_head_padded,
        DN=DN,
        DR=DR,
        Q_FP8=q_pad.dtype == torch.float8_e4m3fn,
        num_warps=2,
    )
    return q_pad
