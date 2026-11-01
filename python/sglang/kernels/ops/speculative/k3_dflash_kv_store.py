"""Fused RoPE + prefix-valid KV-cache write for DFlash draft KV materialization.

Replaces, per draft layer, ``rotary_embedding`` (on K and a dummy Q) +
``v.contiguous()`` + ``set_kv_buffer_prefix_valid_tiled`` with one kernel.

RoPE is bit-exact with sgl-kernel's ``rotary_embedding_kernel<c10::BFloat16,
IS_NEOX=true>``: every BFloat16 product/sum there rounds to bf16, so this kernel
rounds after each multiply and after the add/sub as well. cos/sin come from the
same bf16 copy of the cache the baseline kernel reads.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_store_prefix_valid_kernel(
    K,  # [T, H, D] bf16 (normed, pre-RoPE), row stride stride_kt
    V,  # [T, H, D] bf16 (strided ok)
    pos_ptr,  # [T] int64
    cos_sin_ptr,  # [max_pos, D] bf16 (cos | sin halves)
    loc_ptr,  # [bs, L] int64
    commit_ptr,  # [bs] int32
    K_Buf,
    V_Buf,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_cs,
    stride_loc_b,
    stride_kbuf,
    stride_kbuf_h,
    stride_vbuf,
    stride_vbuf_h,
    L: tl.constexpr,
    H: tl.constexpr,
    HALF: tl.constexpr,
):
    t = tl.program_id(0)
    b = t // L
    j = t % L
    commit = tl.load(commit_ptr + b)
    if j >= commit:
        return
    loc = tl.load(loc_ptr + b * stride_loc_b + j)
    pos = tl.load(pos_ptr + t)
    offs = tl.arange(0, HALF)
    c = tl.load(cos_sin_ptr + pos * stride_cs + offs).to(tl.float32)
    s = tl.load(cos_sin_ptr + pos * stride_cs + HALF + offs).to(tl.float32)
    for h in tl.static_range(H):
        kx = tl.load(K + t * stride_kt + h * stride_kh + offs).to(tl.float32)
        ky = tl.load(K + t * stride_kt + h * stride_kh + HALF + offs).to(tl.float32)
        xc = (kx * c).to(tl.bfloat16).to(tl.float32)
        ys = (ky * s).to(tl.bfloat16).to(tl.float32)
        yc = (ky * c).to(tl.bfloat16).to(tl.float32)
        xs = (kx * s).to(tl.bfloat16).to(tl.float32)
        ox = (xc - ys).to(tl.bfloat16)
        oy = (yc + xs).to(tl.bfloat16)
        tl.store(K_Buf + loc * stride_kbuf + h * stride_kbuf_h + offs, ox)
        tl.store(K_Buf + loc * stride_kbuf + h * stride_kbuf_h + HALF + offs, oy)
        v0 = tl.load(V + t * stride_vt + h * stride_vh + offs)
        v1 = tl.load(V + t * stride_vt + h * stride_vh + HALF + offs)
        tl.store(V_Buf + loc * stride_vbuf + h * stride_vbuf_h + offs, v0)
        tl.store(V_Buf + loc * stride_vbuf + h * stride_vbuf_h + HALF + offs, v1)


def rope_store_prefix_valid(
    k: torch.Tensor,  # [T, H, D]
    v: torch.Tensor,  # [T, H, D]
    positions: torch.Tensor,
    cos_sin_bf16: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    k_buffer: torch.Tensor,  # [slots, H, D]
    v_buffer: torch.Tensor,
) -> None:
    T, H, D = k.shape
    bs, L = loc_2d.shape
    assert T == bs * L and D % 2 == 0 and cos_sin_bf16.shape[-1] == D
    assert k.stride(2) == 1 and v.stride(2) == 1 and loc_2d.stride(1) == 1
    assert k_buffer.dtype == k.dtype == v.dtype == torch.bfloat16
    assert cos_sin_bf16.dtype == torch.bfloat16 and cos_sin_bf16.stride(1) == 1
    if T == 0:
        return
    _rope_store_prefix_valid_kernel[(T,)](
        k,
        v,
        positions,
        cos_sin_bf16,
        loc_2d,
        commit_lens,
        k_buffer,
        v_buffer,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos_sin_bf16.stride(0),
        loc_2d.stride(0),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        L=L,
        H=H,
        HALF=D // 2,
        num_warps=1,
    )
