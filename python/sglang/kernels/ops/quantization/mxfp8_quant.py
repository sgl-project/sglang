"""MXFP8 quantization helpers for Inkling attention."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

MXFP8_BLOCK_SIZE = 32


class MXFP8Tensor(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    ssm,
    ssk,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)

    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    tl.store(
        s_ptr + offs_m * ssm + pid_b * ssk,
        scale_biased.to(tl.uint8),
        mask=m_mask,
    )


def to_mxfp8(x: torch.Tensor) -> MXFP8Tensor:
    """Quantize the last dimension into E4M3 values plus E8M0 scale bytes."""
    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires last dim divisible by {MXFP8_BLOCK_SIZE}, "
            f"got {x.shape[-1]}."
        )
    orig_shape = x.shape
    x2d = x.contiguous().view(-1, orig_shape[-1])
    M, K = x2d.shape
    xq = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
    scales = torch.empty((M, K // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)
    block_m = 64
    grid = (triton.cdiv(M, block_m), K // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_kernel[grid](
        x2d,
        xq,
        scales,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=block_m,
    )
    return MXFP8Tensor(
        data=xq.view(orig_shape),
        scale=scales.view(*orig_shape[:-1], K // MXFP8_BLOCK_SIZE),
    )


@triton.jit
def _mxfp8_quant_store_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    loc_ptr,
    q8_ptr,
    sfq_ptr,
    kc_ptr,
    vc_ptr,
    sfk_ptr,
    sfv_ptr,
    sqt,
    sqh,
    skt,
    skh,
    svt,
    svh,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
):
    t = tl.program_id(0)
    r = tl.program_id(1)
    SF: tl.constexpr = D // 32
    blk = tl.arange(0, SF)
    off2 = blk[:, None] * 32 + tl.arange(0, 32)[None, :]

    if r < NQ:
        src = q_ptr + t * sqt + r * sqh
    elif r < NQ + NKV:
        src = k_ptr + t * skt + (r - NQ) * skh
    else:
        src = v_ptr + t * svt + (r - NQ - NKV) * svh

    x = tl.load(src + off2).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(q8_ptr.dtype.element_ty)
    sf = scale_biased.to(tl.uint8)

    if r < NQ:
        tl.store(q8_ptr + (t * NQ + r) * D + off2, xq)
        tl.store(sfq_ptr + (t * NQ + r) * SF + blk, sf)
    else:
        myloc = tl.load(loc_ptr + t).to(tl.int64)
        # Slot 0 is the reserved CUDA-graph padding sink; skip writes to it.
        if myloc == 0:
            return
        if r < NQ + NKV:
            h = r - NQ
            cache = kc_ptr
            sfb = sfk_ptr
        else:
            h = r - NQ - NKV
            cache = vc_ptr
            sfb = sfv_ptr
        tl.store(cache + (myloc * NKV + h) * D + off2, xq)
        # BlockScaledBasicChunk byte layout (see mxfp8_interleave_sf.py):
        # a page block is (NKV, 32, PAGE//32, SF) bytes.
        po = myloc % PAGE
        base = (
            ((myloc // PAGE) * NKV + h) * (32 * (PAGE // 32) * SF)
            + (po % 32) * ((PAGE // 32) * SF)
            + (po // 32) * SF
        )
        tl.store(sfb + base + blk, sf)


def quant_store_kv_mxfp8(
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sfk: torch.Tensor,
    sfv: torch.Tensor,
    page_size: int = 128,
) -> None:
    """One kernel quantizes bf16 K/V rows, scatters the fp8 payload into the paged
    cache at `loc`, and writes UE8M0 scales in the interleaved FA4 layout."""
    T, NKV, D = k.shape
    assert D % 32 == 0 and page_size % 32 == 0
    assert k.stride(2) == 1 and v.stride(2) == 1
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    if T == 0:
        return
    _mxfp8_quant_store_qkv_kernel[(T, 2 * NKV)](
        k,  # unused q_ptr (same elem dtype)
        k,
        v,
        loc,
        k_cache,  # unused q8_ptr, supplies the fp8 store dtype
        sfk.view(torch.uint8),  # unused sfq_ptr (uint8, matches the dead store)
        k_cache,
        v_cache,
        sfk.view(torch.uint8),
        sfv.view(torch.uint8),
        0,
        0,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        NQ=0,
        NKV=NKV,
        D=D,
        PAGE=page_size,
    )
