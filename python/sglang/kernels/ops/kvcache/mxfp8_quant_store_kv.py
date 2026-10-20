"""Quantize bf16 K/V rows to MXFP8 and scatter them into the paged KV cache."""

import torch
import triton
import triton.language as tl


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
