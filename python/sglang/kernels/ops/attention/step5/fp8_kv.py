# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

STEP5_FP8_KV_DIM = 192
_FP8_MAX = 448.0


@triton.jit
def _store_kernel(
    K,
    V,
    Loc,
    KBuf,
    VBuf,
    KScale,
    VScale,
    k_stride,
    v_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    loc = tl.load(Loc + t).to(tl.int64)
    d = tl.arange(0, BLOCK)
    mask = d < D
    k = tl.load(K + t * k_stride + d, mask, other=0.0).to(tl.float32)
    v = tl.load(V + t * v_stride + d, mask, other=0.0).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(k), axis=0), 1e-12) / FP8_MAX
    vs = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-12) / FP8_MAX
    tl.store(KBuf + loc * D + d, (k / ks).to(KBuf.dtype.element_ty), mask)
    tl.store(VBuf + loc * D + d, (v / vs).to(VBuf.dtype.element_ty), mask)
    tl.store(KScale + loc, ks)
    tl.store(VScale + loc, vs)


@triton.jit
def _dequant_pages_kernel(
    Pages,
    KBuf,
    VBuf,
    KScale,
    VScale,
    KOut,
    VOut,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    TOK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1).to(tl.int64)
    page = tl.load(Pages + index).to(tl.int64)
    tok = part * TOK + tl.arange(0, TOK).to(tl.int64)
    d = tl.arange(0, BLOCK)
    mask = (tok[:, None] >= 0) & (d[None, :] < D)
    src = page * PAGE + tok
    out = index * PAGE + tok
    ks = tl.load(KScale + src)
    vs = tl.load(VScale + src)
    k = tl.load(KBuf + src[:, None] * D + d[None, :], mask, other=0.0)
    v = tl.load(VBuf + src[:, None] * D + d[None, :], mask, other=0.0)
    tl.store(
        KOut + out[:, None] * D + d[None, :],
        (k.to(tl.float32) * ks[:, None]).to(KOut.dtype.element_ty),
        mask,
    )
    tl.store(
        VOut + out[:, None] * D + d[None, :],
        (v.to(tl.float32) * vs[:, None]).to(VOut.dtype.element_ty),
        mask,
    )


@triton.jit
def _remap_pages_kernel(
    Packed,
    PageMap,
    PackedOut,
    cols,
    packed_stride,
    out_stride,
    BLOCK: tl.constexpr,
    REGIONS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < cols
    value = tl.load(Packed + row * packed_stride + col, mask, other=-1)
    live = value >= 0
    region = value >> 32
    logical = value & 0xFFFFFFFF
    page = tl.load(PageMap + region // REGIONS_PER_PAGE, mask & live, other=0)
    remapped = ((page * REGIONS_PER_PAGE + region % REGIONS_PER_PAGE) << 32) | logical
    tl.store(PackedOut + row * out_stride + col, tl.where(live, remapped, value), mask)


def _check_cache(k_buf, v_buf, k_scale, v_scale):
    for tensor in (k_buf, v_buf):
        if (
            tensor.dtype != torch.float8_e4m3fn
            or tensor.shape[-1] != STEP5_FP8_KV_DIM
            or not tensor.is_contiguous()
        ):
            raise ValueError("Step5 FP8 KV requires contiguous E4M3 [slots,1,192]")
    for tensor in (k_scale, v_scale):
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("Step5 FP8 KV scales must be contiguous FP32 [slots]")


def step5_fp8_kv_store(k, v, loc, k_buf, v_buf, k_scale, v_scale):
    _check_cache(k_buf, v_buf, k_scale, v_scale)
    tokens = loc.numel()
    k = k.reshape(tokens, -1)
    v = v.reshape(tokens, -1)
    if k.shape[1] != STEP5_FP8_KV_DIM or k.stride(1) != 1 or v.stride(1) != 1:
        raise ValueError("Step5 FP8 KV store requires [T,192] rows with unit stride")
    if tokens == 0:
        return
    _store_kernel[(tokens,)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        k.stride(0),
        v.stride(0),
        D=STEP5_FP8_KV_DIM,
        BLOCK=256,
        FP8_MAX=_FP8_MAX,
        num_warps=2,
    )


def step5_fp8_prefill_pages(req_to_token, slots, totals, total_list, num_pages):
    device = req_to_token.device
    counts = [(total + 63) // 64 for total in total_list]
    num_live = sum(counts)
    page_map = torch.zeros(num_pages, dtype=torch.int64, device=device)
    if num_live == 0:
        return torch.empty(0, dtype=torch.int64, device=device), page_map
    count_t = (totals + 63) // 64
    owner = torch.repeat_interleave(
        torch.arange(len(counts), device=device), count_t, output_size=num_live
    )
    first = torch.cumsum(count_t, 0) - count_t
    starts = (torch.arange(num_live, device=device) - first[owner]) * 64
    tokens = req_to_token[slots.long()[owner], starts]
    pages = tokens.to(torch.int64) // 64
    # Different requests can share physical prefix pages. Materialize each
    # page only once so scratch space and dequantization scale with live KV,
    # not the sum of all request lengths. A single request owns distinct pages.
    if len(counts) > 1:
        pages = torch.unique(pages)
    page_map[pages] = torch.arange(pages.numel(), device=device)
    return pages, page_map


def step5_fp8_kv_dequant_pages(pages, k_buf, v_buf, k_scale, v_scale, k_out, v_out):
    _check_cache(k_buf, v_buf, k_scale, v_scale)
    if k_out.shape[0] < pages.numel() * 64:
        raise ValueError("Step5 FP8 prefill scratch is too small")
    if pages.numel() == 0:
        return
    _dequant_pages_kernel[(pages.numel(), 4)](
        pages,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        k_out,
        v_out,
        D=STEP5_FP8_KV_DIM,
        BLOCK=256,
        PAGE=64,
        TOK=16,
        num_warps=4,
    )


def step5_fp8_kv_remap_pages(packed, page_map, packed_out):
    rows, cols = packed.shape
    if packed_out.shape != packed.shape or packed.stride(1) != 1:
        raise ValueError("Step5 FP8 prefill remap received invalid buffers")
    if rows == 0:
        return
    _remap_pages_kernel[(rows, triton.cdiv(cols, 512))](
        packed,
        page_map,
        packed_out,
        cols,
        packed.stride(0),
        packed_out.stride(0),
        BLOCK=512,
        REGIONS_PER_PAGE=8,
        num_warps=4,
    )
