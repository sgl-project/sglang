"""Gather NHD FP8 KV pages directly into contiguous BF16 buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_dequant_kv(
    k_src,
    v_src,
    pages,
    k_dst,
    v_dst,
    n_pages,
    k_page_stride,
    v_page_stride,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGES_PER_PROGRAM: tl.constexpr,
):
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(PAGES_PER_PROGRAM):
        page = tl.program_id(0) * PAGES_PER_PROGRAM + i
        if page < n_pages:
            src = tl.load(pages + page).to(tl.int64)
            dst = page.to(tl.int64) * ROW + cols
            k = tl.load(k_src + src * k_page_stride + cols, mask=cols < ROW)
            v = tl.load(v_src + src * v_page_stride + cols, mask=cols < ROW)
            tl.store(k_dst + dst, k.to(tl.bfloat16), mask=cols < ROW)
            tl.store(v_dst + dst, v.to(tl.bfloat16), mask=cols < ROW)


def _page_contiguous(x: torch.Tensor) -> bool:
    if x.ndim != 4:
        return False
    _, page_size, heads, dim = x.shape
    return (
        x.stride(3) == 1
        and x.stride(2) == dim
        and (page_size == 1 or x.stride(1) == heads * dim)
    )


def gather_dequant_kv(k: torch.Tensor, v: torch.Tensor, pages: torch.Tensor):
    """Convert selected pages, preserving their order and duplicates.

    K/V have shape [pool_pages, page_size, heads, dim]. Only the inside of
    each page must be contiguous; stacked/interleaved K/V pools are supported.
    Values are converted without scales: attention still applies k_scale/v_scale.
    Page IDs must be valid pool indices, as for the original attention kernel.
    """
    if not (
        k.is_cuda
        and v.device == k.device
        and pages.device == k.device
        and k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and v.dtype == k.dtype
        and v.shape == k.shape
        and _page_contiguous(k)
        and _page_contiguous(v)
        and pages.ndim == 1
        and pages.is_contiguous()
        and pages.dtype in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected CUDA FP8 NHD pages and contiguous integer page IDs")
    n = pages.numel()
    shape = (n, *k.shape[1:])
    kb = torch.empty(shape, dtype=torch.bfloat16, device=k.device)
    vb = torch.empty_like(kb)
    if n:
        row = k.shape[1] * k.shape[2] * k.shape[3]
        block = min(triton.next_power_of_2(row), 2048)
        per_program = max(1, min(8, 4096 // block)) if row <= block else 1
        _gather_dequant_kv[(triton.cdiv(n, per_program), triton.cdiv(row, block))](
            k,
            v,
            pages,
            kb,
            vb,
            n,
            k.stride(0),
            v.stride(0),
            ROW=row,
            BLOCK=block,
            PAGES_PER_PROGRAM=per_program,
            num_warps=4,
        )
    return kb, vb
