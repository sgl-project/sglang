"""Seg-packed MLA KV pages for aiter's gfx1250 non-persistent asm MLA decode.

The kernel reads each KV page as ``[page_size x nope | page_size x rope]``
instead of ``page_size`` token rows of ``[nope | rope]``. The pool keeps its
``(num_slots, 1, nope + rope)`` shape and byte size; only the placement of a
token inside its page changes, so token ``loc`` lives at

    nope: page * page_size * (nope + rope) + slot * nope
    rope: page * page_size * (nope + rope) + page_size * nope + slot * rope

with ``page, slot = divmod(loc, page_size)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _seg_offsets(loc, PAGE_SIZE: tl.constexpr, NOPE: tl.constexpr, ROPE: tl.constexpr):
    page = loc // PAGE_SIZE
    slot = loc % PAGE_SIZE
    base = page * (PAGE_SIZE * (NOPE + ROPE))
    return base + slot * NOPE, base + PAGE_SIZE * NOPE + slot * ROPE


@triton.jit
def _seg_write_kernel(
    kv_ptr,
    nope_ptr,
    rope_ptr,
    loc_ptr,
    nope_stride,
    rope_stride,
    PAGE_SIZE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    loc = tl.load(loc_ptr + pid).to(tl.int64)
    # Slot 0 is the padded dummy slot; same skip as set_mla_kv_buffer_triton.
    valid = loc != 0
    nope_off, rope_off = _seg_offsets(loc, PAGE_SIZE, NOPE, ROPE)
    offs_n = tl.arange(0, NOPE)
    tl.store(
        kv_ptr + nope_off + offs_n,
        tl.load(nope_ptr + pid * nope_stride + offs_n),
        mask=valid & (offs_n < NOPE),
    )
    offs_r = tl.arange(0, ROPE)
    tl.store(
        kv_ptr + rope_off + offs_r,
        tl.load(rope_ptr + pid * rope_stride + offs_r),
        mask=valid & (offs_r < ROPE),
    )


@triton.jit
def _seg_read_kernel(
    kv_ptr,
    nope_ptr,
    rope_ptr,
    loc_ptr,
    nope_stride,
    rope_stride,
    PAGE_SIZE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    loc = tl.load(loc_ptr + pid).to(tl.int64)
    nope_off, rope_off = _seg_offsets(loc, PAGE_SIZE, NOPE, ROPE)
    offs_n = tl.arange(0, NOPE)
    nope = tl.load(kv_ptr + nope_off + offs_n)
    tl.store(
        nope_ptr + pid * nope_stride + offs_n,
        nope.to(nope_ptr.dtype.element_ty),
    )
    offs_r = tl.arange(0, ROPE)
    rope = tl.load(kv_ptr + rope_off + offs_r)
    tl.store(
        rope_ptr + pid * rope_stride + offs_r,
        rope.to(rope_ptr.dtype.element_ty),
    )


def _row_stride(t: torch.Tensor) -> int:
    assert t.stride(-1) == 1
    return t.stride(0)


def seg_set_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter token rows into seg-packed pages. Inputs already in pool dtype."""
    n = loc.numel()
    if n == 0:
        return
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    assert kv_buffer.is_contiguous() and kv_buffer.shape[-1] == nope_dim + rope_dim
    nope = cache_k_nope.reshape(n, nope_dim)
    rope = cache_k_rope.reshape(n, rope_dim)
    _seg_write_kernel[(n,)](
        kv_buffer,
        nope,
        rope,
        loc,
        _row_stride(nope),
        _row_stride(rope),
        PAGE_SIZE=page_size,
        NOPE=nope_dim,
        ROPE=rope_dim,
    )


def seg_get_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    page_size: int,
) -> None:
    """Gather token rows out of seg-packed pages, casting to the output dtype."""
    n = loc.numel()
    if n == 0:
        return
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    assert kv_buffer.is_contiguous() and kv_buffer.shape[-1] == nope_dim + rope_dim
    nope = cache_k_nope.view(n, nope_dim)
    rope = cache_k_rope.view(n, rope_dim)
    _seg_read_kernel[(n,)](
        kv_buffer,
        nope,
        rope,
        loc,
        _row_stride(nope),
        _row_stride(rope),
        PAGE_SIZE=page_size,
        NOPE=nope_dim,
        ROPE=rope_dim,
    )


def seg_gather_kv_rows(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    nope_dim: int,
    page_size: int,
    dst_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Token-major ``(n, 1, nope + rope)`` rows, as ``kv_buffer[loc]`` would be."""
    dim = kv_buffer.shape[-1]
    out = torch.empty(
        (loc.numel(), 1, dim),
        dtype=dst_dtype or kv_buffer.dtype,
        device=kv_buffer.device,
    )
    flat = out.view(-1, dim)
    seg_get_mla_kv_buffer(
        kv_buffer, loc, flat[:, :nope_dim], flat[:, nope_dim:], page_size
    )
    return out


@triton.jit
def _page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    last_page_len_ptr,
    req_to_token_stride,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    rp = tl.load(req_pool_indices_ptr + b).to(tl.int64)
    seq = tl.load(seq_lens_ptr + b).to(tl.int32)
    seq = tl.maximum(seq, 1)
    n_pages = tl.cdiv(seq, PAGE_SIZE)
    start = tl.load(kv_indptr_ptr + b)
    tl.store(last_page_len_ptr + b, seq - (n_pages - 1) * PAGE_SIZE)
    row = req_to_token_ptr + rp * req_to_token_stride
    for p0 in range(0, n_pages, BLOCK):
        p = p0 + tl.arange(0, BLOCK)
        mask = p < n_pages
        tok = tl.load(row + p * PAGE_SIZE, mask=mask, other=0)
        tl.store(kv_indices_ptr + start + p, (tok // PAGE_SIZE).to(tl.int32), mask=mask)


def build_mla_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
) -> None:
    """Page-level ragged indptr / page ids / last-page lengths, graph safe.

    ``kv_indptr`` holds ``bs + 1`` entries; a zero length counts as one token
    so cuda-graph padding rows still point at a real (dummy) page.
    """
    bs = req_pool_indices.shape[0]
    n_pages = (seq_lens.clamp_min(1) + page_size - 1) // page_size
    kv_indptr[:1].zero_()
    torch.cumsum(n_pages.to(torch.int32), dim=0, out=kv_indptr[1 : bs + 1])
    _page_table_kernel[(bs,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_indptr,
        kv_indices,
        last_page_len,
        req_to_token.stride(0),
        PAGE_SIZE=page_size,
        BLOCK=256,
    )


@triton.jit
def _quant_pad_q_kernel(
    q_ptr,
    out_ptr,
    scale_ptr,
    q_row_stride,
    out_row_stride,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < DIM
    x = tl.load(q_ptr + row * q_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.load(scale_ptr)
    y = tl.clamp(x * inv, -FP8_MAX, FP8_MAX)
    tl.store(
        out_ptr + row * out_row_stride + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def quant_pad_mla_q(
    q: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    padded_dim: int = 768,
):
    """bf16 ``(T, H, D)`` Q -> fp8 rows of ``padded_dim`` and a per-tensor scale.

    ``scale`` None computes a dynamic per-tensor scale (one extra reduction);
    otherwise values are divided by it and saturated.
    Returns the ``(T, H, D)`` view of the padded buffer and the fp32 scale.
    """
    t, h, d = q.shape
    fp8_max = torch.finfo(fp8_dtype).max
    if scale is None:
        scale = (q.abs().amax().float() / fp8_max).clamp_min(1e-10).reshape(1)
    out = torch.empty((t, h, padded_dim), dtype=fp8_dtype, device=q.device)
    rows = q.reshape(t * h, d)
    if t:
        _quant_pad_q_kernel[(t * h,)](
            rows,
            out,
            scale,
            rows.stride(0),
            padded_dim,
            DIM=d,
            BLOCK=triton.next_power_of_2(d),
            FP8_MAX=fp8_max,
        )
    return out[..., :d], scale
