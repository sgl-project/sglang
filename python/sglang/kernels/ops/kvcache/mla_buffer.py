from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.kvcache.paged_rows import paged_layout, paged_row_offset
from sglang.srt.runtime_context import get_parallel


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    reserved_skip_index,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    # A paged buffer (`paged_layout`): buffer_stride is then its row stride.
    PAGE_SIZE: tl.constexpr = 1,
    PAGE_STRIDE: tl.constexpr = 0,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    is_valid = (loc != reserved_skip_index) & (loc % DCP_WORLD_SIZE == DCP_RANK)
    safe_loc = tl.where(is_valid, loc, 0)
    safe_loc = safe_loc // DCP_WORLD_SIZE
    dst_ptr = (
        kv_buffer_ptr
        + paged_row_offset(safe_loc, PAGE_STRIDE, buffer_stride, PAGE_SIZE)
        + offs
    )

    # Three-way branch to handle boundary correctly while preserving fast path
    if base + BLOCK <= nope_dim:
        # Fast path: entire block is in nope region
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    elif base >= nope_dim:
        # Fast path: entire block is in rope region
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )
    else:
        # Boundary case: block spans nope/rope boundary (e.g., FP8 with nope_dim=528)
        # Handle each offset individually to avoid negative indexing
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < (nope_dim + rope_dim))

        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & is_rope,
            other=0,
        )

        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask & is_valid)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def set_mla_kv_buffer_kernel_norope(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    loc_ptr,
    reserved_skip_index,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    PAGE_STRIDE: tl.constexpr = 0,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < nope_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    is_valid = loc != reserved_skip_index
    safe_loc = tl.where(is_valid, loc, 0)
    dst_ptr = (
        kv_buffer_ptr
        + paged_row_offset(safe_loc, PAGE_STRIDE, buffer_stride, PAGE_SIZE)
        + offs
    )

    src = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs,
        mask=mask,
    )
    tl.store(dst_ptr, src, mask=mask & is_valid)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_naive(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: Optional[torch.Tensor],
    *,
    reserved_skip_index: int,
    dcp_world_size: int,
    dcp_rank: int,
) -> None:
    """Torch-native scatter for platforms that cannot launch Triton.

    Mirrors ``set_mla_kv_buffer_kernel``: rows land at
    ``kv_buffer[loc // dcp_world_size]`` as nope then rope, a loc equal to
    ``reserved_skip_index`` is dropped, and under DCP a rank keeps only the
    locs it owns. Duplicate locs race here exactly as they do in the kernel.
    """
    # view, not reshape: the destination must alias kv_buffer, and a layout
    # reshape would have to copy is one this write cannot serve. The sources
    # are read-only, so a copy there is harmless.
    paged = paged_layout(kv_buffer)
    page_size = 1 if paged is None else paged[0]
    if paged is None:
        rows = kv_buffer.view(kv_buffer.shape[0], -1)
    else:
        assert dcp_world_size == 1, "a paged buffer holds rank-local rows"
        rows = kv_buffer[:, :, 0, :]
    nope = cache_k_nope.reshape(cache_k_nope.shape[0], -1)
    nope_dim = nope.shape[-1]
    has_rope = cache_k_rope is not None and cache_k_rope.numel() > 0

    loc = loc.to(torch.int64)
    is_valid = loc != reserved_skip_index
    if dcp_world_size > 1:
        is_valid &= loc % dcp_world_size == dcp_rank
    dst = loc[is_valid] // dcp_world_size
    at = dst if page_size == 1 else (dst // page_size, dst % page_size)

    nope_rows = nope[is_valid]
    if page_size == 1:
        rows[dst, :nope_dim] = nope_rows
    else:
        rows[at[0], at[1], :nope_dim] = nope_rows
    if has_rope:
        rope = cache_k_rope.reshape(cache_k_rope.shape[0], -1)[is_valid]
        span = slice(nope_dim, nope_dim + rope.shape[-1])
        if page_size == 1:
            rows[dst, span] = rope
        else:
            rows[at[0], at[1], span] = rope


def _set_mla_kv_buffer_impl(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: Optional[torch.Tensor] = None,
    *,
    reserved_skip_index: int,
    dcp_world_size: int,
    dcp_rank: int,
):
    """Dispatch MLA paged-KV scatter writes to the fastest available path.

    Two paths:

    - SM90+ with TMA-compatible row widths: JIT CUDA kernel where each warp
      loads one (nope, rope) row into shared memory and issues a single
      ``cp.async.bulk.global.shared::cta`` store to scatter the row at
      ``kv_buffer[loc[item]]``. It packs 4-8 items per CTA, so the CTA count
      falls well below single-CTA-per-loc.
    - Otherwise: Triton kernel with ``BLOCK = next_pow2(nope_dim + rope_dim)``,
      i.e. one CTA per loc covering the entire row in one tile. This is the
      path for SM<90 and for shapes that violate the TMA 16-byte alignment.
    - Platforms declaring ``capabilities.supports_triton=False``: the
      torch-native scatter, which no device-specific kernel backs.

    Speedup vs the legacy BLOCK=128 Triton kernel on GB300 (BF16, nope=512,
    rope=64): ~1.05x at bs=8, ~1.5x at bs=128, 3.5x at bs=512, **11.7x at
    bs=16384**.

    Name retained for caller compatibility; the implementation is no longer
    Triton-only.

    Writes targeting ``reserved_skip_index`` are skipped. Slot 0 is reserved
    for CUDA-graph padding by default; pass -1 to disable skipping.

    Shared body of the two entry points below; the owner rule reaches it as
    ``1, 0`` (nothing to select) or as the live topology.
    """
    # Imported here, not at module scope: `current_platform` is a lazy module
    # attribute, so a top-level `from ... import` resolves the platform during
    # this module's import.
    from sglang.srt.platforms import current_platform

    if not current_platform.capabilities.supports_triton:
        set_mla_kv_buffer_naive(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            reserved_skip_index=reserved_skip_index,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
        )
        return

    has_rope = cache_k_rope is not None and cache_k_rope.numel() > 0
    n_loc = loc.numel()
    nope_dim = cache_k_nope.shape[-1]
    paged = paged_layout(kv_buffer)
    if paged is None:
        page_size, page_stride, row_stride = 1, 0, kv_buffer.stride(0)
    else:
        assert dcp_world_size == 1, "a paged buffer holds rank-local rows"
        page_size, page_stride, row_stride = paged
    page_kwargs = {"PAGE_SIZE": page_size, "PAGE_STRIDE": page_stride}

    if not has_rope:
        BLOCK = triton.next_power_of_2(nope_dim)
        grid = (n_loc, 1)
        pdl_kwargs = (
            {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        )
        set_mla_kv_buffer_kernel_norope[grid](
            kv_buffer,
            cache_k_nope,
            loc,
            reserved_skip_index,
            row_stride,
            cache_k_nope.stride(0),
            nope_dim,
            BLOCK=BLOCK,
            **page_kwargs,
            **pdl_kwargs,
        )
        return

    from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
        can_use_set_mla_kv_buffer,
    )
    from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
        set_mla_kv_buffer as jit_set_mla_kv_buffer,
    )

    nope_bytes = cache_k_nope.shape[-1] * cache_k_nope.element_size()
    rope_bytes = cache_k_rope.shape[-1] * cache_k_rope.element_size()
    if (
        is_arch_support_pdl()
        and can_use_set_mla_kv_buffer(nope_bytes, rope_bytes)
        and dcp_world_size == 1
    ):
        jit_set_mla_kv_buffer(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            reserved_skip_index=reserved_skip_index,
        )
        return

    # Fallback: Triton with BLOCK = next_pow2(total_dim). One CTA per loc; the
    # whole row in one tile (the existing 3-way nope/rope/boundary branch in
    # ``set_mla_kv_buffer_kernel`` handles the over-allocation past total_dim
    # via the offs<total_dim mask). Beats BLOCK=128 by 60-2700 ns across the
    # 2 <= bs <= 512 range on GB300.
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = triton.next_power_of_2(total_dim)
    grid = (n_loc, 1)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        reserved_skip_index,
        row_stride,
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        **page_kwargs,
        **pdl_kwargs,
    )


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    reserved_skip_index: int = 0,
):
    """Scatter at locs already addressing this rank's rows (widened ->
    `set_mla_kv_buffer_dcp_sharded_triton`)."""
    _set_mla_kv_buffer_impl(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        reserved_skip_index=reserved_skip_index,
        dcp_world_size=1,
        dcp_rank=0,
    )


def set_mla_kv_buffer_dcp_sharded_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    reserved_skip_index: int = 0,
):
    """Scatter at DCP-WIDENED locs: select this rank's ids and collapse them."""
    parallel = get_parallel()
    _set_mla_kv_buffer_impl(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        reserved_skip_index=reserved_skip_index,
        dcp_world_size=parallel.attn_dcp_size,
        dcp_rank=parallel.attn_dcp_rank,
    )


def set_mla_kv_buffer_cast_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    reserved_skip_index: int = 0,
):
    """Scatter bf16 rows into a lower-precision buffer (raw fp8), casting on
    store: one kernel instead of a cast per input plus the copy."""
    nope_dim, rope_dim = cache_k_nope.shape[-1], cache_k_rope.shape[-1]
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    set_mla_kv_buffer_kernel[(loc.numel(), 1)](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        reserved_skip_index,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=triton.next_power_of_2(nope_dim + rope_dim),
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        **pdl_kwargs,
    )


@triton.jit
def set_mla_kv_buffer_fp8_quant_kernel(
    kv_buffer_fp8_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    reserved_skip_index,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    DCP_RANK: tl.constexpr = 0,
    DCP_WORLD_SIZE: tl.constexpr = 1,
    USE_GDC: tl.constexpr = False,
):
    """Fuse BF16/FP16->FP8 cast with paged KV write."""
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    is_valid = (loc != reserved_skip_index) & (loc % DCP_WORLD_SIZE == DCP_RANK)
    safe_loc = tl.where(is_valid, loc, 0)
    safe_loc = safe_loc // DCP_WORLD_SIZE
    dst_ptr = kv_buffer_fp8_ptr + safe_loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
            other=0.0,
        )
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
            other=0.0,
        )
    else:
        is_nope = offs < nope_dim
        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0.0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & ~is_nope,
            other=0.0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    # Destination pointer is FP8-typed view; tl.store performs downcast.
    tl.store(dst_ptr, src, mask=mask & is_valid)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_triton_fp8_quant(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    reserved_skip_index: int = 0,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
):
    """Fuse BF16/FP16 MLA K quantization with paged KV write.

    Writes targeting ``reserved_skip_index`` are skipped. Pass -1 to disable.
    With ``dcp_world_size > 1`` the locs are DCP-widened: only this rank's slots
    are written, at row ``loc // dcp_world_size``.
    """
    kv_buffer_fp8 = kv_buffer.view(fp8_dtype)

    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    set_mla_kv_buffer_fp8_quant_kernel[grid](
        kv_buffer_fp8,
        cache_k_nope,
        cache_k_rope,
        loc,
        reserved_skip_index,
        kv_buffer_fp8.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        **pdl_kwargs,
    )


@triton.jit
def set_mla_kv_scale_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    reserved_skip_index,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim  # Make sure don't cross the boundary

    loc = tl.load(loc_ptr + pid_loc)
    is_valid = loc != reserved_skip_index
    safe_loc = tl.where(is_valid, loc, 0)
    dst_ptr = kv_buffer_ptr + safe_loc * buffer_stride + offs

    # Check each offs should read 'nope' or 'rope'
    is_nope = offs < nope_dim
    src_nope = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs, mask=mask & is_nope, other=0.0
    )
    src_rope = tl.load(
        cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
        mask=mask & ~is_nope,
        other=0.0,
    )

    # Combine nope + rope
    src = src_nope + src_rope
    tl.store(dst_ptr, src, mask=mask & is_valid)


def set_mla_kv_scale_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    reserved_skip_index: int = 0,
):
    """Write MLA scale rows while preserving the reserved padding slot."""
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128  # Keep origin, works for smaller total_dim as well.
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_scale_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        reserved_skip_index,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def get_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src_ptr = loc_src_ptr + nope_offs
    nope_src = tl.load(nope_src_ptr)

    tl.store(
        cache_k_nope_ptr + pid_loc * nope_stride + nope_offs,
        nope_src,
    )

    rope_offs = tl.arange(0, rope_dim)
    rope_src_ptr = loc_src_ptr + nope_dim + rope_offs
    rope_src = tl.load(rope_src_ptr)
    tl.store(
        cache_k_rope_ptr + pid_loc * rope_stride + rope_offs,
        rope_src,
    )


@triton.jit
def get_mla_kv_buffer_kernel_norope(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    loc_src_ptr = kv_buffer_ptr + loc * buffer_stride

    nope_offs = tl.arange(0, nope_dim)
    nope_src = tl.load(loc_src_ptr + nope_offs)
    tl.store(
        cache_k_nope_ptr + pid_loc * nope_stride + nope_offs,
        nope_src,
    )


def get_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: Optional[torch.Tensor] = None,
):
    # The source data type will be implicitly converted to the target data type.
    nope_dim = cache_k_nope.shape[-1]  # 512
    n_loc = loc.numel()
    grid = (n_loc,)

    has_rope = cache_k_rope is not None and cache_k_rope.numel() > 0
    if not has_rope:
        get_mla_kv_buffer_kernel_norope[grid](
            kv_buffer,
            cache_k_nope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            nope_dim,
        )
        return

    rope_dim = cache_k_rope.shape[-1]  # 64
    get_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


# ---------------------------------------------------------------------------
# NVFP4 D576 sparse-MLA cache rows (FlashInfer Prims-TS "nvfp4.fp8rope")
# ---------------------------------------------------------------------------
# One 352-byte record per token (flashinfer-ai/flashinfer#7410):
#   [0, 256)    512 NoPE values as E2M1, two per byte, even index in the low nibble
#   [256, 320)  64 RoPE values as unscaled E4M3
#   [320, 352)  32 E4M3 block scales (one per 16 NoPE values); block b sits at
#               byte 320 + 8 * (b % 4) + b // 4
# No global scale. Prims-TS expands value * scale to E4M3 in its loader before
# the FP8 MMA, so a dequantized element is e4m3(e2m1 * scale).
NVFP4_DS_MLA_ROW_BYTES = 352
NVFP4_DS_MLA_NOPE_DIM = 512
NVFP4_DS_MLA_ROPE_DIM = 64
# E4M3 subnormal floor for a block scale.
_NVFP4_MIN_BLOCK_SCALE = 2.0**-9


@triton.jit
def _e2m1_code(v):
    """Round-to-nearest-even E2M1 code (sign in bit 3) of an fp32 tensor; |v| > 6 saturates."""
    a = tl.abs(v)
    c = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return tl.where(v < 0, c | 8, c)


@triton.jit
def set_mla_kv_buffer_nvfp4_ds_mla_kernel(
    kv_buffer_ptr,  # uint8 rows
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    reserved_skip_index,
    buffer_stride,
    nope_stride,
    rope_stride,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid).to(tl.int64)
    is_valid = (loc != reserved_skip_index) & (loc % DCP_WORLD_SIZE == DCP_RANK)
    if is_valid:
        dst = kv_buffer_ptr + (loc // DCP_WORLD_SIZE) * buffer_stride
        src = cache_k_nope_ptr + pid.to(tl.int64) * nope_stride
        # [32 blocks, 8 byte pairs]: block b, pair j holds elements 16b+2j, 16b+2j+1
        blk = tl.arange(0, 32)[:, None]
        pair = tl.arange(0, 8)[None, :]
        offs = blk * 16 + 2 * pair
        x_even = tl.load(src + offs).to(tl.float32)
        x_odd = tl.load(src + offs + 1).to(tl.float32)
        amax = tl.max(tl.maximum(tl.abs(x_even), tl.abs(x_odd)), axis=1)
        # div_rn: Triton's "/" is approximate; the torch reference rounds.
        six = tl.full(amax.shape, 6.0, tl.float32)
        scale_e4m3 = tl.maximum(tl.math.div_rn(amax, six), 0.001953125).to(
            tl.float8e4nv
        )
        scale = scale_e4m3.to(tl.float32)[:, None]
        code_even = _e2m1_code(tl.math.div_rn(x_even, scale))
        code_odd = _e2m1_code(tl.math.div_rn(x_odd, scale))
        tl.store(dst + blk * 8 + pair, (code_even | (code_odd << 4)).to(tl.uint8))
        b = tl.arange(0, 32)
        tl.store(
            dst + 320 + 8 * (b % 4) + b // 4,
            scale_e4m3.to(tl.uint8, bitcast=True),
        )
        r = tl.arange(0, 64)
        rope = tl.load(cache_k_rope_ptr + pid.to(tl.int64) * rope_stride + r)
        tl.store(
            dst + 256 + r,
            rope.to(tl.float32).to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        )

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_nvfp4_ds_mla_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    reserved_skip_index: int = 0,
):
    """Quantize BF16 (or E4M3) [T, 512] NoPE + [T, 64] RoPE rows to the 352 B
    NVFP4 D576 record and scatter them, with the same skip-slot and DCP owner
    rule as ``set_mla_kv_buffer_kernel`` (row ``loc // dcp_world_size`` on the
    rank with ``loc % dcp_world_size == dcp_rank``)."""
    rows = kv_buffer.view(torch.uint8).view(kv_buffer.shape[0], -1)
    assert rows.shape[1] == NVFP4_DS_MLA_ROW_BYTES and rows.stride(1) == 1
    nope = cache_k_nope.reshape(cache_k_nope.shape[0], -1)
    rope = cache_k_rope.reshape(cache_k_rope.shape[0], -1)
    assert nope.shape[1] == NVFP4_DS_MLA_NOPE_DIM and nope.stride(1) == 1
    assert rope.shape[1] == NVFP4_DS_MLA_ROPE_DIM and rope.stride(1) == 1
    n_loc = loc.numel()
    if n_loc == 0:
        return
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    set_mla_kv_buffer_nvfp4_ds_mla_kernel[(n_loc,)](
        rows,
        nope,
        rope,
        loc,
        reserved_skip_index,
        rows.stride(0),
        nope.stride(0),
        rope.stride(0),
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        num_warps=1,
        **pdl_kwargs,
    )


def _e2m1_lut(device) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device
    )


def _e2m1_codes_ref(v: torch.Tensor) -> torch.Tensor:
    a = v.abs()
    code = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    return torch.where(v < 0, code | 8, code)


def quantize_nvfp4_ds_mla_reference(
    k_nope: torch.Tensor, k_rope: torch.Tensor
) -> torch.Tensor:
    """Torch reference of ``set_mla_kv_buffer_nvfp4_ds_mla_kernel``: [T, 352] uint8."""
    x = k_nope.reshape(-1, 32, 16).float()
    t = x.shape[0]
    amax = x.abs().amax(-1)
    scale_e4m3 = (amax / 6.0).clamp_min(_NVFP4_MIN_BLOCK_SCALE).to(torch.float8_e4m3fn)
    v = x / scale_e4m3.float()[..., None]
    code = _e2m1_codes_ref(v).reshape(t, 256, 2)
    out = torch.empty(t, NVFP4_DS_MLA_ROW_BYTES, dtype=torch.uint8, device=x.device)
    out[:, :256] = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    out[:, 256:320] = (
        k_rope.reshape(t, 64).float().to(torch.float8_e4m3fn).view(torch.uint8)
    )
    out[:, 320:] = (
        scale_e4m3.view(torch.uint8).unflatten(-1, (8, 4)).transpose(-1, -2).flatten(-2)
    )
    return out


def dequantize_nvfp4_ds_mla(
    rows: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, 352] uint8 records -> ([T, 512], [T, 64]) as the Prims-TS loader sees
    them: NoPE is e4m3(e2m1 * block scale), RoPE is the stored E4M3."""
    rows = rows.reshape(-1, NVFP4_DS_MLA_ROW_BYTES)
    t = rows.shape[0]
    packed = rows[:, :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(t, 32, 16)
    vals = _e2m1_lut(rows.device)[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    scales = (
        rows[:, 320:]
        .reshape(t, 4, 8)
        .transpose(-1, -2)
        .reshape(t, 32)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = (vals * scales[..., None]).reshape(t, 512).clamp(-448, 448)
    nope = nope.to(torch.float8_e4m3fn).to(dtype)
    rope = rows[:, 256:320].contiguous().view(torch.float8_e4m3fn).to(dtype)
    return nope, rope
