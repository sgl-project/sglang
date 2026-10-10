from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
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
    dst_ptr = kv_buffer_ptr + safe_loc * buffer_stride + offs

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
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < nope_dim

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    src = tl.load(
        cache_k_nope_ptr + pid_loc * nope_stride + offs,
        mask=mask,
    )
    tl.store(dst_ptr, src, mask=mask)

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
    rows = kv_buffer.view(kv_buffer.shape[0], -1)
    nope = cache_k_nope.reshape(cache_k_nope.shape[0], -1)
    nope_dim = nope.shape[-1]
    has_rope = cache_k_rope is not None and cache_k_rope.numel() > 0

    loc = loc.to(torch.int64)
    if has_rope:
        is_valid = loc != reserved_skip_index
    else:
        # set_mla_kv_buffer_kernel_norope takes no skip index and writes every
        # loc; keep the two paths bit-identical rather than fixing it here.
        is_valid = torch.ones_like(loc, dtype=torch.bool)
    if dcp_world_size > 1:
        is_valid &= loc % dcp_world_size == dcp_rank
    dst = loc[is_valid] // dcp_world_size

    rows[dst, :nope_dim] = nope[is_valid]
    if has_rope:
        rope = cache_k_rope.reshape(cache_k_rope.shape[0], -1)
        rows[dst, nope_dim : nope_dim + rope.shape[-1]] = rope[is_valid]


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
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            nope_dim,
            BLOCK=BLOCK,
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
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
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
    is_valid = loc != reserved_skip_index
    safe_loc = tl.where(is_valid, loc, 0)
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
):
    """Fuse BF16/FP16 MLA K quantization with paged KV write.

    Writes targeting ``reserved_skip_index`` are skipped. Pass -1 to disable.
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


@triton.jit
def _unpack_dcp_kv_kernel(
    gathered,
    metadata,
    out_k,
    out_pe,
    rank_rows: tl.constexpr,
    world_size: tl.constexpr,
    k_dim: tl.constexpr,
    pe_dim: tl.constexpr,
    k_stride: tl.constexpr,
    pe_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    padded_start=0,
    start=0,
    length=0,
    output_start=0,
):
    if metadata is not None:
        req = tl.program_id(1)
        padded_start = tl.load(metadata + req * 4)
        start = tl.load(metadata + req * 4 + 1)
        length = tl.load(metadata + req * 4 + 2)
        output_start = tl.load(metadata + req * 4 + 3)
    dim = k_dim + pe_dim
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offsets // dim, offsets % dim
    position = start + token
    source_row = (
        position % world_size * rank_rows + padded_start + position // world_size
    )
    value = tl.load(
        gathered + source_row.to(tl.int64) * dim + feature,
        mask=token < length,
        other=0.0,
    )
    output_row = (output_start + token).to(tl.int64)
    tl.store(
        out_k + output_row * k_stride + feature,
        value,
        mask=(token < length) & (feature < k_dim),
    )
    tl.store(
        out_pe + output_row * pe_stride + feature - k_dim,
        value,
        mask=(token < length) & (feature >= k_dim),
    )


def unpack_dcp_kv(
    gathered: torch.Tensor,
    metadata: torch.Tensor | tuple[int, int, int, int],
    out_k: torch.Tensor,
    out_pe: torch.Tensor,
    world_size: int,
    max_prefix_len: int,
) -> None:
    """Write prefixes, leaving suffix slots untouched.

    Each metadata row is (padded_start, start % world_size, prefix_len,
    output_start); a single request can pass that tuple directly. Outputs may
    be separate tensors or strided latent/rope views of a combined MLA buffer.
    """
    if max_prefix_len == 0:
        return
    scalar_metadata = (0, 0, 0, 0)
    num_requests = 1 if isinstance(metadata, tuple) else metadata.shape[0]
    if isinstance(metadata, tuple):
        scalar_metadata, metadata = metadata, None
    elif metadata.device.type == "cpu" and num_requests == 1:
        # A single request needs no metadata upload or device allocation.
        scalar_metadata = metadata[0].tolist()
        metadata = None
    else:
        metadata = metadata.to(device=gathered.device, non_blocking=True)
    dim = out_k.shape[-1] + out_pe.shape[-1]
    _unpack_dcp_kv_kernel[(triton.cdiv(max_prefix_len * dim, 2048), num_requests)](
        gathered,
        metadata,
        out_k,
        out_pe,
        gathered.shape[0] // world_size,
        world_size,
        out_k.shape[-1],
        out_pe.shape[-1],
        out_k.stride(0),
        out_pe.stride(0),
        BLOCK=2048,
        padded_start=scalar_metadata[0],
        start=scalar_metadata[1],
        length=scalar_metadata[2],
        output_start=scalar_metadata[3],
    )
