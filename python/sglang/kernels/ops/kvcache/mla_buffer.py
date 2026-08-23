from __future__ import annotations

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
    is_valid = loc % DCP_WORLD_SIZE == DCP_RANK
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


# Above this loc count the TMA bulk-store path overtakes the single-CTA-per-loc
# Triton kernel. Below it, Triton with BLOCK = next_pow2(total_dim) (one CTA
# does the whole row in one tile, no boundary fan-out) is the winning fallback.
# Tuned on GB300 with DSv4 row widths.
_TMA_BULK_STORE_MIN_LOCS = 768


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    """Dispatch MLA paged-KV scatter writes to the fastest available path.

    Two paths, chosen on ``n_loc``:

    - ``n_loc >= 768`` (and SM90+ with TMA-compatible row widths): JIT CUDA
      kernel where each warp loads one (nope, rope) row into shared memory and
      issues a single ``cp.async.bulk.global.shared::cta`` store to scatter the
      row at ``kv_buffer[loc[item]]``. Wins at large bs because it packs 4-8
      items per CTA, drastically reducing the CTA count vs single-CTA-per-loc.
    - Otherwise: Triton kernel with ``BLOCK = next_pow2(nope_dim + rope_dim)``,
      i.e. one CTA per loc covering the entire row in one tile. Wins at small
      bs because there's no per-loc CTA fan-out (5x fewer CTAs than the old
      BLOCK=128 dispatch) and the row-spanning block makes the boundary branch
      a one-shot per CTA. This is also the path for SM<90 and for shapes that
      violate the TMA 16-byte alignment.

    Speedup vs the legacy BLOCK=128 Triton kernel on GB300 (BF16, nope=512,
    rope=64): ~1.05x at bs=8, ~1.5x at bs=128, 3.5x at bs=512, **11.7x at
    bs=16384**.

    Name retained for caller compatibility; the implementation is no longer
    Triton-only.
    """
    from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
        can_use_set_mla_kv_buffer,
    )
    from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
        set_mla_kv_buffer as jit_set_mla_kv_buffer,
    )

    n_loc = loc.numel()
    nope_bytes = cache_k_nope.shape[-1] * cache_k_nope.element_size()
    rope_bytes = cache_k_rope.shape[-1] * cache_k_rope.element_size()
    if (
        n_loc >= _TMA_BULK_STORE_MIN_LOCS
        and is_arch_support_pdl()
        and can_use_set_mla_kv_buffer(nope_bytes, rope_bytes)
        and not get_parallel().dcp_enabled
    ):
        jit_set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return

    # Fallback: Triton with BLOCK = next_pow2(total_dim). One CTA per loc; the
    # whole row in one tile (the existing 3-way nope/rope/boundary branch in
    # ``set_mla_kv_buffer_kernel`` handles the over-allocation past total_dim
    # via the offs<total_dim mask). Beats BLOCK=128 by 60-2700 ns across the
    # 2 <= bs <= 512 range on GB300.
    nope_dim = cache_k_nope.shape[-1]
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
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
        DCP_RANK=get_parallel().attn_dcp_rank,
        DCP_WORLD_SIZE=get_parallel().attn_dcp_size,
        **pdl_kwargs,
    )


@triton.jit
def set_mla_kv_buffer_fp8_quant_kernel(
    kv_buffer_fp8_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
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
    dst_ptr = kv_buffer_fp8_ptr + loc * buffer_stride + offs

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
    tl.store(dst_ptr, src, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def set_mla_kv_buffer_triton_fp8_quant(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    fp8_dtype: torch.dtype,
):
    """Fuse BF16/FP16 MLA K quantization with paged KV write."""
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
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

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
    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_scale_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
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


def get_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    nope_dim = cache_k_nope.shape[-1]  # 512
    rope_dim = cache_k_rope.shape[-1]  # 64
    n_loc = loc.numel()
    grid = (n_loc,)

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
def dequantize_mla_fp8_page_table_kernel(
    src_ptr,
    dst_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    page_epochs_ptr,
    epoch_ptr,
    src_row_stride: tl.constexpr,
    dst_row_stride: tl.constexpr,
    page_table_row_stride: tl.constexpr,
    row_width: tl.constexpr,
    PROGRAMS_PER_SEQUENCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy referenced FP8 MLA pages to their original ids in a dense shadow."""
    seq_idx = tl.program_id(0)
    page_offset = tl.program_id(1)
    cache_seqlen = tl.load(cache_seqlens_ptr + seq_idx).to(tl.int64)
    epoch = tl.load(epoch_ptr)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < row_width

    while page_offset < cache_seqlen:
        page_id = tl.load(
            page_table_ptr + seq_idx * page_table_row_stride + page_offset
        ).to(tl.int64)
        previous_epoch = tl.atomic_xchg(page_epochs_ptr + page_id, epoch)
        if previous_epoch != epoch:
            values = tl.load(
                src_ptr + page_id * src_row_stride + col_offsets,
                mask=col_mask,
            )
            tl.store(
                dst_ptr + page_id * dst_row_stride + col_offsets,
                values,
                mask=col_mask,
            )
        page_offset += PROGRAMS_PER_SEQUENCE


def dequantize_mla_fp8_page_table(
    src: torch.Tensor,
    dst: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_epochs: torch.Tensor,
    epoch: torch.Tensor,
    max_programs: int = 1024,
) -> None:
    """Dequantize only MLA KV pages referenced by ``page_table``.

    The destination keeps the source page ids, so FA3 can consume it with the
    original page table. ``page_epochs`` deduplicates pages shared by requests
    without allocating or clearing temporary state on every invocation. Both
    it and ``epoch`` must be persistent tensors to keep this CUDA-graph safe.

    This first implementation intentionally supports only page size 1. The
    caller owns that policy check; tensors here are viewed as dense rows.
    """
    if src.ndim < 2 or dst.shape != src.shape:
        raise ValueError(
            f"Expected matching row-major MLA buffers, got src.shape={src.shape!r} "
            f"and dst.shape={dst.shape!r}"
        )
    if page_table.ndim != 2:
        raise ValueError(
            f"Expected a 2D page table, got page_table.shape={page_table.shape!r}"
        )
    if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != page_table.shape[0]:
        raise ValueError(
            "cache_seqlens must have one entry for every page-table row, got "
            f"cache_seqlens.shape={cache_seqlens.shape!r} and "
            f"page_table.shape={page_table.shape!r}"
        )
    if page_epochs.numel() != src.shape[0] or epoch.numel() != 1:
        raise ValueError(
            "Epoch state does not match the MLA buffer: "
            f"page_epochs.shape={page_epochs.shape!r}, epoch.shape={epoch.shape!r}, "
            f"pages={src.shape[0]}"
        )

    if page_table.numel() == 0:
        return

    row_width = src.numel() // src.shape[0]
    block_size = triton.next_power_of_2(row_width)
    programs_per_sequence = min(
        max(1, max_programs // page_table.shape[0]), page_table.shape[1]
    )

    epoch.add_(1)
    grid = (page_table.shape[0], programs_per_sequence)
    dequantize_mla_fp8_page_table_kernel[grid](
        src,
        dst,
        page_table,
        cache_seqlens,
        page_epochs,
        epoch,
        src.stride(0),
        dst.stride(0),
        page_table.stride(0),
        row_width,
        PROGRAMS_PER_SEQUENCE=programs_per_sequence,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


_SUPPORTED_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16, torch.float16)


def is_fa3_mla_fp8_shadow_enabled(
    *,
    fa_impl_ver: int,
    use_mla: bool,
    page_size: int,
    unified_dense: bool,
    attn_cp_size: int,
    is_draft_runner: bool,
    dcp_enabled: bool,
    dsa_kv_cache_store_fp8: bool,
) -> bool:
    """Return whether a backend configuration can use the shared shadow."""
    return (
        fa_impl_ver == 3
        and use_mla
        and page_size == 1
        and not unified_dense
        and attn_cp_size == 1
        and not is_draft_runner
        and not dcp_enabled
        and not dsa_kv_cache_store_fp8
    )


class FA3MLAFP8KVShadow:
    """Persistent page-id-preserving shadow used by FA3 absorbed MLA.

    One instance is shared by all MLA layers. Layers execute serially on the
    current CUDA stream, so only the pages needed by the current layer have to
    be materialized before FA3 consumes the buffer.
    """

    def __init__(self, source: torch.Tensor, output_dtype: torch.dtype):
        self.buffer = torch.empty_like(source, dtype=output_dtype)
        self.page_epochs = torch.zeros(
            source.shape[0], dtype=torch.int32, device=source.device
        )
        self.epoch = torch.zeros((), dtype=torch.int32, device=source.device)
        self.source_shape = source.shape
        self.source_dtype = source.dtype
        self.output_dtype = output_dtype

    @classmethod
    def maybe_create(
        cls, source: torch.Tensor, output_dtype: torch.dtype
    ) -> FA3MLAFP8KVShadow | None:
        if not cls.is_supported_source(source, output_dtype):
            return None
        return cls(source, output_dtype)

    @staticmethod
    def is_supported_source(source: torch.Tensor, output_dtype: torch.dtype) -> bool:
        return source.dtype in _SUPPORTED_FP8_DTYPES and (
            output_dtype in _SUPPORTED_OUTPUT_DTYPES
            and source.ndim >= 2
            and source.is_cuda
            and source.is_contiguous()
        )

    def can_materialize(self, source: torch.Tensor, output_dtype: torch.dtype) -> bool:
        return (
            source.shape == self.source_shape
            and source.dtype == self.source_dtype
            and output_dtype == self.output_dtype
            and source.device == self.buffer.device
            and source.is_cuda
            and source.is_contiguous()
        )

    def materialize(
        self,
        source: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        dequantize_mla_fp8_page_table(
            source,
            self.buffer,
            page_table,
            cache_seqlens,
            self.page_epochs,
            self.epoch,
        )
        return self.buffer
