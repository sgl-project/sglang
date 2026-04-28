"""
Triton kernels for fused compressed metadata initialization.

These kernels replace the fragmented tensor operations in DSV4AttnMetadataRadix.init_compressed_metadata,
reducing kernel launch overhead from 10+ launches to 1.

Set environment variable SGLANG_USE_TORCH_COMPRESS_METADATA=1 to use the original PyTorch implementation
instead of the Triton kernel (useful for debugging or compatibility).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

# Environment variable to control implementation dispatch
# Set to "1" to use PyTorch implementation, otherwise use Triton
_USE_TRITON_IMPL = envs.SGLANG_OPT_USE_TRITON_CA_METADATA.get()


# =============================================================================
# Triton Implementation
# =============================================================================


@triton.jit
def _init_compressed_attn_metadata_kernel(
    # Inputs
    seq_lens_ptr,
    positions_ptr,
    raw_out_loc_ptr,
    page_table_ptr,  # Only used when COMPUTE_PAGE_INDICES=True
    # Outputs (C4)
    c4_out_loc_ptr,
    c4_positions_ptr,
    c4_seq_lens_raw_ptr,
    c4_seq_lens_clamp1_ptr,
    # Outputs (C128)
    c128_out_loc_ptr,
    c128_positions_ptr,
    c128_seq_lens_clamp1_ptr,
    c128_page_indices_ptr,  # Only used when COMPUTE_PAGE_INDICES=True
    # Meta
    bs,
    max_pages,
    page_size: tl.constexpr,
    c128_max_seq_len: tl.constexpr,
    c128_page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_PAGE_INDICES: tl.constexpr,
):
    """
    Unified fused kernel for both C4 and C128 compressed metadata computation.

    This kernel computes metadata for both compression ratios (4 and 128) in a single launch.

    For ratio=4:
        should_compress = seq_lens % 4 == 0
        c4_out_loc = (raw_out_loc // 4) if should_compress else 0
        c4_positions = (positions // 4) * 4
        c4_seq_lens_raw = seq_lens // 4
        c4_seq_lens_clamp1 = max(c4_seq_lens_raw, 1)

    For ratio=128:
        should_compress = seq_lens % 128 == 0
        c128_out_loc = (raw_out_loc // 128) if should_compress else 0
        c128_positions = (positions // 128) * 128
        c128_seq_lens_raw = seq_lens // 128
        c128_seq_lens_clamp1 = max(c128_seq_lens_raw, 1)
        c128_page_indices[pos] = page_table[pos // c_page_size] * c_page_size + (pos % c_page_size)
                                 if pos < c128_seq_lens_raw else -1
    """
    batch_id = tl.program_id(0)
    if batch_id >= bs:
        return

    # Load inputs for this batch element
    seq_len = tl.load(seq_lens_ptr + batch_id)
    position = tl.load(positions_ptr + batch_id)
    raw_out_loc = tl.load(raw_out_loc_ptr + batch_id)

    # ========== C4 Metadata Computation ==========
    # Compute compressed metadata for ratio=4
    c4_should_compress = (seq_len % 4) == 0
    c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
    # Use bit masking for efficiency: positions & ~3 == (positions // 4) * 4
    c4_positions = position & (~3)
    c4_seq_lens_raw = seq_len // 4
    c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

    # Store C4 outputs
    tl.store(c4_out_loc_ptr + batch_id, c4_out_loc)
    tl.store(c4_positions_ptr + batch_id, c4_positions)
    tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
    tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

    # ========== C128 Metadata Computation ==========
    # Compute compressed metadata for ratio=128
    c128_should_compress = (seq_len % 128) == 0
    c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
    # Use bit masking: positions & ~127 == (positions // 128) * 128
    c128_positions = position & (~127)
    c128_seq_lens_raw = seq_len // 128
    c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

    # Store C128 scalar outputs
    tl.store(c128_out_loc_ptr + batch_id, c128_out_loc)
    tl.store(c128_positions_ptr + batch_id, c128_positions)
    tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    # ========== C128 Page Indices Computation (conditional) ==========
    if COMPUTE_PAGE_INDICES:
        # Compute page_indices for this batch element
        # Process in blocks for efficiency
        page_indices_base = batch_id * c128_max_seq_len
        for block_start in range(0, c128_max_seq_len, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < c128_max_seq_len

            # Compute page index and offset within page
            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            # Load page table entries (with bounds check)
            page_mask = mask & (page_idx < max_pages)
            page_table_vals = tl.load(
                page_table_ptr + batch_id * max_pages + page_idx,
                mask=page_mask,
                other=0,
            )

            # Compute c_page_indices = page_table[page_idx] * c_page_size + offset_in_page
            c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page

            # Mask out positions >= c128_seq_lens_raw with -1
            valid_mask = offsets < c128_seq_lens_raw
            c_page_indices_vals = tl.where(valid_mask, c_page_indices_vals, -1)

            # Store page indices
            tl.store(
                c128_page_indices_ptr + page_indices_base + offsets,
                c_page_indices_vals,
                mask=mask,
            )


def _init_compressed_attn_metadata_triton(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # C4 outputs
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],  # C128 outputs
]:
    """Triton implementation of compressed metadata computation."""
    bs = seq_lens.shape[0]
    device = seq_lens.device

    # Allocate C4 output tensors
    c4_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c4_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    # Allocate C128 output tensors
    c128_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c128_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c128_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    # Handle page indices computation
    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert page_size > 0, "page_size required when compute_page_indices=True"
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages
        c128_page_indices = torch.empty(
            bs, c128_max_seq_len, dtype=torch.int32, device=device
        )
        BLOCK_SIZE = triton.next_power_of_2(max(c128_page_size, 64))
    else:
        max_pages = 0
        c128_page_size = 1
        c128_max_seq_len = 0
        c128_page_indices = None
        BLOCK_SIZE = 64
        # Create dummy page_table pointer if not provided
        if page_table is None:
            page_table = torch.empty(0, dtype=torch.int32, device=device)

    # Launch unified kernel
    grid = (bs,)
    _init_compressed_attn_metadata_kernel[grid](
        # Inputs
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        # C4 outputs
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        # C128 outputs
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        (
            c128_page_indices
            if c128_page_indices is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        # Meta
        bs,
        max_pages,
        page_size if page_size > 0 else 128,  # Default to avoid division by zero
        c128_max_seq_len,
        c128_page_size,
        BLOCK_SIZE,
        compute_page_indices,
    )

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


# =============================================================================
# PyTorch Reference Implementation (Original)
# =============================================================================


def _init_compressed_attn_metadata_torch(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # C4 outputs
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],  # C128 outputs
]:
    """
    Original PyTorch implementation of compressed metadata computation.

    This is the reference implementation that the Triton kernel replaces.
    It's kept for debugging and compatibility purposes.
    """
    cuda_int32_kwargs = {"device": seq_lens.device, "dtype": torch.int32}

    # ========== C4 Metadata Computation ==========
    c4_should_compress = seq_lens % 4 == 0
    c4_seq_lens_raw = seq_lens // 4
    c4_positions = positions // 4 * 4
    c4_out_loc = torch.where(c4_should_compress, raw_out_loc // 4, 0)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    # ========== C128 Metadata Computation ==========
    c128_should_compress = seq_lens % 128 == 0
    c128_seq_lens_raw = seq_lens // 128
    c128_positions = positions // 128 * 128
    c128_out_loc = torch.where(c128_should_compress, raw_out_loc // 128, 0)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    # ========== C128 Page Indices Computation ==========
    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert page_size > 0, "page_size required when compute_page_indices=True"

        c128_page_size = page_size // 128
        max_pages = page_table.size(1)
        c128_max_seq_len = c128_page_size * max_pages

        # [bs, max_pages] -> [bs, max_pages, c_page_size] -> [bs, c_max_seq_len]
        c_offsets = torch.arange(c128_max_seq_len, **cuda_int32_kwargs)
        c128_page_indices = (
            (page_table.unsqueeze(2) * c128_page_size + c_offsets[:c128_page_size])
            .to(torch.int32)
            .contiguous()
            .view(-1, c128_max_seq_len)
        )
        # Mask out positions >= c128_seq_lens_raw with -1
        mask = c_offsets.unsqueeze(0) >= c128_seq_lens_raw.unsqueeze(1)
        c128_page_indices.masked_fill_(mask, -1)
    else:
        c128_page_indices = None

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


# =============================================================================
# Public API (dispatches based on environment variable)
# =============================================================================


def init_compressed_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # C4 outputs
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],  # C128 outputs
]:
    """
    Unified function for compressed metadata computation.

    Computes both C4 and C128 metadata. Uses Triton kernel by default for
    better performance. Set SGLANG_OPT_USE_TRITON_COMPRESS_METADATA=1 to use
    the original PyTorch implementation instead.

    Args:
        seq_lens: [bs] int32, sequence lengths
        positions: [bs] int32, positions (expanded causally)
        raw_out_loc: [bs] int32, raw output locations
        page_table: [bs, max_pages] int32, page table (required if compute_page_indices=True)
        page_size: int, page size (required if compute_page_indices=True)
        compute_page_indices: bool, whether to compute c128_page_indices

    Returns:
        Tuple of:
            - c4_out_loc: [bs] int32
            - c4_positions: [bs] int32
            - c4_topk_lengths_raw: [bs] int32
            - c4_topk_lengths_clamp1: [bs] int32
            - c128_out_loc: [bs] int32
            - c128_positions: [bs] int32
            - c128_topk_lengths_clamp1: [bs] int32
            - c128_page_indices: [bs, c_max_seq_len] int32 or None
    """
    if not _USE_TRITON_IMPL:
        return _init_compressed_attn_metadata_torch(
            seq_lens,
            positions,
            raw_out_loc,
            page_table,
            page_size,
            compute_page_indices,
        )
    else:
        return _init_compressed_attn_metadata_triton(
            seq_lens,
            positions,
            raw_out_loc,
            page_table,
            page_size,
            compute_page_indices,
        )


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================


def init_c4_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward compatible wrapper for C4-only metadata computation.

    Note: This still computes C128 metadata internally but discards it.
    For better performance, use init_compressed_metadata() directly.
    """
    (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        _,
        _,
        _,
        _,
    ) = init_compressed_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        page_table=None,
        page_size=0,
        compute_page_indices=False,
    )
    return c4_out_loc, c4_positions, c4_seq_lens_raw, c4_seq_lens_clamp1


def init_c128_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward compatible wrapper for C128-only metadata computation.

    Note: This still computes C4 metadata internally but discards it.
    For better performance, use init_compressed_metadata() directly.
    """
    (
        _,
        _,
        _,
        _,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    ) = init_compressed_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        page_table=page_table,
        page_size=page_size,
        compute_page_indices=True,
    )
    return c128_out_loc, c128_positions, c128_seq_lens_clamp1, c128_page_indices
