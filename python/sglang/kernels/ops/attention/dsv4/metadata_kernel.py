from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["bs", "c128_cur_max_seq_len"])
def _init_compressed_attn_metadata_kernel(
    seq_lens_ptr,
    positions_ptr,
    raw_out_loc_ptr,
    page_table_ptr,
    c4_out_loc_ptr,
    c4_positions_ptr,
    c4_seq_lens_raw_ptr,
    c4_seq_lens_clamp1_ptr,
    c128_out_loc_ptr,
    c128_positions_ptr,
    c128_seq_lens_raw_ptr,
    c128_seq_lens_clamp1_ptr,
    c128_page_indices_ptr,
    bs,
    max_pages,
    c128_cur_max_seq_len,
    c128_page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_PAGE_INDICES: tl.constexpr,
):
    batch_id = tl.program_id(0)
    if batch_id >= bs:
        return

    seq_len = tl.load(seq_lens_ptr + batch_id)
    position = tl.load(positions_ptr + batch_id)
    raw_out_loc = tl.load(raw_out_loc_ptr + batch_id)

    c4_should_compress = (seq_len % 4) == 0
    c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
    c4_positions = position & (~3)
    c4_seq_lens_raw = seq_len // 4
    c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

    tl.store(c4_out_loc_ptr + batch_id, c4_out_loc)
    tl.store(c4_positions_ptr + batch_id, c4_positions)
    tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
    tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

    c128_should_compress = (seq_len % 128) == 0
    c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
    c128_positions = position & (~127)
    c128_seq_lens_raw = seq_len // 128
    c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

    tl.store(c128_out_loc_ptr + batch_id, c128_out_loc)
    tl.store(c128_positions_ptr + batch_id, c128_positions)
    tl.store(c128_seq_lens_raw_ptr + batch_id, c128_seq_lens_raw)
    tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        page_indices_base = batch_id * c128_cur_max_seq_len
        for block_start in tl.range(0, c128_cur_max_seq_len, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < c128_cur_max_seq_len

            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            page_mask = mask & (page_idx < max_pages)
            page_table_vals = tl.load(
                page_table_ptr + batch_id * max_pages + page_idx,
                mask=page_mask,
                other=0,
            )

            c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page

            valid_mask = offsets < c128_seq_lens_raw
            c_page_indices_vals = tl.where(valid_mask, c_page_indices_vals, -1)

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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    c4_out_loc = torch.empty(bs, dtype=torch.int64, device=device)
    c4_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    c128_out_loc = torch.empty(bs, dtype=torch.int64, device=device)
    c128_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c128_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c128_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert (
            page_size >= 128 and page_size % 128 == 0
        ), "page_size must be a multiple of 128 when compute_page_indices=True"
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_cur_max_seq_len = c128_page_size * max_pages
        c128_page_indices = torch.empty(
            bs, c128_cur_max_seq_len, dtype=torch.int32, device=device
        )
        BLOCK_SIZE = triton.next_power_of_2(max(c128_page_size, 64))
    else:
        max_pages = 0
        c128_page_size = 1
        c128_cur_max_seq_len = 0
        c128_page_indices = None
        BLOCK_SIZE = 64
        if page_table is None:
            page_table = torch.empty(0, dtype=torch.int32, device=device)

    grid = (bs,)
    _init_compressed_attn_metadata_kernel[grid](
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        (
            c128_page_indices
            if c128_page_indices is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        bs,
        max_pages,
        c128_cur_max_seq_len,
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
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def init_compression_metadata(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    return _init_compressed_attn_metadata_triton(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )


def _init_compression_attn_metadata_pytorch(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    # --- c4 (compress-by-4) metadata ---
    c4_should_compress = (seq_lens % 4) == 0
    c4_out_loc = torch.where(
        c4_should_compress, raw_out_loc // 4, torch.zeros_like(raw_out_loc)
    )
    c4_positions = (positions & (~3)).to(torch.int32)
    c4_seq_lens_raw = (seq_lens // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    # --- c128 (compress-by-128) metadata ---
    c128_should_compress = (seq_lens % 128) == 0
    c128_out_loc = torch.where(
        c128_should_compress, raw_out_loc // 128, torch.zeros_like(raw_out_loc)
    )
    c128_positions = (positions & (~127)).to(torch.int32)
    c128_seq_lens_raw = (seq_lens // 128).to(torch.int32)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    # --- page indices for c128 ---
    c128_page_indices: Optional[torch.Tensor] = None
    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert page_size > 0, "page_size required when compute_page_indices=True"

        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages

        # offsets: [c128_max_seq_len]
        offsets = torch.arange(c128_max_seq_len, device=device, dtype=torch.int32)

        # page_idx and offset_in_page for every position
        page_idx = offsets // c128_page_size  # [c128_max_seq_len]
        offset_in_page = offsets % c128_page_size  # [c128_max_seq_len]

        # Clamp page_idx for safe gather (values beyond max_pages will be masked later)
        page_idx_clamped = torch.clamp(page_idx, max=max_pages - 1)

        # Gather page table values for all batches: page_table is [bs, max_pages]
        # page_idx_clamped is [c128_max_seq_len], expand to [bs, c128_max_seq_len]
        page_idx_expanded = page_idx_clamped.unsqueeze(0).expand(bs, -1)
        page_table_vals = torch.gather(
            page_table, dim=1, index=page_idx_expanded.to(torch.int64)
        ).to(
            torch.int32
        )  # [bs, c128_max_seq_len]

        # c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page
        c128_page_indices_vals = (
            page_table_vals * c128_page_size + offset_in_page.unsqueeze(0)
        )

        # Mask: set to -1 where offsets >= c128_seq_lens_raw per batch
        # c128_seq_lens_raw: [bs], offsets: [c128_max_seq_len]
        valid_mask = offsets.unsqueeze(0) < c128_seq_lens_raw.unsqueeze(
            1
        )  # [bs, c128_max_seq_len]
        c128_page_indices = torch.where(
            valid_mask,
            c128_page_indices_vals,
            torch.tensor(-1, dtype=torch.int32, device=device),
        )

    return (
        c4_out_loc.to(torch.int32),
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc.to(torch.int32),
        c128_positions,
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def init_compression_metadata_torch(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Initialize compressed attention metadata using pure PyTorch operations.
    Computes compress-by-4 and compress-by-128 metadata for attention,
    and optionally computes page indices for paged KV cache.
    Args:
        seq_lens: Sequence lengths per batch element, shape [bs].
        positions: Current token positions per batch element, shape [bs].
        raw_out_loc: Raw output locations per batch element, shape [bs].
        page_table: Page table mapping for paged KV cache, shape [bs, max_pages].
            Required when compute_page_indices is True.
        page_size: Size of each page in the KV cache. Required when
            compute_page_indices is True.
        compute_page_indices: Whether to compute c128 page indices.
    Returns:
        A tuple of:
        - c4_out_loc: Compressed-by-4 output locations, shape [bs].
        - c4_positions: Compressed-by-4 positions (aligned to 4), shape [bs].
        - c4_seq_lens_raw: Raw compressed-by-4 sequence lengths, shape [bs].
        - c4_seq_lens_clamp1: Compressed-by-4 sequence lengths clamped to min 1, shape [bs].
        - c128_out_loc: Compressed-by-128 output locations, shape [bs].
        - c128_positions: Compressed-by-128 positions (aligned to 128), shape [bs].
        - c128_seq_lens_clamp1: Compressed-by-128 sequence lengths clamped to min 1, shape [bs].
        - c128_page_indices: Page indices for c128 compression, shape [bs, c128_max_seq_len],
            or None if compute_page_indices is False.
    """
    return _init_compression_attn_metadata_pytorch(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )
