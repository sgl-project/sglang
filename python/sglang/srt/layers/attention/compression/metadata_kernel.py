from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
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
    c128_seq_lens_clamp1_ptr,
    c128_page_indices_ptr,
    bs,
    max_pages,
    page_size: tl.constexpr,
    c128_max_seq_len: tl.constexpr,
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
    tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        page_indices_base = batch_id * c128_max_seq_len
        for block_start in range(0, c128_max_seq_len, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < c128_max_seq_len

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
    Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    c4_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c4_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    c128_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c128_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c128_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

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
        c128_seq_lens_clamp1,
        (
            c128_page_indices
            if c128_page_indices is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        bs,
        max_pages,
        page_size if page_size > 0 else 128,
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
