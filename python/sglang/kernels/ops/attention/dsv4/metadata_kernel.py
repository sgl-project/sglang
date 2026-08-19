from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["bs", "num_write_tokens", "c128_cur_max_seq_len"])
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
    num_write_tokens,
    max_pages,
    c128_cur_max_seq_len,
    c128_page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_PAGE_INDICES: tl.constexpr,
    LIVE_PREFIX_ONLY: tl.constexpr,
):
    batch_id = tl.program_id(0)
    column_block = tl.program_id(1)
    if batch_id >= bs:
        return

    seq_len = tl.load(seq_lens_ptr + batch_id)
    c4_seq_lens_raw = seq_len // 4
    c128_seq_lens_raw = seq_len // 128

    # Full-tail launches one column program per row; live-prefix launches a
    # graph-static column grid whose first program exclusively owns scalars.
    if (not LIVE_PREFIX_ONLY) or column_block == 0:
        position = tl.load(positions_ptr + batch_id)
        is_write_token = batch_id < num_write_tokens
        raw_out_loc = tl.load(raw_out_loc_ptr + batch_id, mask=is_write_token, other=0)

        c4_should_compress = (seq_len % 4) == 0
        c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
        c4_positions = position & (~3)
        c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

        tl.store(c4_out_loc_ptr + batch_id, c4_out_loc, mask=is_write_token)
        tl.store(c4_positions_ptr + batch_id, c4_positions)
        tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
        tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

        c128_should_compress = (seq_len % 128) == 0
        c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
        c128_positions = position & (~127)
        c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

        tl.store(c128_out_loc_ptr + batch_id, c128_out_loc, mask=is_write_token)
        tl.store(c128_positions_ptr + batch_id, c128_positions)
        tl.store(c128_seq_lens_raw_ptr + batch_id, c128_seq_lens_raw)
        tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        block_begin = 0
        block_end = c128_cur_max_seq_len
        write_bound = c128_cur_max_seq_len
        if LIVE_PREFIX_ONLY:
            block_begin = column_block * BLOCK_SIZE
            # FlashMLA clamps C128 lengths to one. Preserve one explicit
            # sentinel for empty/negative raw lengths and cap live writes to
            # output capacity.
            live_bound = tl.maximum(c128_seq_lens_raw, 1)
            write_bound = tl.minimum(live_bound, c128_cur_max_seq_len)
            block_end = tl.minimum(block_begin + BLOCK_SIZE, write_bound)

        if block_begin >= write_bound:
            return

        page_indices_base = batch_id * c128_cur_max_seq_len
        for block_start in tl.range(block_begin, block_end, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < write_bound
            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            valid_mask = mask & (offsets < c128_seq_lens_raw)
            page_mask = valid_mask & (page_idx < max_pages)
            page_table_vals = tl.load(
                page_table_ptr + batch_id * max_pages + page_idx,
                mask=page_mask,
                other=0,
            )
            c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page
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
    live_prefix_only: bool = False,
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
    # CP-v2 may add padding rows to the attention metadata, but those rows have
    # no cache-write locations. Keep the write buffers unpadded and mask those
    # rows in the kernel.
    num_write_tokens = raw_out_loc.shape[0]
    assert (
        num_write_tokens <= bs
    ), f"raw_out_loc has {num_write_tokens} rows, expected at most {bs} metadata rows"
    device = seq_lens.device

    c4_out_loc = torch.empty(num_write_tokens, dtype=torch.int64, device=device)
    c4_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    c128_out_loc = torch.empty(num_write_tokens, dtype=torch.int64, device=device)
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

    args = (
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
        num_write_tokens,
        max_pages,
        c128_cur_max_seq_len,
        c128_page_size,
    )
    if live_prefix_only:
        BLOCK_SIZE = 256
        num_column_blocks = max(triton.cdiv(c128_cur_max_seq_len, BLOCK_SIZE), 1)
    else:
        num_column_blocks = 1
    _init_compressed_attn_metadata_kernel[(bs, num_column_blocks)](
        *args,
        BLOCK_SIZE=BLOCK_SIZE,
        COMPUTE_PAGE_INDICES=compute_page_indices,
        LIVE_PREFIX_ONLY=live_prefix_only,
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
    live_prefix_only: bool = False,
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
    """Build compressed-attention metadata.

    ``live_prefix_only`` is an internal CUDA-graph optimization. When enabled,
    only each row's advertised C128 prefix is refreshed; the remaining
    capacity is undefined. Native FlashMLA predicates KV address generation by
    the matching C128 length, so it may vector-load but cannot consume this
    suffix. The default initializes the full output exactly as before.
    """
    return _init_compressed_attn_metadata_triton(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
        live_prefix_only,
    )
