from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_PAGE_INDEX_ALIGNMENT = 64


@triton.jit(
    do_not_specialize=[
        "bs",
        "c128_cur_max_seq_len",
        "c128_storage_max_seq_len",
    ]
)
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
    c128_storage_max_seq_len,
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
        raw_out_loc = tl.load(raw_out_loc_ptr + batch_id)

        c4_should_compress = (seq_len % 4) == 0
        c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
        c4_positions = position & (~3)
        c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

        tl.store(c4_out_loc_ptr + batch_id, c4_out_loc)
        tl.store(c4_positions_ptr + batch_id, c4_positions)
        tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
        tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

        c128_should_compress = (seq_len % 128) == 0
        c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
        c128_positions = position & (~127)
        c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

        tl.store(c128_out_loc_ptr + batch_id, c128_out_loc)
        tl.store(c128_positions_ptr + batch_id, c128_positions)
        tl.store(c128_seq_lens_raw_ptr + batch_id, c128_seq_lens_raw)
        tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        block_begin = 0
        block_end = c128_storage_max_seq_len
        write_bound = c128_storage_max_seq_len
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

        page_indices_base = batch_id * c128_storage_max_seq_len
        for block_start in tl.range(block_begin, block_end, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < write_bound
            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            valid_mask = (
                mask & (offsets < c128_seq_lens_raw) & (offsets < c128_cur_max_seq_len)
            )
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
        c128_storage_max_seq_len = (
            triton.cdiv(c128_cur_max_seq_len, _PAGE_INDEX_ALIGNMENT)
            * _PAGE_INDEX_ALIGNMENT
        )
        c128_page_indices = torch.empty(
            bs, c128_storage_max_seq_len, dtype=torch.int32, device=device
        )
        BLOCK_SIZE = triton.next_power_of_2(max(c128_page_size, 64))
    else:
        max_pages = 0
        c128_page_size = 1
        c128_cur_max_seq_len = 0
        c128_storage_max_seq_len = 0
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
        max_pages,
        c128_cur_max_seq_len,
        c128_storage_max_seq_len,
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

    The returned C128 page-index rows are the final, 64-element-aligned storage
    consumed by FlashMLA. ``live_prefix_only`` is an internal CUDA-graph
    optimization. When enabled, only each row's advertised C128 prefix is
    refreshed; the remaining logical capacity and alignment padding are
    undefined. Native FlashMLA predicates KV address generation by the matching
    C128 length, so it may vector-load but cannot consume this suffix. The
    default initializes the full logical suffix and alignment padding to -1.
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


@triton.jit
def _init_c4_sparse_metadata_kernel(
    c4_topk_lengths_clamp1_ptr,
    c4_sparse_topk_lengths_ptr,
    c4_sparse_page_indices_ptr,
    c4_topk_lengths_stride,
    C4_SPARSE_TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(c4_topk_lengths_clamp1_ptr + row * c4_topk_lengths_stride)
    tl.store(c4_sparse_topk_lengths_ptr + row, tl.minimum(length, C4_SPARSE_TOPK))

    columns = tl.arange(0, BLOCK_SIZE)
    tl.store(
        c4_sparse_page_indices_ptr + row * C4_SPARSE_TOPK + columns,
        -1,
        mask=columns < C4_SPARSE_TOPK,
    )


def init_c4_sparse_metadata(
    c4_topk_lengths_clamp1: torch.Tensor,
    c4_sparse_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clamp C4 lengths and reset the sparse index workspace.

    CUDA uses one row-wise Triton launch. Other devices retain the equivalent
    Torch implementation used before this fusion.
    """
    assert c4_topk_lengths_clamp1.dim() == 1
    assert c4_topk_lengths_clamp1.dtype == torch.int32
    assert c4_sparse_topk in (512, 1024)

    batch_size = c4_topk_lengths_clamp1.shape[0]
    # PyTorch reports ROCm tensors as ``is_cuda`` too. Keep the pre-existing
    # Torch path there until this Triton launch has explicit HIP coverage.
    if not c4_topk_lengths_clamp1.is_cuda or torch.version.hip is not None:
        return (
            torch.clamp(c4_topk_lengths_clamp1, max=c4_sparse_topk),
            torch.full(
                (batch_size, c4_sparse_topk),
                -1,
                dtype=torch.int32,
                device=c4_topk_lengths_clamp1.device,
            ),
        )

    c4_sparse_topk_lengths = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=c4_topk_lengths_clamp1.device,
    )
    c4_sparse_page_indices = torch.empty(
        (batch_size, c4_sparse_topk),
        dtype=torch.int32,
        device=c4_topk_lengths_clamp1.device,
    )
    if batch_size == 0:
        return c4_sparse_topk_lengths, c4_sparse_page_indices

    _init_c4_sparse_metadata_kernel[(batch_size,)](
        c4_topk_lengths_clamp1,
        c4_sparse_topk_lengths,
        c4_sparse_page_indices,
        c4_topk_lengths_clamp1.stride(0),
        C4_SPARSE_TOPK=c4_sparse_topk,
        BLOCK_SIZE=c4_sparse_topk,
        num_warps=8,
    )
    return c4_sparse_topk_lengths, c4_sparse_page_indices
