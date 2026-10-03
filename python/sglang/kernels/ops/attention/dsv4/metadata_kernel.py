from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_all_compressed_indices_kernel(
    page_table,
    seq_lens,
    page_indices,
    raw_indices,
    PAGE_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    positions = tl.arange(0, BLOCK)
    length = tl.load(seq_lens + row)
    valid = (positions < length) & (positions < TOPK)
    slots_per_page = PAGE_SIZE // RATIO
    pages = tl.load(
        page_table + row * PAGE_STRIDE + positions // slots_per_page,
        mask=valid,
        other=0,
    )
    slots = pages * slots_per_page + positions % slots_per_page
    tl.store(
        page_indices + row * TOPK + positions,
        tl.where(valid, slots, -1),
        positions < TOPK,
    )
    if raw_indices is not None:
        tl.store(
            raw_indices + row * TOPK + positions,
            tl.where(valid, positions, -1),
            positions < TOPK,
        )


def fill_all_compressed_indices(
    page_table: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    compress_ratio: int,
    page_size: int,
    raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fill all reachable slots; the caller guarantees compressed length <= top-k."""
    topk = page_indices.shape[1]
    _fill_all_compressed_indices_kernel[(compressed_seq_lens.numel(),)](
        page_table,
        compressed_seq_lens,
        page_indices,
        raw_indices,
        page_table.stride(0),
        topk,
        compress_ratio,
        page_size,
        triton.next_power_of_2(topk),
    )


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
):
    batch_id = tl.program_id(0)
    if batch_id >= bs:
        return

    seq_len = tl.load(seq_lens_ptr + batch_id)
    position = tl.load(positions_ptr + batch_id)
    is_write_token = batch_id < num_write_tokens
    raw_out_loc = tl.load(raw_out_loc_ptr + batch_id, mask=is_write_token, other=0)

    c4_should_compress = (seq_len % 4) == 0
    c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
    c4_positions = position & (~3)
    c4_seq_lens_raw = seq_len // 4
    c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

    tl.store(c4_out_loc_ptr + batch_id, c4_out_loc, mask=is_write_token)
    tl.store(c4_positions_ptr + batch_id, c4_positions)
    tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
    tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

    c128_should_compress = (seq_len % 128) == 0
    c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
    c128_positions = position & (~127)
    c128_seq_lens_raw = seq_len // 128
    c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

    tl.store(c128_out_loc_ptr + batch_id, c128_out_loc, mask=is_write_token)
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
    # CP may add padding rows to the attention metadata, but those rows have
    # no cache-write locations. Keep the write buffers unpadded and mask those
    # rows in the kernel.
    num_write_tokens = raw_out_loc.shape[0]
    assert num_write_tokens <= bs, (
        f"raw_out_loc has {num_write_tokens} rows, expected at most {bs} metadata rows"
    )
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
        assert page_table is not None, (
            "page_table required when compute_page_indices=True"
        )
        assert page_size >= 128 and page_size % 128 == 0, (
            "page_size must be a multiple of 128 when compute_page_indices=True"
        )
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
        num_write_tokens,
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


@triton.jit
def _store_compressed_positions(
    seq, valid, offsets, dst, BS: tl.constexpr, RATIO: tl.constexpr
):
    keep = valid & (seq % RATIO == 0)
    ordinal = tl.cumsum(keep.to(tl.int32), 0) - 1
    count = tl.sum(keep.to(tl.int32), 0)
    # Selected requests own [0, count), in their original order. Tail stores
    # own [count, BS): disjoint writes, with no zero-then-scatter race.
    tl.store(dst + ordinal, seq.to(tl.int64) - RATIO, mask=keep)
    tl.store(dst + offsets, 0, mask=(offsets >= count) & (offsets < BS))


@triton.jit(do_not_specialize=["n_c4", "n_c128"])
def _refresh_dsv4_decode_metadata_kernel(
    seq_lens,
    c4_src,
    c128_src,
    c4_loc,
    c128_loc,
    c4_positions,
    c128_positions,
    start_pos,
    seqused,
    n_c4,
    n_c128,
    BS: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    C4_STRIDE: tl.constexpr,
    C128_STRIDE: tl.constexpr,
    HAS_C4: tl.constexpr,
    HAS_C128: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    seq = tl.load(seq_lens + offsets * SEQ_STRIDE, mask=offsets < BS, other=0)
    valid = (offsets < BS) & (seq > 0)
    tl.store(start_pos + offsets, tl.maximum(seq - 1, 0), mask=offsets < BS)
    tl.store(seqused + offsets, valid.to(tl.int32), mask=offsets < BS)

    if HAS_C4:
        loc = tl.load(c4_src + offsets * C4_STRIDE, mask=offsets < n_c4, other=0)
        tl.store(c4_loc + offsets, loc, mask=offsets < BS)
        _store_compressed_positions(seq, valid, offsets, c4_positions, BS, 4)
    if HAS_C128:
        loc = tl.load(c128_src + offsets * C128_STRIDE, mask=offsets < n_c128, other=0)
        tl.store(c128_loc + offsets, loc, mask=offsets < BS)
        _store_compressed_positions(seq, valid, offsets, c128_positions, BS, 128)


def refresh_dsv4_decode_metadata(
    seq_lens: torch.Tensor,
    *,
    c4_src: Optional[torch.Tensor],
    c128_src: Optional[torch.Tensor],
    c4_loc: torch.Tensor,
    c128_loc: torch.Tensor,
    c4_positions: torch.Tensor,
    c128_positions: torch.Tensor,
    start_pos: torch.Tensor,
    seqused: torch.Tensor,
    has_c4: bool,
    has_c128: bool,
) -> None:
    """Update persistent NPU decode buffers on the current stream, without scratch.

    Decode has one output slot per graph request in all six contiguous output
    vectors. Zero sequence lengths represent graph padding/idle rows. Location
    sources may be missing, empty, strided or a different integer dtype; only
    enabled compression ratios are written. Counts are runtime scalars so a
    C4/C128 boundary does not trigger count-specific JIT compilation.
    """
    bs = seq_lens.numel()
    assert seq_lens.ndim == 1
    for output in (c4_loc, c128_loc, c4_positions, c128_positions, start_pos, seqused):
        assert output.ndim == 1 and output.numel() == bs and output.is_contiguous(), (
            "fused decode metadata requires one contiguous output slot per request"
        )
    n_c4 = c4_src.numel() if has_c4 and c4_src is not None else 0
    n_c128 = c128_src.numel() if has_c128 and c128_src is not None else 0
    assert n_c4 <= bs and n_c128 <= bs, (
        f"graph replay 1D metadata overflow: c4={n_c4}, c128={n_c128}, dst={bs}"
    )
    if bs == 0:
        return
    # Masked-out loads use a valid device pointer even for empty source tensors.
    # No placeholder tensor or dtype-conversion allocation is needed.
    c4_src = c4_src if n_c4 else c4_loc
    c128_src = c128_src if n_c128 else c128_loc
    _refresh_dsv4_decode_metadata_kernel[(1,)](
        seq_lens,
        c4_src,
        c128_src,
        c4_loc,
        c128_loc,
        c4_positions,
        c128_positions,
        start_pos,
        seqused,
        n_c4,
        n_c128,
        BS=bs,
        SEQ_STRIDE=seq_lens.stride(0),
        C4_STRIDE=c4_src.stride(0),
        C128_STRIDE=c128_src.stride(0),
        HAS_C4=has_c4,
        HAS_C128=has_c128,
        BLOCK_SIZE=triton.next_power_of_2(bs),
    )


@triton.jit
def _low_ratio_metadata(
    LENS,
    LOC,
    OUT1,
    LEN1,
    SPARSE1,
    PAGE1,
    OUT2,
    LEN2,
    SPARSE2,
    PAGE2,
    TOPK: tl.constexpr,
    PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(LENS + row).to(tl.int32)
    loc = tl.load(LOC + row).to(tl.int64)
    len1, len2 = tl.maximum(length, 1), tl.maximum(length >> 1, 1)
    tl.store(OUT1 + row, loc)
    tl.store(OUT2 + row, tl.where((length & 1) == 0, loc >> 1, -1))
    tl.store(LEN1 + row, len1)
    tl.store(LEN2 + row, len2)
    tl.store(SPARSE1 + row, tl.minimum(len1, TOPK))
    tl.store(SPARSE2 + row, tl.minimum(len2, TOPK))
    cols = tl.arange(0, BLOCK)
    tl.store(PAGE1 + row * PADDED + cols, -1, cols < PADDED)
    tl.store(PAGE2 + row * PADDED + cols, -1, cols < PADDED)


class LowRatioMetadata(NamedTuple):
    """Per-request slots and lengths of the ratio-1 and ratio-2 compressed caches."""

    c1_out_loc: torch.Tensor
    c1_seq_lens: torch.Tensor
    c1_sparse_lens: torch.Tensor
    c1_page_indices: torch.Tensor
    c2_out_loc: torch.Tensor
    c2_seq_lens: torch.Tensor
    c2_sparse_lens: torch.Tensor
    c2_page_indices: torch.Tensor


def build_low_ratio_metadata(seq_lens, out_loc, topk) -> LowRatioMetadata:
    assert seq_lens.numel() == out_loc.numel()
    rows = seq_lens.numel()
    kw = dict(device=seq_lens.device, dtype=torch.int32)
    padded = triton.cdiv(topk, 64) * 64
    outputs = []
    for _ in range(2):
        outputs.extend(
            [
                torch.empty(rows, device=out_loc.device, dtype=torch.int64),
                torch.empty(rows, **kw),
                torch.empty(rows, **kw),
                torch.empty((rows, padded), **kw),
            ]
        )
    _low_ratio_metadata[(rows,)](
        seq_lens, out_loc, *outputs, topk, padded, triton.next_power_of_2(padded)
    )
    return LowRatioMetadata(*outputs)
