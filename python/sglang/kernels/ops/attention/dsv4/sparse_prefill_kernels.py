"""SWA token-id build and topk+SWA index combine kernels for DSV4 sparse prefill.

Migrated from ``sglang.srt.layers.attention.dsv4.sparse_prefill_utils`` (RFC #29630, Phase 2.5).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import ceil_align

# FlashMLA sparse prefill asserts ``params.topk % B_TOPK == 0``. B_TOPK is 64
# for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _build_swa_token_ids_kernel(
    out_ptr,
    swa_first_pos_ptr,
    swa_gather_lens_ptr,
    swa_offsets_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    req_to_token_stride,
    full_to_swa_ptr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    first_pos = tl.load(swa_first_pos_ptr + batch_idx)
    gather_len = tl.load(swa_gather_lens_ptr + batch_idx)
    out_off = tl.load(swa_offsets_ptr + batch_idx).to(tl.int64)
    req_pool_idx = tl.load(req_pool_indices_ptr + batch_idx).to(tl.int64)

    for i in range(worker_id, gather_len, num_workers):
        pos = first_pos + i
        full_id = tl.load(
            req_to_token_ptr + req_pool_idx * req_to_token_stride + pos
        ).to(tl.int64)
        swa_id = tl.load(full_to_swa_ptr + full_id).to(tl.int32)
        tl.store(out_ptr + out_off + i, swa_id)


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "top_k"])
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    query_pos_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    compressed_base_ptr,
    swa_base_ptr,
    swa_indices_ptr,
    swa_indices_stride,
    swa_lengths_ptr,
    num_tokens,
    num_reqs,
    top_k,
    EXPLICIT_SWA: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    # BLOCK_T consecutive tokens per program, so the grid scales with tokens, not requests.
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = token < num_tokens

    # query_start_loc may be a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    base = tl.load(query_start_loc_ptr)
    # Each token's request: the last r with query_start_loc[r] - base <= token.
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.full([BLOCK_T], 0, dtype=tl.int32) + num_reqs
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        start = tl.load(query_start_loc_ptr + mid, mask=lo < hi, other=0) - base
        go_right = (lo < hi) & (start <= token)
        lo = tl.where(go_right, mid, lo)
        hi = tl.where((lo < hi) & ~go_right, mid - 1, hi)
    batch_idx = tl.minimum(lo, num_reqs - 1)
    # Rows past the last request's queries belong to no request: -1 rows, length 0.
    owned = valid & (token < tl.load(query_start_loc_ptr + num_reqs) - base)

    pos = tl.load(query_pos_ptr + token, mask=owned, other=0)
    seq_len = tl.load(seq_lens_ptr + batch_idx, mask=owned, other=0)
    gather_len = tl.load(gather_lens_ptr + batch_idx, mask=owned, other=0)
    compressed_base = tl.load(compressed_base_ptr + batch_idx, mask=owned, other=0)
    swa_base = tl.load(swa_base_ptr + batch_idx, mask=owned, other=0)
    # SWA portion of the gathered buffer starts from position
    # (seq_len - gather_len), not 0. The +pos-gather_start formula maps a
    # query's window back into the workspace's SWA region.
    gather_start = seq_len - gather_len

    # -1 entries inside the top-k span stay -1 (attention skips them).
    # top_k=0 disables the compressed portion for SWA-only layers.
    topk_len = tl.where(owned, tl.minimum((pos + 1) // COMPRESS_RATIO, top_k), 0)
    if EXPLICIT_SWA:
        swa_len = tl.load(swa_lengths_ptr + token, mask=owned, other=0)
    else:
        swa_len = tl.where(owned, tl.minimum(pos + 1, WINDOW_SIZE), 0)

    combined_row = token.to(tl.int64)[:, None] * combined_indices_stride
    topk_row = token.to(tl.int64)[:, None] * topk_indices_stride

    # Whole rows, -1 padded, so the output needs no fill pass:
    # [top-k (-1 holes kept) | window | -1 up to WIDTH].
    offset = tl.arange(0, PADDED_WIDTH)[None, :]
    in_topk = offset < topk_len[:, None]
    topk_vals = tl.load(
        topk_indices_ptr + topk_row + offset, mask=valid[:, None] & in_topk, other=-1
    )
    # Workspace SWA index: swa_base[r] + (gather_offset_in_buffer).
    # For positions [pos - swa_len + 1, pos], the buffer offsets are
    # [pos - swa_len + 1 - gather_start, pos - gather_start].
    in_swa = offset < (topk_len + swa_len)[:, None]
    if EXPLICIT_SWA:
        swa_vals = tl.load(
            swa_indices_ptr
            + token.to(tl.int64)[:, None] * swa_indices_stride
            + (offset - topk_len[:, None]),
            mask=owned[:, None] & ~in_topk & in_swa,
            other=-1,
        )
        swa_vals = tl.where(swa_vals >= 0, swa_base[:, None] + swa_vals, -1)
    else:
        swa_start = swa_base + pos - swa_len + 1 - gather_start
        swa_vals = swa_start[:, None] + (offset - topk_len[:, None])
    vals = tl.where(
        in_topk,
        tl.where(topk_vals >= 0, topk_vals + compressed_base[:, None], -1),
        tl.where(in_swa, swa_vals, -1),
    )
    tl.store(
        combined_indices_ptr + combined_row + offset,
        vals,
        mask=valid[:, None] & (offset < WIDTH),
    )

    tl.store(combined_lens_ptr + token, topk_len + swa_len, mask=valid)


def combined_topk_width(topk: int, window_size: int) -> int:
    """Width of the padded combined_indices last dim that
    ``combine_topk_swa_indices`` would produce for these args."""
    return ceil_align(topk + window_size, SPARSE_PREFILL_TOPK_ALIGNMENT)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_pos: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
    swa_indices: Optional[torch.Tensor] = None,
    swa_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine top-k and SWA indices for flash_mla_sparse_fwd.

    Top-k indices are int32 [num_tokens, K] in each request's compressed region.
    Invalid entries within topk_len must be -1; later entries are ignored.
    query_start_loc can include a cross-chunk offset; query_pos is absolute.
    compressed_base and swa_base address the flat workspace. SWA-only layers use
    topk == 0, but compress_ratio must still be positive.
    Optional SWA indices are relative to swa_base and preserve replay floors.

    Returns int32 indices [num_tokens, padded_topk_swa] and per-token scanned-prefix
    lengths, including -1 entries skipped by attention. Width is padded to 128.
    Rows are written whole and -1 padded, so preallocated out_indices / out_lens
    are overwritten entirely; rows past the last request are -1 with length 0.
    """
    assert topk_indices.dtype == torch.int32
    assert query_start_loc.dtype == torch.int32
    assert query_pos.dtype == torch.int32
    assert seq_lens.dtype == torch.int32
    assert gather_lens.dtype == torch.int32
    assert compressed_base.dtype == torch.int32
    assert swa_base.dtype == torch.int32
    assert compress_ratio >= 1, "compress_ratio must be >= 1 (use topk=0 for SWA-only)"
    assert topk_indices.shape[-1] >= topk, (
        f"topk_indices width {topk_indices.shape[-1]} must be >= topk {topk}"
    )

    num_tokens = topk_indices.shape[0]
    if swa_indices is not None:
        assert swa_indices.shape == (num_tokens, window_size)
        assert swa_lengths is not None and swa_lengths.shape == (num_tokens,)
        assert swa_indices.dtype == swa_lengths.dtype == torch.int32
    num_reqs = seq_lens.shape[0]
    combined_topk = combined_topk_width(topk, window_size)
    if out_indices is None:
        # The kernel writes whole -1 padded rows.
        combined_indices = torch.empty(
            (num_tokens, combined_topk),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        assert out_indices.shape == (num_tokens, combined_topk)
        assert out_indices.dtype == torch.int32
        combined_indices = out_indices
    if out_lens is None:
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        assert out_lens.shape == (num_tokens,)
        assert out_lens.dtype == torch.int32
        combined_lens = out_lens

    if num_tokens == 0:
        return combined_indices, combined_lens
    block_t = 2  # Measured on B200 with num_warps=2, 16384 tokens, bs 1 to 64.
    _combine_topk_swa_indices_kernel[(triton.cdiv(num_tokens, block_t),)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        query_pos,
        seq_lens,
        gather_lens,
        compressed_base,
        swa_base,
        swa_indices,
        0 if swa_indices is None else swa_indices.stride(0),
        swa_lengths,
        num_tokens,
        num_reqs,
        top_k=topk,
        EXPLICIT_SWA=swa_indices is not None,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        WIDTH=combined_topk,
        PADDED_WIDTH=triton.next_power_of_2(combined_topk),
        BLOCK_T=block_t,
        SEARCH_STEPS=max(num_reqs, 1).bit_length(),
        num_warps=2,
    )
    return combined_indices, combined_lens


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a flat list of physical SWA-cache token IDs covering each
    request's positional union of every query's SWA window.

    Per request, the union spans seq positions
    ``[max(0, seq_len - extend - W + 1), seq_len)``, of length
    ``min(seq_len, extend + W - 1)``. Each position is translated through
    ``req_to_token`` (full kv-cache id) and then ``full_to_swa`` (SWA
    cache id) to land in the SWA-cache token-id space that
    ``dequantize_k_cache_paged_dsv4`` consumes.

    Args:
        seq_lens: (num_reqs,) int32, per-request total sequence length.
        extend_seq_lens: (num_reqs,) int32, per-request query length.
        req_pool_indices: (num_reqs,) int32, per-request row in
            ``req_to_token``.
        req_to_token: (num_reqs_max, max_seq_len) int32. Full kv-cache id
            per (request, seq position).
        full_to_swa: (full_pool_size + extra,) int64. Maps full kv id to
            SWA-cache id.
        swa_window: int. SWA window size.
        total_swa: Number of token IDs to allocate, computed from CPU lengths.

    Returns:
        swa_token_ids: (total_swa,) int32, flat physical SWA-cache token IDs.
        swa_first_pos: (num_reqs,) int32, first seq position covered per req.
        swa_gather_lens: (num_reqs,) int32, gather length per request.
        swa_offsets: (num_reqs+1,) int32, exclusive cumsum of swa_gather_lens.
    """
    assert seq_lens.dtype == torch.int32
    assert extend_seq_lens.dtype == torch.int32
    assert req_pool_indices.dtype == torch.int32
    assert req_to_token.dtype == torch.int32
    assert full_to_swa.dtype == torch.int64

    num_reqs = seq_lens.shape[0]
    device = seq_lens.device

    swa_gather_lens = torch.minimum(seq_lens, extend_seq_lens + (swa_window - 1)).to(
        torch.int32
    )
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if total_swa == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    NUM_WORKERS = 128
    _build_swa_token_ids_kernel[(num_reqs, NUM_WORKERS)](
        swa_token_ids,
        swa_first_pos,
        swa_gather_lens,
        swa_offsets,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        full_to_swa,
    )
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets
