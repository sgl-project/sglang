"""SWA token-id build and topk+SWA index combine kernels for DSV4 sparse prefill.

Migrated from ``sglang.srt.layers.attention.dsv4.sparse_prefill_utils`` (RFC #29630, Phase 2.5).
"""

import triton
import triton.language as tl


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
