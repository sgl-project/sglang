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


@triton.jit(do_not_specialize=["top_k"])
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    compressed_base_ptr,
    swa_base_ptr,
    top_k,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    # query_start_loc may be a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    compressed_base = tl.load(compressed_base_ptr + batch_idx)
    swa_base = tl.load(swa_base_ptr + batch_idx)
    start_pos = seq_len - query_len
    # SWA portion of the gathered buffer starts from position
    # (seq_len - gather_len), not 0. The +pos-gather_start formula maps a
    # query's window back into the workspace's SWA region.
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        # Both the C4 indexer and the C128 metadata builder emit
        # min((pos+1)//compress_ratio, topk_tokens) valid entries. Caller
        # passes top_k=0 for SWA-only layers to zero this out.
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, top_k)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        combined_row = token_idx.to(tl.int64) * combined_indices_stride
        topk_row = token_idx.to(tl.int64) * topk_indices_stride

        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_vals = tl.load(
            topk_indices_ptr + topk_row + offset,
            mask=mask,
        )
        tl.store(
            combined_indices_ptr + combined_row + offset,
            topk_vals + compressed_base,
            mask=mask,
        )

        offset = tl.arange(0, WINDOW_SIZE)
        # Workspace SWA index: swa_base[r] + (gather_offset_in_buffer).
        # For positions [pos - swa_len + 1, pos], the buffer offsets are
        # [pos - swa_len + 1 - gather_start, pos - gather_start].
        tl.store(
            combined_indices_ptr + combined_row + topk_len + offset,
            swa_base + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)
