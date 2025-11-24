import torch
import triton
import triton.language as tl


@triton.jit
def combine_indices_paged_kernel(
    retrived_cache_indices_page_ptr,
    cur_req_pool_indices_ptr,
    pre_req_pool_indices_ptr,
    req_to_token_ptr,
    seq_lens_ptr,
    diff_ptr,
    new_seq_lens_ptr,
    page_table_ptr,
    cur_bs,
    pre_bs,
    max_bs: tl.constexpr,
    num_kv_heads: tl.constexpr,
    cache_len: tl.constexpr,
    num_sink_pages: tl.constexpr,
    num_local_pages: tl.constexpr,
    max_seq_len: tl.constexpr,
    max_pages: tl.constexpr,
    page_size: tl.constexpr,
    budget_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    batch_idx = pid // num_kv_heads
    kv_head_idx = pid % num_kv_heads

    if batch_idx >= cur_bs:
        return

    cur_req_idx = tl.load(cur_req_pool_indices_ptr + batch_idx)

    is_in_pre_bs = False
    pre_position = -1

    for i in range(pre_bs):
        pre_req_idx = tl.load(pre_req_pool_indices_ptr + i)

        match_found = (pre_req_idx != -1) & (pre_req_idx == cur_req_idx)
        is_in_pre_bs = is_in_pre_bs | match_found
        pre_position = tl.where(match_found, i, pre_position)

    page_table_offset = cur_req_idx * num_kv_heads * max_pages + kv_head_idx * max_pages

    seq_len = tl.load(seq_lens_ptr + batch_idx)

    if is_in_pre_bs and seq_len >= budget_size:
        cache_offsets = tl.arange(0, BLOCK_SIZE)
        retrived_offset = (
            pre_position * num_kv_heads * cache_len + kv_head_idx * cache_len
        )

        if cache_len <= BLOCK_SIZE:
            cache_mask = cache_offsets < cache_len
            cache_vals = tl.load(
                retrived_cache_indices_page_ptr + retrived_offset + cache_offsets,
                mask=cache_mask,
                other=0,
            )
            tl.store(
                page_table_ptr + page_table_offset + cache_offsets,
                cache_vals,
                mask=cache_mask,
            )
        else:
            for block_start in range(0, cache_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < cache_len
                cache_vals = tl.load(
                    retrived_cache_indices_page_ptr + retrived_offset + block_offsets,
                    mask=block_mask,
                    other=0,
                )
                tl.store(
                    page_table_ptr + page_table_offset + block_offsets,
                    cache_vals,
                    mask=block_mask,
                )

        num_pages_per_seq = (seq_len + page_size - 1) // page_size

        sink_page_offsets = tl.arange(0, BLOCK_SIZE) * page_size
        sink_mask = tl.arange(0, BLOCK_SIZE) < num_sink_pages

        token_offset_base = cur_req_idx * max_seq_len
        sink_token_vals = tl.load(
            req_to_token_ptr + token_offset_base + sink_page_offsets,
            mask=sink_mask,
            other=0,
        )
        sink_page_vals = sink_token_vals // page_size

        tl.store(
            page_table_ptr + page_table_offset + cache_len + tl.arange(0, BLOCK_SIZE),
            sink_page_vals,
            mask=sink_mask,
        )

        local_start_page = num_pages_per_seq - num_local_pages
        local_page_offsets = (local_start_page + tl.arange(0, BLOCK_SIZE)) * page_size
        local_mask = tl.arange(0, BLOCK_SIZE) < num_local_pages

        local_token_vals = tl.load(
            req_to_token_ptr + token_offset_base + local_page_offsets,
            mask=local_mask,
            other=0,
        )
        local_page_vals = local_token_vals // page_size

        tl.store(
            page_table_ptr
            + page_table_offset
            + cache_len
            + num_sink_pages
            + tl.arange(0, BLOCK_SIZE),
            local_page_vals,
            mask=local_mask,
        )

        if kv_head_idx == 0:
            diff = tl.load(diff_ptr + batch_idx)
            new_seq_len = (
                page_size * (cache_len + num_sink_pages + num_local_pages) - diff
            )
            tl.store(new_seq_lens_ptr + batch_idx, new_seq_len)
    else:
        num_pages = (seq_len + page_size - 1) // page_size

        page_offsets = tl.arange(0, BLOCK_SIZE)

        if num_pages <= BLOCK_SIZE:
            page_mask = page_offsets < num_pages
            token_positions = page_offsets * page_size
            token_mask = token_positions < max_seq_len
            mask = page_mask & token_mask

            token_vals = tl.load(
                req_to_token_ptr + cur_req_idx * max_seq_len + token_positions,
                mask=mask,
                other=0,
            )
            page_vals = token_vals // page_size

            tl.store(
                page_table_ptr + page_table_offset + page_offsets, page_vals, mask=mask
            )
        else:
            for block_start in range(0, num_pages, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < num_pages
                token_positions = block_offsets * page_size
                token_mask = token_positions < max_seq_len
                mask = block_mask & token_mask

                token_vals = tl.load(
                    req_to_token_ptr + cur_req_idx * max_seq_len + token_positions,
                    mask=mask,
                    other=0,
                )
                page_vals = token_vals // page_size

                tl.store(
                    page_table_ptr + page_table_offset + block_offsets,
                    page_vals,
                    mask=mask,
                )

        if kv_head_idx == 0:
            tl.store(new_seq_lens_ptr + batch_idx, seq_len)


@triton.jit
def combine_indices_paged_sync_kernel(
    retrived_cache_indices_page_ptr,
    cur_req_pool_indices_ptr,
    req_to_token_ptr,
    seq_lens_ptr,
    diff_ptr,
    new_seq_lens_ptr,
    page_table_ptr,
    num_kv_heads: tl.constexpr,
    cache_len: tl.constexpr,
    num_sink_pages: tl.constexpr,
    num_local_pages: tl.constexpr,
    max_seq_len: tl.constexpr,
    max_pages: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    batch_idx = pid // num_kv_heads
    kv_head_idx = pid % num_kv_heads

    # bs, num_kv_heads, num_pages
    # copy sink
    cur_req_idx = tl.load(cur_req_pool_indices_ptr + batch_idx)
    if cur_req_idx < 0:
        return
    req_to_token_ptr = req_to_token_ptr + cur_req_idx * max_seq_len
    retrived_cache_indices_page_ptr = (
        retrived_cache_indices_page_ptr
        + batch_idx * cache_len * num_kv_heads
        + kv_head_idx * cache_len
    )
    page_table_ptr = (
        page_table_ptr
        + cur_req_idx * num_kv_heads * max_pages
        + kv_head_idx * max_pages
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
    num_pages = (seq_len + page_size - 1) // page_size
    if num_pages < cache_len + num_sink_pages + num_local_pages:
        for i in range(0, num_pages, BLOCK_SIZE):
            block_offsets = tl.arange(0, BLOCK_SIZE) + i
            block_mask = block_offsets < num_pages
            cache_vals = tl.load(
                retrived_cache_indices_page_ptr + block_offsets,
                mask=block_mask,
                other=0,
            )
            tl.store(page_table_ptr + block_offsets, cache_vals, mask=block_mask)
            tl.store(new_seq_lens_ptr + batch_idx, seq_len)
    else:
        for i in range(num_sink_pages):
            token_offset = i * page_size
            sink_indices = (
                tl.load(req_to_token_ptr + token_offset, mask=token_offset < seq_len)
                // page_size
            )
            tl.store(page_table_ptr + i, sink_indices)

        for i in range(0, cache_len, BLOCK_SIZE):
            block_offsets = tl.arange(0, BLOCK_SIZE) + i
            block_mask = block_offsets < cache_len
            cache_vals = tl.load(
                retrived_cache_indices_page_ptr + block_offsets,
                mask=block_mask,
                other=0,
            )
            tl.store(
                page_table_ptr + block_offsets + num_sink_pages,
                cache_vals,
                mask=block_mask,
            )

        for i in range(num_local_pages):
            token_offset = (num_pages - num_local_pages + i) * page_size
            local_indices = tl.load(req_to_token_ptr + token_offset) // page_size
            tl.store(page_table_ptr + num_sink_pages + cache_len + i, local_indices)

        if kv_head_idx == 0:
            diff_val = tl.load(diff_ptr + batch_idx)
            tl.store(
                new_seq_lens_ptr + batch_idx,
                page_size * (cache_len + num_sink_pages + num_local_pages) - diff_val,
            )


def combine_indices(
    retrived_cache_indices: torch.Tensor,  # [pre_bs, cache_len] or [max_bs, num_kv_heads, top_k]
    cur_req_pool_indices: torch.Tensor,  # [cur_bs]
    pre_req_pool_indices: torch.Tensor,  # [pre_bs] or [max_bs]
    req_to_token: torch.Tensor,  # [max_bs, max_seq_len]
    page_table: torch.Tensor,  # [max_bs, max_seq_len] or [max_bs, num_kv_heads, max_pages]
    seq_lens: torch.Tensor,  # [cur_bs]
    new_seq_lens: torch.Tensor,  # [cur_bs]
    diff: torch.Tensor,  # [cur_bs]
    num_sink_pages: int,
    num_local_pages: int,
    page_size: int,
    budget_size: int,
    async_retrive: bool,
) -> torch.Tensor:
    cur_bs = cur_req_pool_indices.shape[0]
    max_bs_pre = pre_req_pool_indices.shape[0]
    _, num_kv_heads, cache_len = retrived_cache_indices.shape
    _, _, max_pages = page_table.shape
    max_bs, max_seq_len = req_to_token.shape

    grid = (cur_bs * num_kv_heads,)

    BLOCK_SIZE = 32

    if async_retrive:
        combine_indices_paged_kernel[grid](
            retrived_cache_indices,
            cur_req_pool_indices,
            pre_req_pool_indices,
            req_to_token,
            seq_lens,
            diff,
            new_seq_lens,
            page_table,
            cur_bs,
            max_bs_pre,
            max_bs,
            num_kv_heads,
            cache_len,
            num_sink_pages,
            num_local_pages,
            max_seq_len,
            max_pages,
            page_size,
            budget_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        combine_indices_paged_sync_kernel[grid](
            retrived_cache_indices,
            cur_req_pool_indices,
            req_to_token,
            seq_lens,
            diff,
            new_seq_lens,
            page_table,
            num_kv_heads,
            cache_len,
            num_sink_pages,
            num_local_pages,
            req_to_token.stride(0),
            page_table.stride(1),
            page_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return new_seq_lens[:cur_bs]
