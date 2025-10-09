import torch
import triton
import triton.language as tl


@triton.jit
def combine_indices_paged_kernel(
    stream_indices_page_ptr,
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
    max_bs,
    num_kv_heads,
    cache_len,
    stream_len,
    max_seq_len,
    max_pages,
    page_size,
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
    
    if is_in_pre_bs:
        cache_offsets = tl.arange(0, BLOCK_SIZE)
        retrived_offset = pre_position * num_kv_heads * cache_len + kv_head_idx * cache_len
        
        if cache_len <= BLOCK_SIZE:
            cache_mask = cache_offsets < cache_len
            cache_vals = tl.load(
                retrived_cache_indices_page_ptr + retrived_offset + cache_offsets,
                mask=cache_mask,
                other=0
            )
            tl.store(
                page_table_ptr + page_table_offset + cache_offsets,
                cache_vals,
                mask=cache_mask
            )
        else:
            for block_start in range(0, cache_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < cache_len
                cache_vals = tl.load(
                    retrived_cache_indices_page_ptr + retrived_offset + block_offsets,
                    mask=block_mask,
                    other=0
                )
                tl.store(
                    page_table_ptr + page_table_offset + block_offsets,
                    cache_vals,
                    mask=block_mask
                )
        
        stream_offsets = tl.arange(0, BLOCK_SIZE)
        if stream_len <= BLOCK_SIZE:
            stream_mask = stream_offsets < stream_len
            stream_vals = tl.load(
                stream_indices_page_ptr + batch_idx * stream_len + stream_offsets,
                mask=stream_mask,
                other=0
            )
            tl.store(
                page_table_ptr + page_table_offset + cache_len + stream_offsets,
                stream_vals,
                mask=stream_mask
            )
        else:
            for block_start in range(0, stream_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < stream_len
                stream_vals = tl.load(
                    stream_indices_page_ptr + batch_idx * stream_len + block_offsets,
                    mask=block_mask,
                    other=0
                )
                tl.store(
                    page_table_ptr + page_table_offset + cache_len + block_offsets,
                    stream_vals,
                    mask=block_mask
                )
        
        if kv_head_idx == 0:
            diff = tl.load(diff_ptr + batch_idx)
            new_seq_len = page_size * (cache_len + stream_len) - diff
            tl.store(new_seq_lens_ptr + batch_idx, new_seq_len)
    else:
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        
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
                other=0
            )
            page_vals = token_vals // page_size
            
            tl.store(
                page_table_ptr + page_table_offset + page_offsets,
                page_vals,
                mask=mask
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
                    other=0
                )
                page_vals = token_vals // page_size
                
                tl.store(
                    page_table_ptr + page_table_offset + block_offsets,
                    page_vals,
                    mask=mask
                )
        
        if kv_head_idx == 0:
            tl.store(new_seq_lens_ptr + batch_idx, seq_len)


def combine_indices(
    stream_indices: torch.Tensor,  # [cur_bs, stream_len]
    retrived_cache_indices: torch.Tensor,  # [pre_bs, cache_len] or [max_bs, num_kv_heads, top_k]
    cur_req_pool_indices: torch.Tensor,  # [cur_bs]
    pre_req_pool_indices: torch.Tensor,  # [pre_bs] or [max_bs]
    req_to_token: torch.Tensor,  # [max_bs, max_seq_len]
    page_table: torch.Tensor,  # [max_bs, max_seq_len] or [max_bs, num_kv_heads, max_pages]
    seq_lens: torch.Tensor,  # [cur_bs]
    diff: torch.Tensor,  # [cur_bs]
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_bs, stream_len = stream_indices.shape
    max_bs_pre = pre_req_pool_indices.shape[0]
    _, num_kv_heads, cache_len = retrived_cache_indices.shape
    _, _, max_pages = page_table.shape
    max_bs, max_seq_len = req_to_token.shape
    new_seq_lens = torch.zeros(cur_bs, dtype=torch.int32, device=stream_indices.device)
    
    grid = (cur_bs * num_kv_heads,)
    
    BLOCK_SIZE = 256
    
    combine_indices_paged_kernel[grid](
        stream_indices,
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
        stream_len,
        max_seq_len,
        max_pages,
        page_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return new_seq_lens

