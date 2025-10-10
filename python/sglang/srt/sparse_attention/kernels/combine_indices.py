import torch
import triton
import triton.language as tl


@triton.jit
def combine_indices_kernel(
    stream_indices_ptr,
    retrived_cache_indices_ptr,
    cur_req_pool_indices_ptr,
    pre_req_pool_indices_ptr,
    req_to_token_ptr,
    seq_lens_ptr,
    new_seq_lens_ptr,
    page_table_ptr,
    cur_bs,
    pre_bs,
    cache_len,
    stream_len,
    max_seq_len,
    page_table_stride, 
    
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    
    if bid >= cur_bs:
        return
    
    cur_req_idx = tl.load(cur_req_pool_indices_ptr + bid)
    
    is_in_pre_bs = False
    pre_position = -1
    
    for i in range(pre_bs):
        pre_req_idx = tl.load(pre_req_pool_indices_ptr + i)
        
        match_found = (pre_req_idx != -1) & (pre_req_idx == cur_req_idx)
        is_in_pre_bs = is_in_pre_bs | match_found
        pre_position = tl.where(match_found, i, pre_position)
        
    page_table_row_offset = cur_req_idx * page_table_stride
    
    if is_in_pre_bs:
        # 1. copy stream indices first
        stream_offsets = tl.arange(0, BLOCK_SIZE)
        stream_mask = stream_offsets < stream_len
        
        if stream_len <= BLOCK_SIZE:
            stream_vals = tl.load(stream_indices_ptr + bid * stream_len + stream_offsets, mask=stream_mask)
            tl.store(page_table_ptr + page_table_row_offset + stream_offsets, stream_vals, mask=stream_mask)
        else:
            for block_start in range(0, stream_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < stream_len
                stream_vals = tl.load(stream_indices_ptr + bid * stream_len + block_offsets, mask=block_mask)
                tl.store(page_table_ptr + page_table_row_offset + block_offsets, stream_vals, mask=block_mask)

        # 2. copy retrived indices
        cache_offsets = tl.arange(0, BLOCK_SIZE)
        if cache_len <= BLOCK_SIZE:
            cache_mask = cache_offsets < cache_len
            cache_vals = tl.load(retrived_cache_indices_ptr + pre_position * cache_len + cache_offsets, mask=cache_mask)
            tl.store(page_table_ptr + page_table_row_offset + stream_len + cache_offsets, cache_vals, mask=cache_mask)
        else:
            for block_start in range(0, cache_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < cache_len
                cache_vals = tl.load(retrived_cache_indices_ptr + pre_position * cache_len + block_offsets, mask=block_mask)
                tl.store(page_table_ptr + page_table_row_offset + stream_len + block_offsets, cache_vals, mask=block_mask)
        tl.store(new_seq_lens_ptr + bid, stream_len + cache_len)
    else:
        # new request, copy all token indices
        seq_len = tl.load(seq_lens_ptr + bid) 
        token_offsets = tl.arange(0, BLOCK_SIZE)
        if seq_len <= BLOCK_SIZE:
            token_mask = token_offsets < seq_len
            
            token_vals = tl.load(req_to_token_ptr + cur_req_idx * max_seq_len + token_offsets, mask=token_mask)
            
            tl.store(page_table_ptr + page_table_row_offset + token_offsets, token_vals, mask=token_mask)
        else:
            for block_start in range(0, seq_len, BLOCK_SIZE):
                block_offsets = tl.arange(0, BLOCK_SIZE) + block_start
                block_mask = block_offsets < seq_len
                token_vals = tl.load(req_to_token_ptr + cur_req_idx * max_seq_len + block_offsets, mask=block_mask)
                tl.store(page_table_ptr + page_table_row_offset + block_offsets, token_vals, mask=block_mask)
        tl.store(new_seq_lens_ptr + bid, seq_len)

def combine_indices(
    stream_indices: torch.Tensor, #[cur_bs, stream_len]
    retrived_cache_indices: torch.Tensor, #[pre_bs, cache_len]
    cur_req_pool_indices: torch.Tensor, #[cur_bs]
    pre_req_pool_indices: torch.Tensor, #[pre_bs]
    req_to_token: torch.Tensor, #[max_bs, max_seq_len]
    page_table: torch.Tensor, #[max_bs, max_seq_len]
    seq_lens: torch.Tensor, #[cur_bs]
) -> tuple[torch.Tensor, torch.Tensor]:
    cur_bs, stream_len = stream_indices.shape
    pre_bs = pre_req_pool_indices.shape[0]
    _, cache_len = retrived_cache_indices.shape
    max_bs, max_seq_len = req_to_token.shape
    page_table_max_bs, page_table_stride = page_table.shape
    
    new_seq_lens = torch.zeros_like(seq_lens)
    
    grid = (cur_bs,)
    BLOCK_SIZE = 256

    combine_indices_kernel[grid](
        stream_indices,
        retrived_cache_indices,
        cur_req_pool_indices,
        pre_req_pool_indices,
        req_to_token,
        seq_lens,
        new_seq_lens,
        page_table,
        cur_bs,
        pre_bs,
        cache_len,
        stream_len,
        max_seq_len,
        page_table_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return page_table, new_seq_lens
