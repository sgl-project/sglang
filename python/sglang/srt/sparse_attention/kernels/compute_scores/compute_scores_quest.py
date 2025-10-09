import torch
import triton
import triton.language as tl

@triton.jit
def _compute_quest_score_kernel(
    Q,  # shape: [batch_size, NUM_Q_HEADS * HEAD_DIM]
    K,  # shape: [num_pages, 2, NUM_KV_HEADS, HEAD_DIM]
    Out,  # shape: [batch_size, NUM_KV_HEADS, num_pages]
    Stream_indices_page,  # shape: [bs, stream_len] or None
    kv_pages_per_seq,  # shape: [bs, max_num_token_pages]
    kv_pages_num_per_seq,  # shape: [bs]
    kv_pages_per_seq_max,  # shape: [bs]
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    scores_stride_b: tl.constexpr,
    scores_stride_h: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STREAM_LEN: tl.constexpr,
    ACTUAL_STREAM_LEN: tl.constexpr,
    MAX_NUM_TOKEN_PAGES: tl.constexpr,
    PADDED_MAX_NUM_TOKEN_PAGES: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    page_id = tl.program_id(2)
    
    # Early exit mechanism based on max page index
    max_page_index = tl.load(kv_pages_per_seq_max + bid)
    if page_id > max_page_index:
        out_ptr = Out + bid * scores_stride_b + kv_head_id * scores_stride_h + page_id
        tl.store(out_ptr, -float('inf'))
        return
    
    # Check if page is in valid pages
    num_valid_pages = tl.load(kv_pages_num_per_seq + bid)
    token_base_ptr = kv_pages_per_seq + bid * MAX_NUM_TOKEN_PAGES
    token_page_offsets = tl.arange(0, PADDED_MAX_NUM_TOKEN_PAGES)
    valid_page_ids = tl.load(token_base_ptr + token_page_offsets, 
                            mask=(token_page_offsets < MAX_NUM_TOKEN_PAGES) & (token_page_offsets < num_valid_pages), other=-1)
    
    is_in_valid_pages = tl.sum((valid_page_ids == page_id).to(tl.int32)) > 0
    is_not_zero = page_id != 0
    is_page_valid = is_in_valid_pages & is_not_zero
    
    # set stream page to -inf
    if ACTUAL_STREAM_LEN > 0:
        stream_base_ptr = Stream_indices_page + bid * ACTUAL_STREAM_LEN
        stream_offsets = tl.arange(0, STREAM_LEN)
        stream_page_ids = tl.load(stream_base_ptr + stream_offsets, mask=stream_offsets < ACTUAL_STREAM_LEN, other=-1)
        is_in_stream = tl.sum((stream_page_ids == page_id).to(tl.int32)) > 0
        is_page_valid = is_page_valid & (~is_in_stream)

    # If page is not in valid pages, set to -inf
    if not is_page_valid:
        out_ptr = Out + bid * scores_stride_b + kv_head_id * scores_stride_h + page_id
        tl.store(out_ptr, -float('inf'))
        return
    
    total_page_score = 0.0
    
    k_base_ptr = K + page_id * (2 * NUM_KV_HEADS * HEAD_DIM) + kv_head_id * HEAD_DIM
    k_min_base_ptr = k_base_ptr
    k_max_base_ptr = k_base_ptr + NUM_KV_HEADS * HEAD_DIM
    
    # Loop over HEAD_DIM in blocks
    for h_start in range(0, HEAD_DIM, BLOCK_D):
        h_offsets = h_start + tl.arange(0, BLOCK_D)
        h_mask = h_offsets < HEAD_DIM
        
        k_min_block = tl.load(k_min_base_ptr + h_offsets, mask=h_mask, other=0.0)
        k_max_block = tl.load(k_max_base_ptr + h_offsets, mask=h_mask, other=0.0)
        
        block_score = 0.0
        
        for group_idx in range(GROUP_SIZE):
            q_head_id = kv_head_id * GROUP_SIZE + group_idx
            
            q_ptr = Q + bid * (NUM_Q_HEADS * HEAD_DIM) + q_head_id * HEAD_DIM + h_offsets
            q_block = tl.load(q_ptr, mask=h_mask, other=0.0)
            
            mul_min = q_block * k_min_block
            mul_max = q_block * k_max_block
            res_block = tl.maximum(mul_min, mul_max)
            
            block_score += tl.sum(res_block, axis=0)
            
        total_page_score += block_score

    out_ptr = Out + bid * scores_stride_b + kv_head_id * scores_stride_h + page_id
    tl.store(out_ptr, total_page_score)
    
def compute_quest_score(q: torch.Tensor, # [bs, NUM_Q_HEADS*HEAD_DIM]
                        k: torch.Tensor, # [num_pages, 2, NUM_KV_HEADS, HEAD_DIM]
                        out: torch.Tensor, # [bs, NUM_KV_HEADS, num_pages]
                        stream_indices_page: torch.Tensor = None,
                        kv_pages_per_seq: torch.Tensor = None,
                        kv_pages_num_per_seq: torch.Tensor = None):
    bs = q.shape[0]
    num_pages = k.shape[0]  
    NUM_KV_HEADS = k.shape[2] 
    HEAD_DIM = k.shape[-1]
    NUM_Q_HEADS = q.shape[-1] // HEAD_DIM
    
    if stream_indices_page is not None:
        assert stream_indices_page.shape[0] == bs
        actual_stream_len = stream_indices_page.shape[1]
        stream_len = triton.next_power_of_2(actual_stream_len) if actual_stream_len > 0 else 0
    else:
        actual_stream_len = 0
        stream_len = 0
        stream_indices_page = torch.empty((bs, 0), dtype=torch.int32, device=q.device)
    
    max_num_token_pages = kv_pages_per_seq.shape[1]
    padded_max_num_token_pages = triton.next_power_of_2(max_num_token_pages)
    
    # NOTE: for early exit
    kv_pages_per_seq_max = kv_pages_per_seq.max(dim=-1).values
    
    grid = (bs, NUM_KV_HEADS, num_pages)
    
    BLOCK_D = 128

    _compute_quest_score_kernel[grid](
        q, 
        k, 
        out, 
        stream_indices_page,
        kv_pages_per_seq,
        kv_pages_num_per_seq,
        kv_pages_per_seq_max,
        NUM_Q_HEADS=NUM_Q_HEADS, 
        NUM_KV_HEADS=NUM_KV_HEADS, 
        HEAD_DIM=HEAD_DIM,
        GROUP_SIZE=NUM_Q_HEADS // NUM_KV_HEADS, 
        scores_stride_b=out.stride(0), 
        scores_stride_h=out.stride(1),
        BLOCK_D=BLOCK_D,
        STREAM_LEN=stream_len,
        ACTUAL_STREAM_LEN=actual_stream_len,
        MAX_NUM_TOKEN_PAGES=max_num_token_pages,
        PADDED_MAX_NUM_TOKEN_PAGES=padded_max_num_token_pages,
    )