
import torch
import triton
import triton.language as tl

@triton.jit
def quest_page_rep_kernel(
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    reqs_ptr,
    seq_lens_ptr,
    start_page_ptr,
    end_page_ptr,
    req_to_token_ptr,
    k_buffer_ptr,
    # Strides
    req_to_token_stride_req,
    req_to_token_stride_token,
    k_buffer_stride_token,
    k_buffer_stride_head,
    k_buffer_stride_dim,
    page_k_stride_page,
    page_k_stride_head,
    page_k_stride_dim,
    # Shapes
    req_to_token_num_tokens, # To clamp
    k_buffer_num_tokens,     # To clamp
    # Constants
    PAGE_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    req_idx = tl.program_id(0)
    page_rel_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Load request info
    req_id = tl.load(reqs_ptr + req_idx)
    s_page = tl.load(start_page_ptr + req_idx)
    e_page = tl.load(end_page_ptr + req_idx)
    
    current_page = s_page + page_rel_idx
    
    if current_page >= e_page:
        return
        
    seq_len = tl.load(seq_lens_ptr + req_idx)
    logical_token_start = current_page * PAGE_SIZE
    
    # Get physical page index from the first token of the page
    # Clamp logical_token_start to be safe for req_to_token lookup
    # logic from python: tok_start.clamp(0, req_to_token.shape[1] - 1)
    
    safe_log_tok_start = tl.minimum(logical_token_start, req_to_token_num_tokens - 1)
        
    offset_req_tok = req_id * req_to_token_stride_req + safe_log_tok_start * req_to_token_stride_token
    first_phys_tok = tl.load(req_to_token_ptr + offset_req_tok)
    phys_page_idx = first_phys_tok // PAGE_SIZE
    
    # Check if phys_page_idx is valid? 
    # Python code: target_pages = phys_pg[...].clamp(0, self.page_k_min[layer_id].shape[0] - 1)
    # We assume output buffer is large enough or we should clamp output index too.
    # But usually physical page index is within bounds of the allocated pool.
    
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    
    # Initialize accumulators
    min_vals = tl.full([BLOCK_DIM], float("inf"), dtype=tl.float32)
    max_vals = tl.full([BLOCK_DIM], float("-inf"), dtype=tl.float32)
    
    # Loop over tokens in the page
    for i in range(PAGE_SIZE):
        log_tok_idx = logical_token_start + i
        
        # Mask calculation from Python:
        # tok_pos < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1))...
        # Basically if log_tok_idx < seq_len
        
        if log_tok_idx < seq_len:
            # Clamp log_tok_idx for req_to_token lookup
            safe_log_tok_idx = tl.minimum(log_tok_idx, req_to_token_num_tokens - 1)
            
            offset_rt = req_id * req_to_token_stride_req + safe_log_tok_idx * req_to_token_stride_token
            phys_tok = tl.load(req_to_token_ptr + offset_rt)
            
            # Clamp phys_tok for k_buffer lookup
            phys_tok = tl.minimum(phys_tok, k_buffer_num_tokens - 1)
            phys_tok = tl.maximum(phys_tok, 0)
            
            # Load key vector
            k_ptr_base = phys_tok * k_buffer_stride_token + head_idx * k_buffer_stride_head
            k_ptrs = k_ptr_base + dim_offsets * k_buffer_stride_dim
            
            keys = tl.load(k_buffer_ptr + k_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
            
            min_vals = tl.minimum(min_vals, keys)
            max_vals = tl.maximum(max_vals, keys)
            
    # Store results
    out_ptr_base = phys_page_idx * page_k_stride_page + head_idx * page_k_stride_head
    out_ptrs = out_ptr_base + dim_offsets * page_k_stride_dim
    
    tl.store(page_k_min_ptr + out_ptrs, min_vals, mask=dim_mask)
    tl.store(page_k_max_ptr + out_ptrs, max_vals, mask=dim_mask)
    
    if head_idx == 0:
        tl.store(page_valid_ptr + phys_page_idx, 1)

@triton.jit
def quest_retrieval_score_kernel(
    scores_ptr,
    reqs_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    queries_ptr,
    # Strides
    scores_stride_req,
    scores_stride_page,
    req_to_token_stride_req,
    req_to_token_stride_token,
    page_k_stride_page,
    page_k_stride_head,
    page_k_stride_dim,
    queries_stride_req,
    queries_stride_head,
    queries_stride_dim,
    # Shapes
    req_to_token_num_tokens,
    page_k_num_pages,
    num_recent_pages,
    # Constants
    PAGE_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    page_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    
    # Mask invalid pages
    if page_idx >= num_pages:
        offset = req_idx * scores_stride_req + page_idx * scores_stride_page
        tl.store(scores_ptr + offset, float("-inf"))
        return
        
    # Mask recent pages
    # recent_start = max(num_pages - num_recent_pages, 0)
    recent_start = num_pages - num_recent_pages
    recent_start = tl.maximum(recent_start, 0)
        
    if page_idx >= recent_start:
        offset = req_idx * scores_stride_req + page_idx * scores_stride_page
        tl.store(scores_ptr + offset, float("-inf"))
        return

    req_id = tl.load(reqs_ptr + req_idx)

    # Get physical page index
    log_tok_idx = page_idx * PAGE_SIZE
    safe_log_tok_idx = tl.minimum(log_tok_idx, req_to_token_num_tokens - 1)
    
    offset_req_tok = req_id * req_to_token_stride_req + safe_log_tok_idx * req_to_token_stride_token
    phys_tok = tl.load(req_to_token_ptr + offset_req_tok)
    phys_page_idx = phys_tok // PAGE_SIZE
    
    phys_page_idx = tl.minimum(phys_page_idx, page_k_num_pages - 1)
    phys_page_idx = tl.maximum(phys_page_idx, 0)

    acc = 0.0
    
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    
    for h in range(HEAD_NUM):
        # Average Query over Group
        q_avg = tl.zeros([BLOCK_DIM], dtype=tl.float32)
        
        for g in range(GROUP_SIZE):
            q_head_idx = h * GROUP_SIZE + g
            q_off = req_idx * queries_stride_req + q_head_idx * queries_stride_head + dim_offsets * queries_stride_dim
            q_val = tl.load(queries_ptr + q_off, mask=dim_mask, other=0.0).to(tl.float32)
            q_avg += q_val
            
        q_avg = q_avg / GROUP_SIZE
        
        # Load K Min/Max
        k_base = phys_page_idx * page_k_stride_page + h * page_k_stride_head
        k_off = k_base + dim_offsets * page_k_stride_dim
        
        k_min = tl.load(page_k_min_ptr + k_off, mask=dim_mask, other=0.0).to(tl.float32)
        k_max = tl.load(page_k_max_ptr + k_off, mask=dim_mask, other=0.0).to(tl.float32)
        
        # Compute term
        term = tl.where(q_avg >= 0, q_avg * k_max, q_avg * k_min)
        acc += tl.sum(term)
        
    # Store score
    offset = req_idx * scores_stride_req + page_idx * scores_stride_page
    tl.store(scores_ptr + offset, acc)


@triton.jit
def quest_combine_indices_kernel(
    topk_indices_ptr,
    out_indices_ptr,
    out_lengths_ptr,
    seq_lens_ptr,
    k_per_req_ptr,
    # Strides
    topk_stride_req,
    out_stride_req,
    # Constants
    num_recent_pages,
    page_size,
    max_k,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Load info
    seq_len = tl.load(seq_lens_ptr + pid)
    num_pages = (seq_len + page_size - 1) // page_size
    
    k_val = tl.load(k_per_req_ptr + pid)
    
    # 1. Copy TopK indices
    offs = tl.arange(0, BLOCK_SIZE)
    
    # We iterate up to max_k (which is size of topk_indices row)
    # But we only copy k_val valid indices
    for i in range(0, max_k, BLOCK_SIZE):
        mask = (offs + i) < k_val
        idx_vals = tl.load(topk_indices_ptr + pid * topk_stride_req + (offs + i), mask=mask, other=-1)
        
        # Store to out_indices
        # Mask ensures we don't write garbage, but we must be careful not to write out of bounds of out_indices?
        # Assuming out_indices is large enough.
        tl.store(out_indices_ptr + pid * out_stride_req + (offs + i), idx_vals, mask=mask)
        
    current_offset = k_val
    
    # 2. Generate Recent indices
    # range [recent_start, num_pages)
    recent_start = num_pages - num_recent_pages
    recent_start = tl.maximum(recent_start, 0)
    
    num_recent = num_pages - recent_start
    
    for i in range(0, num_recent_pages, BLOCK_SIZE): # Iterate max possible recent pages
        # Mask: i < num_recent
        mask = (offs + i) < num_recent
        val = recent_start + offs + i
        
        tl.store(out_indices_ptr + pid * out_stride_req + current_offset + (offs + i), val, mask=mask)
        
    current_offset += num_recent
    
    # 3. Store Length
    tl.store(out_lengths_ptr + pid, current_offset)
    
    # 4. Pad with INT_MAX (using 2147483647)
    # We can pad slightly beyond current_offset to ensure torch.sort pushes them to end
    # Or rely on PyTorch initialization?
    # Python code initializes out_indices with -1.
    # If we want to sort, we need to pad with INT_MAX.
    # We can pad the rest of the row?
    # Getting row size in kernel is hard without passing it.
    # Let's assume the caller initializes with INT_MAX.

def quest_retrieval_score_and_combine_indices_triton(
    bs,
    device,
    seq_lens,
    page_size,
    req_to_token,
    page_k_min,
    page_k_max,
    queries,
    req_pool_indices,
    num_recent_pages,
    fixed_topk_page_cnt,
    sparsity_ratio,
    sparse_mask,
    
):
    # Calculate dimensions
    num_pages = (seq_lens + page_size - 1) // page_size
    max_pages = int(num_pages.max().item())
    if max_pages == 0:
        return (
            torch.full((bs, 0), -1, dtype=torch.int32, device=device),
            torch.zeros(bs, dtype=torch.int32, device=device),
        )
        
    head_num = page_k_min.shape[1]
    head_dim = page_k_min.shape[2]
    BLOCK_DIM = triton.next_power_of_2(head_dim)
    
    scores = torch.empty((bs, max_pages), dtype=torch.float32, device=device)
    grid = (bs, max_pages)
    
    # 2D queries [bs, hidden_dim]
    if queries.dim() == 2:
        bs_q, hidden = queries.shape
        if hidden % head_dim != 0:
                raise ValueError(f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}")
        q_heads = hidden // head_dim
        q = queries.view(bs_q, q_heads, head_dim)
    elif queries.dim() == 3:
        q = queries
    else:
        raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")
    
    q_heads = q.shape[1]
    kv_heads = head_num
    
    GROUP_SIZE = 1
    if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(f"Query heads {q_heads} not divisible by KV heads {kv_heads}")
            GROUP_SIZE = q_heads // kv_heads
            
    q = q.contiguous()
    
    quest_retrieval_score_kernel[grid](
        scores,
        req_pool_indices,
        seq_lens,
        req_to_token,
        page_k_min,
        page_k_max,
        q,
        # Strides
        scores.stride(0),
        scores.stride(1),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_k_min.stride(0),
        page_k_min.stride(1),
        page_k_min.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # Shapes
        req_to_token.shape[1],
        page_k_min.shape[0],
        num_recent_pages,
        # Constants
        PAGE_SIZE=page_size,
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        BLOCK_DIM=BLOCK_DIM,
        GROUP_SIZE=GROUP_SIZE,
    )
    
    # Determine K per request
    recent_start = (num_pages - num_recent_pages).clamp(min=0)
    history_pages = recent_start.clamp(min=1)
    
    if fixed_topk_page_cnt is not None:
            k_target = max(fixed_topk_page_cnt - num_recent_pages, 1)
            k_per_req = torch.full((bs,), k_target, device=device)
    else:
            k_per_req = (history_pages * sparsity_ratio).long().clamp(min=1)
            
    k_per_req = torch.min(k_per_req, history_pages.long())
    k_per_req = k_per_req * sparse_mask.long()
    
    max_k = int(k_per_req.max().item())
    
    if max_k > 0:
        # Perform TopK with max_k
        # scores already has -inf for recent/invalid pages from kernel
        topk_vals, topk_indices = torch.topk(scores, k=min(max_k, max_pages), dim=1, sorted=False)
    else:
        topk_indices = torch.empty((bs, 0), dtype=torch.int64, device=device)
        
    max_out = max_k + num_recent_pages
    pad_value = torch.iinfo(torch.int32).max
    out_indices = torch.full((bs, max_out), pad_value, dtype=torch.int32, device=device)
    out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)
    
    combine_grid = (bs,)
    combine_block_size = 128
    
    quest_combine_indices_kernel[combine_grid](
        topk_indices,
        out_indices,
        out_lengths,
        seq_lens,
        k_per_req,
        # Strides
        topk_indices.stride(0) if topk_indices.numel() > 0 else 0,
        out_indices.stride(0),
        # Constants
        num_recent_pages,
        page_size,
        topk_indices.shape[1] if topk_indices.numel() > 0 else 0,
        BLOCK_SIZE=combine_block_size
    )
    
    out_indices, _ = out_indices.sort(dim=1)
    # Replace INT_MAX with -1 for padding
    out_indices.masked_fill_(out_indices == pad_value, -1)
    
    return out_indices, out_lengths
