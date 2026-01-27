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
    req_to_token_num_tokens,  # To clamp
    k_buffer_num_tokens,  # To clamp
    # Constants
    PAGE_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
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

    safe_log_tok_start = tl.minimum(logical_token_start, req_to_token_num_tokens - 1)

    offset_req_tok = (
        req_id * req_to_token_stride_req
        + safe_log_tok_start * req_to_token_stride_token
    )
    first_phys_tok = tl.load(req_to_token_ptr + offset_req_tok)
    phys_page_idx = first_phys_tok // PAGE_SIZE

    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM

    # Initialize accumulators
    min_vals = tl.full([BLOCK_DIM], float("inf"), dtype=tl.float32)
    max_vals = tl.full([BLOCK_DIM], float("-inf"), dtype=tl.float32)

    # Loop over tokens in the page
    for i in range(PAGE_SIZE):
        log_tok_idx = logical_token_start + i

        if log_tok_idx < seq_len:
            safe_log_tok_idx = tl.minimum(log_tok_idx, req_to_token_num_tokens - 1)

            offset_rt = (
                req_id * req_to_token_stride_req
                + safe_log_tok_idx * req_to_token_stride_token
            )
            phys_tok = tl.load(req_to_token_ptr + offset_rt)

            # Clamp phys_tok for k_buffer lookup
            phys_tok = tl.minimum(phys_tok, k_buffer_num_tokens - 1)
            phys_tok = tl.maximum(phys_tok, 0)

            # Load key vector
            k_ptr_base = (
                phys_tok * k_buffer_stride_token + head_idx * k_buffer_stride_head
            )
            k_ptrs = k_ptr_base + dim_offsets * k_buffer_stride_dim

            keys = tl.load(k_buffer_ptr + k_ptrs, mask=dim_mask, other=0.0).to(
                tl.float32
            )

            min_vals = tl.minimum(min_vals, keys)
            max_vals = tl.maximum(max_vals, keys)

    # Store results
    out_ptr_base = phys_page_idx * page_k_stride_page + head_idx * page_k_stride_head
    out_ptrs = out_ptr_base + dim_offsets * page_k_stride_dim

    tl.store(page_k_min_ptr + out_ptrs, min_vals, mask=dim_mask)
    tl.store(page_k_max_ptr + out_ptrs, max_vals, mask=dim_mask)

    if head_idx == 0:
        tl.store(page_valid_ptr + phys_page_idx, 1)
