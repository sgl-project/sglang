import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_P': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_P': 32}, num_warps=2, num_stages=4), 
        triton.Config({'BLOCK_SIZE_P': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_P': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_P': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_P': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_P': 256}, num_warps=8, num_stages=2),
    ],
    key=['NUM_Q_HEADS', 'NUM_KV_HEADS', 'HEAD_DIM'],
)
@triton.jit
def _compute_average_score_kernel(
    Q, K, Out, Stream_indices_page,kv_pages_per_seq,kv_pages_num_per_seq,kv_pages_per_seq_max,
    # Strides
    q_stride_b, q_stride_h,
    k_stride_p, k_stride_h,
    scores_stride_b, scores_stride_h,
    # Dimensions
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    num_pages: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Streaming info
    STREAM_LEN: tl.constexpr,
    ACTUAL_STREAM_LEN: tl.constexpr,
    # Block sizes and padding
    BLOCK_SIZE_P: tl.constexpr, 
    PADDED_GROUP_SIZE: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    MAX_NUM_TOKEN_PAGES: tl.constexpr,
    PADDED_MAX_NUM_TOKEN_PAGES: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    pid_block = tl.program_id(2)

    max_page_index = tl.load(kv_pages_per_seq_max + bid)
    max_page_block = tl.cdiv(max_page_index, BLOCK_SIZE_P)
    if pid_block >= max_page_block:
        page_offsets = pid_block * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
        out_ptrs = (Out + bid * scores_stride_b + 
                    kv_hid * scores_stride_h + 
                    page_offsets)
        store_mask = (kv_hid < NUM_KV_HEADS) & (page_offsets < num_pages)
        neg_inf_scores = tl.full([BLOCK_SIZE_P], -float('inf'), dtype=tl.float32)
        tl.store(out_ptrs, neg_inf_scores, mask=store_mask)
        return

    q_head_start_idx = kv_hid * GROUP_SIZE
    q_head_offsets = tl.arange(0, PADDED_GROUP_SIZE)
    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    q_ptrs = (Q + bid * q_stride_b + 
              (q_head_start_idx + q_head_offsets[:, None]) * q_stride_h + 
              dim_offsets[None, :])
    q_load_mask = (q_head_offsets[:, None] < GROUP_SIZE) & \
                  (dim_offsets[None, :] < HEAD_DIM)
    q_matrix = tl.load(q_ptrs, mask=q_load_mask, other=0.0)

    page_offsets = pid_block * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    k_ptrs = (K + page_offsets[:, None] * k_stride_p + 
              kv_hid * k_stride_h + 
              dim_offsets[None, :])
    k_load_mask = (page_offsets[:, None] < num_pages) & \
                  (dim_offsets[None, :] < HEAD_DIM)
    k_matrix = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

    scores_matrix = tl.dot(q_matrix, tl.trans(k_matrix), out_dtype=tl.float32, allow_tf32=False)
    final_scores = tl.sum(scores_matrix, axis=0)

    num_valid_pages = tl.load(kv_pages_num_per_seq + bid)
    token_base_ptr = kv_pages_per_seq + bid * MAX_NUM_TOKEN_PAGES
    token_page_offsets = tl.arange(0, PADDED_MAX_NUM_TOKEN_PAGES)
    valid_page_ids = tl.load(token_base_ptr + token_page_offsets, 
                            mask=(token_page_offsets < MAX_NUM_TOKEN_PAGES) & (token_page_offsets < num_valid_pages), other=-1)
    
    is_in_valid_pages_matrix = (page_offsets[:, None] == valid_page_ids[None, :])
    is_in_valid_pages = tl.sum(is_in_valid_pages_matrix.to(tl.int32), axis=1) > 0
    is_not_zero = page_offsets != 0
    is_page_valid = is_in_valid_pages & is_not_zero

    
    if ACTUAL_STREAM_LEN > 0:
        stream_base_ptr = Stream_indices_page + bid * ACTUAL_STREAM_LEN
        stream_offsets = tl.arange(0, STREAM_LEN)
        stream_page_ids = tl.load(stream_base_ptr + stream_offsets, mask=stream_offsets < ACTUAL_STREAM_LEN, other=-1)
        is_in_stream_matrix = (page_offsets[:, None] == stream_page_ids[None, :])
        is_in_stream = tl.sum(is_in_stream_matrix.to(tl.int32), axis=1) > 0
        is_page_valid = is_page_valid & (~is_in_stream)

    out_ptrs = (Out + bid * scores_stride_b + 
                kv_hid * scores_stride_h + 
                page_offsets)
    store_mask = (kv_hid < NUM_KV_HEADS) & (page_offsets < num_pages)
    final_scores_with_inf = tl.where(is_page_valid, final_scores, -float('inf'))
    tl.store(out_ptrs, final_scores_with_inf, mask=store_mask)


def compute_average_score(q: torch.Tensor,
                           k: torch.Tensor,
                           out: torch.Tensor,
                           stream_indices_page: torch.Tensor = None,
                           kv_pages_per_seq: torch.Tensor = None,
                           kv_pages_num_per_seq: torch.Tensor = None,
                           ):

    bs = q.shape[0]
    num_pages = k.shape[0]  
    NUM_KV_HEADS = k.shape[2]
    HEAD_DIM = k.shape[-1]
    
    q = q.view(bs, -1, HEAD_DIM)
    k = k.squeeze(1)
    
    NUM_Q_HEADS = q.shape[1]
    GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS

    PADDED_GROUP_SIZE = max(16, triton.next_power_of_2(GROUP_SIZE))
    PADDED_HEAD_DIM = max(16, triton.next_power_of_2(HEAD_DIM))

    actual_stream_len = stream_indices_page.shape[1]
    stream_len = triton.next_power_of_2(actual_stream_len) if actual_stream_len > 0 else 0

    max_num_token_pages = kv_pages_per_seq.shape[1]
    padded_max_num_token_pages = triton.next_power_of_2(max_num_token_pages)

    # NOTE: for early exit
    kv_pages_per_seq_max = kv_pages_per_seq.max(dim=-1).values
    
    grid = lambda meta: (
        bs, 
        NUM_KV_HEADS, 
        triton.cdiv(num_pages, meta['BLOCK_SIZE_P'])
    )
    
    _compute_average_score_kernel[grid](
        q, k, out, stream_indices_page,kv_pages_per_seq, kv_pages_num_per_seq, kv_pages_per_seq_max,
        # Strides
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        out.stride(0), out.stride(1),
        # Dimensions
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        num_pages=num_pages,
        GROUP_SIZE=GROUP_SIZE, 
        # Streaming info
        STREAM_LEN=stream_len,
        ACTUAL_STREAM_LEN=actual_stream_len,
        # Padding info
        PADDED_GROUP_SIZE=PADDED_GROUP_SIZE,
        PADDED_HEAD_DIM=PADDED_HEAD_DIM,
        MAX_NUM_TOKEN_PAGES=max_num_token_pages,
        PADDED_MAX_NUM_TOKEN_PAGES=padded_max_num_token_pages,
        
    )
