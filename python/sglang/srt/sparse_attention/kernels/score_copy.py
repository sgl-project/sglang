import torch
import triton
import triton.language as tl

@triton.jit
def score_copy_kernel(
    req_to_token_ptr,
    topk_indices_ptr,
    req_to_token_stride,
    num_kv_heads : tl.constexpr,
    cache_len : tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    ptr = topk_indices_ptr + bid * num_kv_heads * cache_len + head_idx * cache_len
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < cache_len
    page_offsets = tl.load(ptr + offset, mask=mask, other=0)
    topk_indices_vals = tl.load(req_to_token_ptr + bid * req_to_token_stride + page_offsets, mask=mask, other=0)
    tl.store(ptr + offset, topk_indices_vals, mask=mask)
    
    
    
    
    
def score_copy(req_to_token: torch.Tensor, topk_indices: torch.Tensor, bs: int):
    cache_len = topk_indices.shape[2]
    BLOCK_SIZE = triton.next_power_of_2(cache_len)
    num_kv_heads = topk_indices.shape[1]
    grid = (bs, num_kv_heads)
    score_copy_kernel[grid](
        req_to_token,
        topk_indices,
        req_to_token.stride(0),
        num_kv_heads=num_kv_heads,
        cache_len=cache_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )