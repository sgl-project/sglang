import torch
import triton
import triton.language as tl

@triton.jit
def score_copy_kernel(
    topk_indices_ptr,
    kv_pages_per_seq_ptr,
    kv_pages_per_seq_stride,
    num_kv_heads : tl.constexpr,
    cache_len : tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    topk_ptr = topk_indices_ptr + bid * num_kv_heads * cache_len + head_idx * cache_len
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < cache_len
    
    logical_indices = tl.load(topk_ptr + offset, mask=mask, other=0)
    
    kv_pages_ptr = kv_pages_per_seq_ptr + bid * kv_pages_per_seq_stride
    physical_indices = tl.load(kv_pages_ptr + logical_indices, mask=mask, other=0)
    
    tl.store(topk_ptr + offset, physical_indices, mask=mask)
    
def score_copy(topk_indices: torch.Tensor, kv_pages_per_seq: torch.Tensor, bs: int):
    cache_len = topk_indices.shape[2]
    BLOCK_SIZE = triton.next_power_of_2(cache_len)
    num_kv_heads = topk_indices.shape[1]
    grid = (bs, num_kv_heads)
    score_copy_kernel[grid](
        topk_indices,
        kv_pages_per_seq,
        kv_pages_per_seq.stride(0),
        num_kv_heads=num_kv_heads,
        cache_len=cache_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )