import torch
import triton
import triton.language as tl


@triton.jit
def score_copy_kernel(
    topk_indices_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    req_to_token_stride,
    num_kv_heads: tl.constexpr,
    cache_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    head_idx = tl.program_id(1)

    topk_ptr = topk_indices_ptr + bid * num_kv_heads * cache_len + head_idx * cache_len
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < cache_len

    logical_indices = tl.load(topk_ptr + offset, mask=mask, other=0)
    b_offset = tl.load(req_pool_indices_ptr + bid)
    if b_offset < 0:
        return
    kv_pages_ptr = req_to_token_ptr + b_offset * req_to_token_stride
    physical_indices = (
        tl.load(kv_pages_ptr + logical_indices * PAGE_SIZE, mask=mask, other=0)
        // PAGE_SIZE
    )
    tl.store(topk_ptr + offset, physical_indices, mask=mask)


def score_copy(
    topk_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    bs: int,
    page_size: int,
):
    cache_len = topk_indices.shape[2]
    BLOCK_SIZE = triton.next_power_of_2(cache_len)
    num_kv_heads = topk_indices.shape[1]
    grid = (bs, num_kv_heads)
    score_copy_kernel[grid](
        topk_indices,
        req_to_token,
        req_pool_indices,
        req_to_token.stride(0),
        num_kv_heads=num_kv_heads,
        cache_len=cache_len,
        BLOCK_SIZE=BLOCK_SIZE,
        PAGE_SIZE=page_size,
    )
