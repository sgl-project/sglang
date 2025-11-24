import torch
import triton
import triton.language as tl


@triton.jit
def moving_average_kernel(
    query: tl.tensor,
    new_query: tl.tensor,
    cur_req_pool_indices: tl.tensor,
    prev_req_pool_indices: tl.tensor,
    moving_average_factor: tl.constexpr,
    pre_bs: int,
    q_stride_b: tl.constexpr,
    new_q_stride_b: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_m = tl.program_id(1)
    offs_m = start_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_ptr = query + offs_m + pid * q_stride_b
    new_q_ptr = new_query + offs_m + pid * new_q_stride_b
    factor = tl.full([], moving_average_factor, dtype=query.dtype.element_ty)
    cur_req_pool_idx = tl.load(cur_req_pool_indices + pid)

    is_in_pre_bs = False

    for i in range(pre_bs):
        pre_req_idx = tl.load(prev_req_pool_indices + i)
        match_found = (pre_req_idx != -1) & (pre_req_idx == cur_req_pool_idx)
        is_in_pre_bs = is_in_pre_bs | match_found

    if is_in_pre_bs:
        update_q = tl.load(q_ptr) * factor + tl.load(new_q_ptr) * (1 - factor)
    else:
        update_q = tl.load(new_q_ptr)
    tl.store(q_ptr, update_q)


def moving_average_update(
    query: torch.Tensor,
    new_query: torch.Tensor,
    cur_req_pool_indices: torch.Tensor,
    prev_req_pool_indices: torch.Tensor,
    moving_average_factor: float,
):
    pre_bs = prev_req_pool_indices.shape[0]
    cur_bs = cur_req_pool_indices.shape[0]
    BLOCK_SIZE = 64
    assert query.shape[1] % BLOCK_SIZE == 0
    grid = (cur_bs, query.shape[1] // BLOCK_SIZE)
    moving_average_kernel[grid](
        query,
        new_query,
        cur_req_pool_indices,
        prev_req_pool_indices,
        moving_average_factor,
        pre_bs,
        query.stride(0),
        new_query.stride(0),
        BLOCK_SIZE,
    )
