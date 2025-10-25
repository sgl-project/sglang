"""
Triton kernels for V2 sparse ReqToTokenPool.

These kernels work with both:
- V1: Dense tensor (via materialize to get pointers)
- V2: Sparse dict of tensors (direct pointers)

Key difference: Instead of req_to_token_ptr + stride * req_pool_index,
we use req_to_token_ptrs[pid] to get per-request tensor pointers directly.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def write_req_to_token_pool_triton_v2(
    req_to_token_ptrs,  # List of pointers, one per request [batch_size]
    prefix_tensors,  # List of pointers to prefix tensors [batch_size]
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
):
    """
    Write to sparse req_to_token pool using per-request pointers.

    For V1: req_to_token_ptrs[i] = req_to_token_base + req_pool_idx[i] * stride
    For V2: req_to_token_ptrs[i] = pool[req_pool_idx[i]].data_ptr()
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    # Load the pointer to this request's tensor
    req_ptr = tl.load(req_to_token_ptrs + pid).to(tl.pointer_type(tl.int32))
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # Write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(req_ptr + offset, value, mask=mask)

    # Write new tokens
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(req_ptr + offset + pre_len, value, mask=mask)


@triton.jit
def create_flashinfer_kv_indices_triton_v2(
    req_to_token_ptrs,  # List of pointers, one per request [batch_size]
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
):
    """
    Create flashinfer kv_indices using sparse req_to_token pool.

    For V1: req_to_token_ptrs[i] = req_to_token_base + req_pool_idx[i] * stride
    For V2: req_to_token_ptrs[i] = pool[req_pool_idx[i]].data_ptr()
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # Load the pointer to this request's tensor
    req_ptr = tl.load(req_to_token_ptrs + pid).to(tl.pointer_type(tl.int32))
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(req_ptr + kv_start + offset, mask=mask)
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def get_req_to_token_ptrs(
    req_to_token_pool, req_pool_indices: torch.Tensor, device: str
):
    """
    Get per-request tensor pointers for both V1 and V2.

    V1: Dense tensor - compute pointers using base + stride * index
    V2: Sparse dict - get pointers directly from dict
    """
    # Check if V2 (no req_to_token attribute)
    if not hasattr(req_to_token_pool, "req_to_token"):
        assert req_pool_indices.is_cpu
        # V2: Sparse storage
        import torch

        ptrs = []
        for idx in req_pool_indices.tolist():
            ptrs.append(req_to_token_pool.pool[idx].data_ptr())
        return torch.tensor(ptrs, dtype=torch.int64, device=device)
    else:
        # V1: Dense tensor
        import torch

        base_ptr = req_to_token_pool.req_to_token.data_ptr()
        stride = (
            req_to_token_pool.req_to_token.stride(0)
            * req_to_token_pool.req_to_token.element_size()
        )
        ptrs = []
        for idx in req_pool_indices.tolist():
            ptrs.append(base_ptr + idx * stride)
        return torch.tensor(ptrs, dtype=torch.int64, device=device)
