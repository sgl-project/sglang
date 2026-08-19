import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv_kernel(
    k_cache,
    v_cache,
    topk_indices,
    req_to_token,
    req_pool_indices,
    gathered_k,
    gathered_v,
    stride_cache_token: tl.constexpr,
    stride_cache_head: tl.constexpr,
    stride_idx_batch: tl.constexpr,
    stride_idx_head: tl.constexpr,
    stride_idx_topk: tl.constexpr,
    stride_r2t_req: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_topk: tl.constexpr,
    kv_heads: tl.constexpr,
    topk_heads: tl.constexpr,
    topk: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    topk_id = pid % topk
    pid = pid // topk
    head_id = pid % topk_heads
    batch_id = pid // topk_heads
    kv_head_id = head_id // (topk_heads // kv_heads)

    logical_pos = tl.load(
        topk_indices
        + batch_id * stride_idx_batch
        + head_id * stride_idx_head
        + topk_id * stride_idx_topk
    )
    offs = tl.arange(0, block_d)
    dim_mask = offs < head_dim
    out_offset = batch_id * stride_out_batch + head_id * stride_out_head + topk_id * stride_out_topk

    if logical_pos >= 0:
        req_idx = tl.load(req_pool_indices + batch_id)
        physical_loc = tl.load(req_to_token + req_idx * stride_r2t_req + logical_pos)
        cache_offset = physical_loc * stride_cache_token + kv_head_id * stride_cache_head
        k_vals = tl.load(k_cache + cache_offset + offs, mask=dim_mask, other=0.0)
        v_vals = tl.load(v_cache + cache_offset + offs, mask=dim_mask, other=0.0)
    else:
        k_vals = tl.zeros([block_d], dtype=tl.bfloat16)
        v_vals = tl.zeros([block_d], dtype=tl.bfloat16)

    tl.store(gathered_k + out_offset + offs, k_vals, mask=dim_mask)
    tl.store(gathered_v + out_offset + offs, v_vals, mask=dim_mask)


def gather_kv_by_indices(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
) -> None:
    batch_size, topk_heads, topk = topk_indices.shape
    kv_heads = k_cache.shape[1]
    assert topk_heads % kv_heads == 0
    head_dim = k_cache.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    _gather_kv_kernel[(batch_size * topk_heads * topk,)](
        k_cache.contiguous(),
        v_cache.contiguous(),
        topk_indices.to(torch.int32).contiguous(),
        req_to_token,
        req_pool_indices.to(torch.int32).contiguous(),
        gathered_k,
        gathered_v,
        stride_cache_token=k_cache.stride(0),
        stride_cache_head=k_cache.stride(1),
        stride_idx_batch=topk_indices.stride(0),
        stride_idx_head=topk_indices.stride(1),
        stride_idx_topk=topk_indices.stride(2),
        stride_r2t_req=req_to_token.stride(0),
        stride_out_batch=gathered_k.stride(0),
        stride_out_head=gathered_k.stride(1),
        stride_out_topk=gathered_k.stride(2),
        kv_heads=kv_heads,
        topk_heads=topk_heads,
        topk=topk,
        head_dim=head_dim,
        block_d=block_d,
    )
