"""Triton decode kernel for Qwen sparse grouped-query attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_gqa_decode_kernel(
    q,
    k_cache,
    v_cache,
    token_slots,
    out,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    ss_m: tl.constexpr,
    ss_n: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    query_heads = kv_head * GROUP_SIZE + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    query = tl.load(
        q + row * sq_m + query_heads[:, None] * sq_h + dimensions[None, :] * sq_d,
        mask=(tl.arange(0, BLOCK_M) < GROUP_SIZE)[:, None],
        other=0.0,
    )
    query = (query * scale * 1.4426950408889634).to(query.dtype)

    maximum = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    columns = tl.arange(0, BLOCK_N)
    for start in range(0, topk, BLOCK_N):
        positions = start + columns
        slots = tl.load(
            token_slots + row * ss_m + positions * ss_n,
            mask=positions < topk,
            other=-1,
        )
        valid = (positions < topk) & (slots >= 0)
        keys = tl.load(
            k_cache
            + slots[None, :] * sk_n
            + kv_head * sk_h
            + dimensions[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache
            + slots[:, None] * sv_n
            + kv_head * sv_h
            + dimensions[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(query, keys), -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        correction = tl.math.exp2(maximum - next_maximum)
        probabilities = tl.math.exp2(scores - next_maximum[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * correction[:, None],
        )
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        maximum = next_maximum

    result = accumulator / normalizer[:, None]
    tl.store(
        out + row * so_m + query_heads[:, None] * so_h + dimensions[None, :] * so_d,
        result,
        mask=(tl.arange(0, BLOCK_M) < GROUP_SIZE)[:, None],
    )


def sparse_gqa_decode_physical_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run sparse GQA decode directly over physical KV-cache slots."""
    if not all(tensor.is_cuda for tensor in (q, k_cache, v_cache, token_slots)):
        raise ValueError("the sparse GQA Triton decode path requires CUDA tensors")
    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("Q, K, and V must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError("token_slots must have shape [batch, selected_tokens]")
    if q.shape[-1] != k_cache.shape[-1] or q.shape[-1] != v_cache.shape[-1]:
        raise ValueError("Q, K, and V head dimensions must match")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")

    rows, query_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    group_size = query_heads // kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    output = torch.empty_like(q)
    _sparse_gqa_decode_kernel[(rows, kv_heads)](
        q,
        k_cache,
        v_cache,
        token_slots,
        output,
        softmax_scale,
        token_slots.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        token_slots.stride(0),
        token_slots.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=32,
        HEAD_DIM=head_dim,
        num_warps=8,
        num_stages=2,
    )
    return output
