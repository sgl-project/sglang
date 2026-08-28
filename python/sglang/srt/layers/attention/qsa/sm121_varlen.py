"""Packed varlen attention fallback for QSA decode on SM121.

The QSA backend compacts each request's selected KV rows into a packed buffer
and issues one query row per request. FlashAttention-4's CuTe varlen kernel
does not compile for this call shape on SM121, while FlashInfer's TRT-LLM
decode kernel is not numerically safe there. This module implements only the
narrow packed contract needed by QSA so other architectures keep their native
backends.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_one_query_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    query_idx = tl.load(cu_seqlens_q_ptr + sequence_idx)
    kv_start = tl.load(cu_seqlens_k_ptr + sequence_idx)
    kv_end = tl.load(cu_seqlens_k_ptr + sequence_idx + 1)

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    query = tl.load(
        q_ptr + query_idx * q_stride_t + query_head_idx * q_stride_h + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head_idx = query_head_idx // queries_per_kv
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

    for block_start in range(kv_start, kv_end, BLOCK_KV):
        kv_offsets = block_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < kv_end
        key_offsets = kv_offsets * k_stride_t + kv_head_idx * k_stride_h
        keys = tl.load(
            k_ptr + key_offsets[:, None] + dim_offsets[None, :],
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query[None, :] * keys, axis=1) * softmax_scale
        scores = tl.where(kv_mask, scores, -float("inf"))

        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        accumulator *= old_scale

        value_offsets = kv_offsets * v_stride_t + kv_head_idx * v_stride_h
        values = tl.load(
            v_ptr + value_offsets[:, None] + dim_offsets[None, :],
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(probabilities[:, None] * values, axis=0)
        running_max = new_max

    output = accumulator / tl.where(running_sum > 0.0, running_sum, 1.0)
    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + query_head_idx * out_stride_h
        + dim_offsets,
        output.to(out_ptr.dtype.element_ty),
        mask=dim_mask,
    )


def qsa_sm121_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **_: object,
) -> torch.Tensor:
    """Run the one-query packed-varlen attention contract emitted by QSA."""

    del max_seqlen_k, causal
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("SM121 QSA varlen attention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"expected 3-D q/k/v, got {q.shape}/{k.shape}/{v.shape}")
    if max_seqlen_q != 1:
        raise ValueError(f"QSA requires max_seqlen_q=1, got {max_seqlen_q}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported query dtype: {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v dtypes must match, got {q.dtype}/{k.dtype}/{v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on the same CUDA device")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("cu_seqlens_q/k must be on the same CUDA device as q")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q/k must use torch.int32")

    total_queries, num_q_heads, head_dim = q.shape
    _, num_kv_heads, key_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError(f"k/v shapes must match, got {k.shape}/{v.shape}")
    if key_head_dim != head_dim:
        raise ValueError(
            f"q/k head dimensions must match, got {head_dim}/{key_head_dim}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"QSA GQA requires q heads divisible by kv heads, got "
            f"{num_q_heads}/{num_kv_heads}"
        )
    if cu_seqlens_q.numel() != total_queries + 1:
        raise ValueError("cu_seqlens_q must contain one entry per query plus the end")
    if cu_seqlens_k.numel() != total_queries + 1:
        raise ValueError("cu_seqlens_k must contain one entry per query plus the end")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    padded_head_dim = triton.next_power_of_2(max(head_dim, 16))
    _qsa_one_query_varlen_kernel[(total_queries, num_q_heads)](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        BLOCK_KV=64,
        q_stride_t=q.stride(0),
        q_stride_h=q.stride(1),
        k_stride_t=k.stride(0),
        k_stride_h=k.stride(1),
        v_stride_t=v.stride(0),
        v_stride_h=v.stride(1),
        out_stride_t=output.stride(0),
        out_stride_h=output.stride(1),
        num_warps=4,
    )
    return output
