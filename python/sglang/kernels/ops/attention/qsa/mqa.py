"""Triton MQA scoring kernels for the Qwen4-Exp sparse indexer."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_prefill_kernel(
    q,
    k,
    logits,
    row_starts,
    row_ends,
    scale,
    rows,
    keys,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_row,
    stride_k_dim,
    stride_l_row,
    stride_l_col,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    rows_offset = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    keys_offset = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows_offset < rows
    key_mask = keys_offset < keys
    starts = tl.load(row_starts + rows_offset, mask=row_mask, other=keys)
    ends = tl.load(row_ends + rows_offset, mask=row_mask, other=0)
    output_ptrs = (
        logits
        + rows_offset[:, None] * stride_l_row
        + keys_offset[None, :] * stride_l_col
    )
    output_mask = row_mask[:, None] & key_mask[None, :]

    tile_start = block_n * BLOCK_N
    tile_end = tile_start + BLOCK_N
    if tile_start >= tl.max(ends):
        tl.store(output_ptrs, -float("inf"), mask=output_mask)
        return
    if tile_end <= tl.min(starts):
        tl.store(output_ptrs, -float("inf"), mask=output_mask)
        return

    dims = tl.arange(0, BLOCK_D)
    query_columns = tl.arange(0, BLOCK_M * BLOCK_HEADS)
    query_rows = query_columns // BLOCK_HEADS
    query_heads = query_columns % BLOCK_HEADS
    query_ptrs = (
        q
        + (block_m * BLOCK_M + query_rows)[None, :] * stride_q_row
        + query_heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim
    )
    query = tl.load(
        query_ptrs,
        mask=((block_m * BLOCK_M + query_rows)[None, :] < rows)
        & (query_heads[None, :] < HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    key_ptrs = k + keys_offset[:, None] * stride_k_row + dims[None, :] * stride_k_dim
    key = tl.load(
        key_ptrs,
        mask=key_mask[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    scores = tl.dot(key, query, out_dtype=tl.float32)
    scores = tl.reshape(scores, (BLOCK_N, BLOCK_M, BLOCK_HEADS))
    scores = tl.sum(tl.maximum(scores, 0.0), axis=2)
    valid = (
        output_mask
        & (keys_offset[None, :] >= starts[:, None])
        & (keys_offset[None, :] < ends[:, None])
    )
    tl.store(
        output_ptrs,
        tl.where(valid, tl.trans(scores) / scale, -float("inf")),
        mask=output_mask,
    )


@triton.jit
def _qsa_mqa_decode_kernel(
    q,
    k_cache,
    page_table,
    context_lens,
    logits,
    scale,
    max_model_len,
    max_pages,
    stride_q_batch,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_dim,
    stride_pt_batch,
    stride_pt_page,
    stride_l_batch,
    stride_l_token,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    positions = block * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_HEADS)
    context_len = tl.load(context_lens + batch)
    page_columns = positions // PAGE_SIZE
    page_offsets = positions - page_columns * PAGE_SIZE
    position_mask = positions < max_model_len
    valid = position_mask & (positions < context_len) & (page_columns < max_pages)
    pages = tl.load(
        page_table + batch * stride_pt_batch + page_columns * stride_pt_page,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= pages >= 0

    # int64 page ids: the compressed cache is addressed as
    # page * page_size * head_dim, which passes 2**31 elements once the pool is
    # large enough, and Triton keeps the offset in the width of its operands.
    key_ptrs = (
        k_cache
        + pages[:, None] * stride_k_page
        + page_offsets[:, None] * stride_k_token
        + dims[None, :] * stride_k_dim
    )
    key = tl.load(
        key_ptrs,
        mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    query_ptrs = (
        q
        + batch * stride_q_batch
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim
    )
    query = tl.load(
        query_ptrs,
        mask=(dims[:, None] < HEAD_DIM) & (heads[None, :] < HEADS),
        other=0.0,
    )
    scores = tl.dot(key, query, out_dtype=tl.float32)
    scores = tl.sum(tl.maximum(scores, 0.0), axis=1) / scale
    tl.store(
        logits + batch * stride_l_batch + positions * stride_l_token,
        tl.where(valid, scores, -float("inf")),
        mask=position_mask,
    )


def _validate_common(q: torch.Tensor, k: torch.Tensor) -> None:
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("Triton QSA MQA requires CUDA tensors")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError("Triton QSA MQA requires matching BF16 or FP16 inputs")
    if q.ndim != 3 or k.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if q.shape[-1] not in (64, 128, 256):
        raise ValueError(f"unsupported QSA head dimension {q.shape[-1]}")


def triton_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Score packed variable-length queries against packed compressed keys."""

    _validate_common(q, k)
    if k.ndim != 3 or k.shape[1] != 1:
        raise ValueError("QSA MQA keys must have one head")
    rows, heads, head_dim = q.shape
    keys = k.shape[0]
    if row_starts.numel() != rows or row_ends.numel() != rows:
        raise ValueError("QSA prefill row ranges must have one entry per query")
    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    if rows == 0 or keys == 0:
        return logits.fill_(-float("inf"))

    block_m = 16 if rows >= 16 else triton.next_power_of_2(rows)
    block_m = max(4, block_m)
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    block_heads = triton.next_power_of_2(heads)
    _qsa_mqa_prefill_kernel[(triton.cdiv(rows, block_m), triton.cdiv(keys, block_n))](
        q,
        k,
        logits,
        row_starts,
        row_ends,
        float(score_scale or math.sqrt(head_dim)),
        rows,
        keys,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(2),
        logits.stride(0),
        logits.stride(1),
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_HEADS=block_heads,
        num_warps=8,
        num_stages=3,
    )
    return logits


def triton_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Score decode and speculative rows directly from the paged key cache."""

    _validate_common(q, k_cache)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA decode cache must have one key head")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")
    batch, heads, head_dim = q.shape
    logits = torch.empty((batch, max_model_len), dtype=torch.float32, device=q.device)
    if batch == 0 or max_model_len == 0:
        return logits

    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    block_heads = max(16, triton.next_power_of_2(heads))
    _qsa_mqa_decode_kernel[(batch, triton.cdiv(max_model_len, block_n))](
        q,
        k_cache,
        page_table,
        context_lens,
        logits,
        float(score_scale or math.sqrt(head_dim)),
        max_model_len,
        page_table.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        logits.stride(1),
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=k_cache.shape[1],
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_HEADS=block_heads,
        num_warps=4,
        num_stages=3,
    )
    return logits


__all__ = ["triton_qsa_mqa_decode", "triton_qsa_mqa_prefill"]
