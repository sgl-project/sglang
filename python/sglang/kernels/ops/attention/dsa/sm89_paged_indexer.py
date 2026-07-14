from __future__ import annotations

import torch
import triton
import triton.language as tl

_PAGE_SIZE = 64
_NUM_HEADS = 32
_HEAD_DIM = 128
_K_BYTES_PER_PAGE = _PAGE_SIZE * _HEAD_DIM
_SCALE_BYTES_PER_PAGE = _PAGE_SIZE * torch.float32.itemsize
_PACKED_BYTES_PER_PAGE = _K_BYTES_PER_PAGE + _SCALE_BYTES_PER_PAGE
_FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _sm89_paged_fp8_index_logits_kernel(
    q_ptr,
    q_stride_batch,
    q_stride_head,
    q_stride_dim,
    kv_ptr,
    kv_stride_page,
    scale_ptr,
    scale_stride_page,
    weights_ptr,
    weights_stride_batch,
    seq_lens_ptr,
    seq_lens_stride_batch,
    page_table_ptr,
    page_table_stride_batch,
    page_table_stride_page,
    logits_ptr,
    logits_stride_batch,
    num_pages,
    max_seq_len,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    logical_page = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + batch_id * seq_lens_stride_batch)
    physical_page = tl.load(
        page_table_ptr
        + batch_id * page_table_stride_batch
        + logical_page * page_table_stride_page
    )
    page_is_valid = (physical_page >= 0) & (physical_page < num_pages)
    safe_page = tl.where(page_is_valid, physical_page, 0)

    token_offsets = tl.arange(0, PAGE_SIZE)
    logical_tokens = logical_page * PAGE_SIZE + token_offsets
    token_is_valid = (
        page_is_valid & (logical_tokens < seq_len) & (logical_tokens < max_seq_len)
    )

    dim_offsets = tl.arange(0, HEAD_DIM)
    key_offsets = (
        safe_page * kv_stride_page
        + token_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    keys = tl.load(
        kv_ptr + key_offsets,
        mask=token_is_valid[:, None],
        other=0.0,
    ).to(tl.float16)

    head_offsets = tl.arange(0, NUM_HEADS)
    query_offsets = (
        batch_id * q_stride_batch
        + head_offsets[:, None] * q_stride_head
        + dim_offsets[None, :] * q_stride_dim
    )
    query = tl.load(q_ptr + query_offsets).to(tl.float16)
    head_weights = tl.load(weights_ptr + batch_id * weights_stride_batch + head_offsets)
    key_scales = tl.load(
        scale_ptr + safe_page * scale_stride_page + token_offsets,
        mask=token_is_valid,
        other=0.0,
    )

    per_head_dot = tl.dot(keys, tl.trans(query), out_dtype=tl.float32)
    scores = (
        tl.sum(tl.maximum(per_head_dot, 0.0) * head_weights[None, :], axis=1)
        * key_scales
    )
    tl.store(
        logits_ptr + batch_id * logits_stride_batch + logical_tokens,
        scores,
        mask=token_is_valid,
    )


def _validate_inputs(
    q_fp8: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> None:
    if q_fp8.dtype != _FP8_DTYPE:
        raise TypeError("q_fp8 must have dtype float8_e4m3fn")
    if q_fp8.ndim != 4 or q_fp8.shape[1:] != (1, _NUM_HEADS, _HEAD_DIM):
        raise ValueError("q_fp8 shape must be [B, 1, 32, 128]")
    if not q_fp8.is_contiguous():
        raise ValueError("q_fp8 must be contiguous")

    batch_size = q_fp8.shape[0]
    if kv_cache_u8.dtype != torch.uint8:
        raise TypeError("kv_cache_u8 must have dtype uint8")
    if kv_cache_u8.ndim != 2 or kv_cache_u8.shape[1] != _PACKED_BYTES_PER_PAGE:
        raise ValueError("kv_cache_u8 shape must be [num_pages, 8448]")
    if not kv_cache_u8.is_contiguous():
        raise ValueError("kv_cache_u8 must be contiguous")

    if weights.dtype != torch.float32:
        raise TypeError("weights must have dtype float32")
    if weights.shape != (batch_size, _NUM_HEADS):
        raise ValueError("weights shape must be [B, 32]")
    if not weights.is_contiguous():
        raise ValueError("weights must be contiguous")

    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must have dtype int32")
    if seq_lens.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("seq_lens shape must be [B] or [B, 1]")

    if page_table.dtype != torch.int32:
        raise TypeError("page_table must have dtype int32")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table shape must be [B, max_pages]")

    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise TypeError("max_seq_len must be a positive integer")
    if max_seq_len != page_table.shape[1] * _PAGE_SIZE:
        raise ValueError("max_seq_len must equal page_table width times 64")

    tensors = (q_fp8, kv_cache_u8, weights, seq_lens, page_table)
    if q_fp8.device.type != "cuda" or any(
        tensor.device != q_fp8.device for tensor in tensors[1:]
    ):
        raise ValueError("all inputs must be on the same CUDA device")


def sm89_paged_fp8_index_logits(
    q_fp8: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute exact GLM DSA index logits from the raw page-64 FP8 cache."""
    _validate_inputs(
        q_fp8,
        kv_cache_u8,
        weights,
        seq_lens,
        page_table,
        max_seq_len,
    )

    batch_size = q_fp8.shape[0]
    num_pages = kv_cache_u8.shape[0]
    kv_cache_fp8 = kv_cache_u8.view(_FP8_DTYPE)
    kv_scales = kv_cache_u8[:, _K_BYTES_PER_PAGE:].view(torch.float32)
    flat_seq_lens = seq_lens.reshape(batch_size)
    logits = torch.full(
        (batch_size, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )

    grid = (batch_size, triton.cdiv(max_seq_len, _PAGE_SIZE))
    _sm89_paged_fp8_index_logits_kernel[grid](
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(2),
        q_fp8.stride(3),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_scales,
        kv_scales.stride(0),
        weights,
        weights.stride(0),
        flat_seq_lens,
        flat_seq_lens.stride(0),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        logits,
        logits.stride(0),
        num_pages,
        max_seq_len,
        PAGE_SIZE=_PAGE_SIZE,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=4,
    )
    return logits
