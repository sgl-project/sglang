"""Sparse attention utilities for MiniCPM models.

This module provides sparse attention helpers and utilities for MiniCPM models,
combining both backend-agnostic sparse attention components and kernel utilities.
"""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

import triton
from sgl_kernel import infllmv2_attn_stage1, max_pooling_1d_varlen

from sglang.srt.layers.attention.minicpm.sparse_kernels import (
    compress_k_complete_kernel_new,
)
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool


def batched_gather(a, lengths_cpu, select):
    offsets = [0, *accumulate(map(int, lengths_cpu))]
    return torch.cat([a[offsets[i] : offsets[i + 1]] for i in select])


def compress_k_core_new(
    full_compressed_k,  # output
    batch,
    key_cache,
    token_table,
    compressed_k_table,
    cu_new_k_token_nums,
    history_compress_k_token_nums,
    cu_total_compress_k_token_nums,
    kernel_size,
    kernel_stride,
    max_context_length,
):
    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]

    # ==============================================================================
    # BUFFER ALLOCATION
    # ==============================================================================

    # Use provided explicit parameters for buffer allocation
    # max_chunks_per_seq is already the maximum possible chunks for any sequence
    # given max_context_length, kernel_size, and kernel_stride
    max_chunks_per_seq = max(0, (max_context_length - kernel_size) // kernel_stride + 1)

    # ==============================================================================
    # Launch kernel for ALL chunks (history + new)
    # ==============================================================================
    # Grid: (batch, max_chunks_per_seq, head_num_k)
    # - chunk_in_seq in [0, history_compress): process HISTORY chunks
    # - chunk_in_seq in [history_compress, total_chunks_in_seq): process NEW chunks
    #
    # max_chunks_per_seq is already the maximum possible chunks for any sequence,
    # so it's sufficient for both history and new chunks.
    #
    # All operations are in a single kernel, CUDA graph compatible.

    # Limit grid size to avoid too many thread blocks
    # If max_chunks_per_seq > max_grid_chunks, kernel will loop to handle remaining chunks
    MAX_GRID_CHUNKS = 1024  # Adjustable limit for grid dimension
    max_grid_chunks = min(max_chunks_per_seq, MAX_GRID_CHUNKS)

    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    # Grid size is now limited, kernel uses loop to handle all chunks
    grid = (batch, max_grid_chunks, head_num_k)

    compress_k_complete_kernel_new[grid](
        key_cache,
        token_table,
        cu_new_k_token_nums,
        history_compress_k_token_nums,
        compressed_k_table,
        cu_total_compress_k_token_nums,
        full_compressed_k,
        batch,
        max_chunks_per_seq,
        token_table.shape[1],
        compressed_k_table.shape[1],
        head_num_k,
        head_dim,
        kernel_size,
        kernel_stride,
        BLOCK_SIZE,
        max_grid_chunks,  # Pass the limit to kernel for loop control
    )

    return


def get_compress_k_v2(
    layer,
    forward_batch,
    metadata: MiniCPMSparseMetadata,
    full_compressed_k1,
    full_compressed_k2,
    max_context_length,
    k1_kernel_size,
    k1_kernel_stride,
    k2_kernel_size,
    k2_kernel_stride,
):
    batch = len(forward_batch.req_pool_indices)
    key_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)

    for full_compressed_k, level, kernel_size, kernel_stride in (
        (
            full_compressed_k1,
            metadata.k1,
            k1_kernel_size,
            k1_kernel_stride,
        ),
        (
            full_compressed_k2,
            metadata.k2,
            k2_kernel_size,
            k2_kernel_stride,
        ),
    ):
        compress_k_core_new(
            full_compressed_k,
            batch,
            key_cache,
            metadata.base.page_table,
            level.table,
            level.cu_new_token_nums,
            level.history_compress_token_nums,
            level.cu_total_compress_token_nums,
            kernel_size,
            kernel_stride,
            max_context_length,
        )


def allocate_and_compress_keys(
    layer,
    forward_batch,
    metadata: MiniCPMSparseMetadata,
    k1_token_nums: int,
    k2_token_nums: int,
    k1_kernel_size: int,
    k1_kernel_stride: int,
    k2_kernel_size: int,
    k2_kernel_stride: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
    max_context_length: int = 32768,
):
    """Allocate compressed key tensors and run compression.

    Args:
        layer: Model layer with head configuration
        forward_batch: Forward batch info
        metadata: MiniCPM sparse metadata
        k1_token_nums: Number of k1 tokens to allocate
        k2_token_nums: Number of k2 tokens to allocate
        k1_kernel_size: K1 compression window
        k1_kernel_stride: K1 compression stride
        k2_kernel_size: K2 compression window
        k2_kernel_stride: K2 compression stride
        dtype: Tensor data type (default: bfloat16)
        device: Tensor device (default: layer device)
        max_context_length: Maximum context length for the model (default: 32768)

    Returns:
        Tuple of (full_compressed_k1, full_compressed_k2)
    """
    if device is None:
        device = forward_batch.input_ids.device

    full_compressed_k1 = torch.full(
        (k1_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )
    full_compressed_k2 = torch.full(
        (k2_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )

    get_compress_k_v2(
        layer,
        forward_batch,
        metadata,
        full_compressed_k1,
        full_compressed_k2,
        max_context_length=max_context_length,
        k1_kernel_size=k1_kernel_size,
        k1_kernel_stride=k1_kernel_stride,
        k2_kernel_size=k2_kernel_size,
        k2_kernel_stride=k2_kernel_stride,
    )

    return full_compressed_k1, full_compressed_k2


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_context_len: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens: Optional[torch.Tensor] = None,
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None,
    max_seqlen_q_adjusted: Optional[int] = None,
) -> torch.Tensor:
    """Compressed attention computation for sparse attention.

    Computes attention scores between query and compressed keys (k and k2),
    then performs max pooling and selects top-k blocks.

    Args:
        q: Query tensor, shape (total_q_len, num_heads, head_dim)
        k: Compressed key tensor k1, shape (total_k_len, num_heads, head_dim)
        k2: Compressed key tensor k2, shape (total_k_len, num_heads, head_dim)
        kernel_stride: Stride of compression kernel
        block_size: Size of attention blocks
        topk: Number of top blocks to select
        cu_seqlens_q: Cumulative sequence lengths for query, shape (batch_size + 1)
        cu_seqlens_k: Cumulative sequence lengths for k, shape (batch_size + 1)
        cu_seqlens_k2: Cumulative sequence lengths for k2, shape (batch_size + 1)
        max_seqlen_q: Maximum sequence length in query
        init_blocks: Number of initial blocks to always attend to
        local_blocks: Number of local blocks to consider
        cache_lens: Cache lengths for each batch (optional)
        cu_seqlens_q_adjusted: Adjusted cumulative sequence lengths for query (for stage1 optimization)
        max_seqlen_q_adjusted: Adjusted maximum sequence length for query (for stage1 optimization)

    Returns:
        Top-k block indices, shape (num_heads, total_q_len, topk)
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        is_prefilling = max_seqlen_q > 1

        if is_prefilling:
            if cache_lens is None:
                cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q_adjusted,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,
            max_seqlen_q=max_seqlen_q_adjusted,
            max_seqlen_k=max_context_len // kernel_stride,
            causal=is_prefilling,
        )

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_context_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )

        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


def compressed_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    topk: int,
    kernel_topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cache_lens=None,
    fused_kernel=None,
    max_cache_len=-1,
) -> torch.Tensor:
    """
    使用 tilelang online topk kernel 计算 compressed attention topk indices
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        total_q_len = q.shape[0]
        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]

        num_heads = q.shape[1]
        groups = num_heads // num_kv_heads
        q_kernel = q.view(total_q_len, num_kv_heads, groups, head_dim)
        q_kernel = (
            q_kernel.transpose(1, 2)
            .reshape(total_q_len * groups, num_kv_heads, head_dim)
            .contiguous()
        )

        k_kernel = k.contiguous()

        pooled_k_len = (max_cache_len + block_size - 1) // block_size

        assert fused_kernel is not None, "fused_kernel is not initialized"

        # Compute actual output topk (same as original: min(topk, num_blocks))
        output_topk = min(topk, pooled_k_len)

        # Allocate output tensors
        topk_indices = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        topk_values = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        if cache_lens is None:
            cache_lens_tensor = torch.zeros(
                batch_size, dtype=torch.int32, device=q.device
            )
        else:
            cache_lens_tensor = cache_lens.to(torch.int32)

        fused_kernel(
            q_kernel,
            k_kernel,
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens_tensor,
            topk_indices,
            topk_values,
        )

        # Note: q_idx masking is handled inside the kernel via causal_mask
        # which sets scores to -1e9 for K blocks beyond the causal boundary.
        # These blocks won't be selected in topk due to their low scores.

        # Sort with -1 values at the end (match original behavior)
        # Replace -1 with large value, sort, then replace back
        large_val = pooled_k_len + 1000  # Any value larger than max valid index
        topk_for_sort = topk_indices.clone()
        topk_for_sort[topk_for_sort == -1] = large_val
        topk_idx = topk_for_sort.sort(-1).values
        topk_idx[topk_idx == large_val] = -1

        # Truncate to output_topk (same as original: min(topk, num_blocks))
        topk_idx = topk_idx[:, :, :output_topk].contiguous()

        return topk_idx


class CompressionLevelMetadata(msgspec.Struct):
    """Metadata for a single compression level (k1 or k2).

    This struct groups all metadata fields for one compression level,
    reducing duplication and making the code more maintainable.
    """

    # Cumulative sequence lengths for compressed cache
    cu_seqlens: Optional[torch.Tensor] = None
    cu_seqlens_cpu: Optional[list[int]] = None

    # Token mapping table (request pool indices -> compressed cache tokens)
    table: Optional[torch.Tensor] = None

    # Compressed cache metadata
    history_compress_token_nums: Optional[torch.Tensor] = None
    cu_new_token_nums: Optional[torch.Tensor] = None
    cu_total_compress_token_nums: Optional[torch.Tensor] = None


class RepeatedSegmentLayout(msgspec.Struct):
    """Per-row gather of each request's compressed segment:
    ``index`` gathers the repeated segments from a flat compressed cache;
    ``cu_seqlens`` bounds them per row."""

    index: torch.Tensor
    cu_seqlens: torch.Tensor


class MiniCPMSparseMetadata(msgspec.Struct):
    """Per-forward sparse-attention metadata: the FlashAttention base geometry
    plus the row, selection and compression fields the MiniCPM backend plans
    around it."""

    base: FlashAttentionMetadata
    # Per-request compressed-level metadata (cu_seqlens, token tables).
    k1: Optional[CompressionLevelMetadata] = None
    k2: Optional[CompressionLevelMetadata] = None
    # Verify-round expansion of the levels: one segment copy per draft token.
    k1_repeat: Optional[RepeatedSegmentLayout] = None
    k2_repeat: Optional[RepeatedSegmentLayout] = None
    # Requests taking the sparse path this prefill,
    # and the sparse_page_table rows they occupy.
    sparse_bs_list: Optional[list[int]] = None
    sparse_idx: Optional[list[int]] = None
    # Dense prefill spans as (batch_idx, sparse_page_table row_start,
    # query_group_start, query_len).
    dense_layout: Optional[list[tuple[int, int, int, int]]] = None
    # Dense rows as (sparse_page_table row_start, batch_idx, kv_len);
    # the row_start heads a head_group_num-long run of rows.
    dense_rows: Optional[list[tuple[int, int, int]]] = None
    # [rows, 1] bool, True on the decode rows the per-layer sparse gather
    # may overwrite (kv >= dense_len); dense rows keep their planned pages.
    sparse_row_mask: Optional[torch.Tensor] = None
    # Per-sparse-request kv lengths: the base cache lengths for decode/verify,
    # the sparse sub-batch lengths for prefill.
    seqlen_k_sparse_bs_tensor: Optional[torch.Tensor] = None
    # Token row -> request index, and the row's 1-based causal position
    # (the attention gate is key_pos < token_pos_in_bs).
    token_to_bs: Optional[torch.Tensor] = None
    token_pos_in_bs: Optional[torch.Tensor] = None
    # Head-group-interleaved block table for the sparse rows.
    sparse_page_table: Optional[torch.Tensor] = None
    # Chain-verify dense-row overwrite (_build_dense_verify_overwrite): prefix pages
    # and the sparse_page_table entries they replace; the verify graphs bind it static.
    verify_dense_rows: Optional[torch.Tensor] = None
    verify_dense_mask: Optional[torch.Tensor] = None
    # Varlen geometry of the sparse row attention (decode or verify rows).
    sparse_cache_seqlens_int32: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q: Optional[torch.Tensor] = None
    sparse_cu_seqlens_k: Optional[torch.Tensor] = None
    sparse_max_seq_len_q: int = 1
    # Stage-1 cache lengths excluding the current token (decode/verify topk gate).
    cache_seqlens_int32_stage1: Optional[torch.Tensor] = None
    # Stage-1 query geometry expanded to one row per (token, head group).
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: int = 1
    topk_cu_seqlens_q: Optional[torch.Tensor] = None
    topk_cu_seqlens_k: Optional[torch.Tensor] = None
    topk_max_seqlen_q: int = 1
    topk_max_seqlen_k: int = 1


def _assign_row_metadata(
    metadata: MiniCPMSparseMetadata,
    *,
    sparse_cache_seqlens_int32: torch.Tensor,
    sparse_cu_seqlens_q: torch.Tensor,
    sparse_cu_seqlens_k: torch.Tensor,
    sparse_page_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    cache_seqlens_int32_stage1: torch.Tensor,
    cu_seqlens_q_adjusted: torch.Tensor,
    max_seqlen_q_adjusted: int,
    topk_cu_seqlens_q: torch.Tensor,
    topk_cu_seqlens_k: torch.Tensor,
    topk_max_seqlen_q: int,
    topk_max_seqlen_k: int,
) -> None:
    metadata.sparse_cache_seqlens_int32 = sparse_cache_seqlens_int32
    metadata.sparse_cu_seqlens_q = sparse_cu_seqlens_q
    metadata.sparse_cu_seqlens_k = sparse_cu_seqlens_k
    metadata.sparse_page_table = sparse_page_table
    metadata.token_to_bs = token_to_bs
    metadata.token_pos_in_bs = token_pos_in_bs
    metadata.seqlen_k_sparse_bs_tensor = metadata.base.cache_seqlens_int32
    metadata.cache_seqlens_int32_stage1 = cache_seqlens_int32_stage1
    metadata.cu_seqlens_q_adjusted = cu_seqlens_q_adjusted
    metadata.max_seqlen_q_adjusted = max_seqlen_q_adjusted
    metadata.topk_cu_seqlens_q = topk_cu_seqlens_q
    metadata.topk_cu_seqlens_k = topk_cu_seqlens_k
    metadata.topk_max_seqlen_q = topk_max_seqlen_q
    metadata.topk_max_seqlen_k = topk_max_seqlen_k


def _compute_single_compression_metadata(
    seq_lens_cpu: torch.Tensor,
    token_nums: torch.Tensor,
    history_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_sparse_token: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
) -> CompressionLevelMetadata:
    seqlen_cpu = torch.clamp(
        (seq_lens_cpu - kernel_size) // kernel_stride + 1,
        min=0,
    )

    cu_seqlens_cpu = F.pad(
        torch.cumsum(seqlen_cpu, dim=0, dtype=torch.int32), (1, 0)
    ).tolist()
    cu_seqlens = F.pad(
        torch.cumsum(seqlen_cpu.to(device=token_nums.device), dim=0, dtype=torch.int32),
        (1, 0),
    )
    token_table = req_to_sparse_token[req_pool_indices]
    history_compress_token_nums = torch.clamp(
        (history_lens - kernel_size) // kernel_stride + 1,
        min=0,
    )
    new_token_nums = token_nums - history_compress_token_nums * kernel_stride
    cu_new_token_nums = F.pad(
        torch.cumsum(new_token_nums, dim=0, dtype=torch.int32), (1, 0)
    )
    new_compress_token_nums = torch.clamp(
        (new_token_nums - kernel_size) // kernel_stride + 1,
        min=0,
    )
    total_compress_token_nums = history_compress_token_nums + new_compress_token_nums
    cu_total_compress_token_nums = F.pad(
        torch.cumsum(total_compress_token_nums, dim=0, dtype=torch.int32), (1, 0)
    )

    return CompressionLevelMetadata(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        table=token_table,
        history_compress_token_nums=history_compress_token_nums,
        cu_new_token_nums=cu_new_token_nums,
        cu_total_compress_token_nums=cu_total_compress_token_nums,
    )


def _build_k1_k2_compression_metadata(
    req_pool_indices: torch.Tensor,
    base_metadata: FlashAttentionMetadata,
    req_to_sparse_k1_token: torch.Tensor,
    req_to_sparse_k2_token: torch.Tensor,
    k1_kernel_size: int,
    k1_kernel_stride: int,
    k2_kernel_size: int,
    k2_kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> tuple[CompressionLevelMetadata, CompressionLevelMetadata]:
    seq_lens_cpu = torch.as_tensor(
        seq_lens_cpu,
        dtype=base_metadata.cu_seqlens_q.dtype,
        device="cpu",
    )
    bs = seq_lens_cpu.numel()
    token_nums = (
        base_metadata.cu_seqlens_k[1 : bs + 1] - base_metadata.cu_seqlens_k[:bs]
    )
    input_lens = cu_seqlens_q[1 : bs + 1] - cu_seqlens_q[:bs]
    history_lens = token_nums - input_lens

    return tuple(
        _compute_single_compression_metadata(
            seq_lens_cpu,
            token_nums,
            history_lens,
            req_pool_indices,
            req_to_sparse_token,
            kernel_size,
            kernel_stride,
        )
        for req_to_sparse_token, kernel_size, kernel_stride in (
            (req_to_sparse_k1_token, k1_kernel_size, k1_kernel_stride),
            (req_to_sparse_k2_token, k2_kernel_size, k2_kernel_stride),
        )
    )


def _get_sparse_cache_lens(
    seq_lens: torch.Tensor,
    sparse_capacity: int,
    block_size: int,
) -> torch.Tensor:
    remainder = seq_lens % block_size
    sparse_lens = torch.where(
        remainder == 0,
        sparse_capacity,
        sparse_capacity - block_size + remainder,
    )
    return torch.where(seq_lens <= sparse_capacity, seq_lens, sparse_lens)


def _plan_dense_rows(
    seq_lens_cpu: torch.Tensor,
    *,
    dense_len: int,
    head_group_num: int,
) -> list[tuple[int, int, int]]:
    return [
        (batch_idx * head_group_num, batch_idx, kv_len)
        for batch_idx, kv_len in enumerate(seq_lens_cpu.tolist())
        if kv_len < dense_len
    ]


def _build_sparse_verify_rows(
    seq_lens: torch.Tensor,
    *,
    head_group_num: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = seq_lens.device
    sparse_capacity = sparse_topk * block_size
    offsets = torch.arange(1, num_draft_tokens + 1, dtype=torch.int32, device=device)
    token_pos = seq_lens.to(torch.int32).repeat_interleave(
        num_draft_tokens
    ) + offsets.repeat(seq_lens.numel())
    row_lens = torch.where(
        token_pos < dense_len,
        token_pos,
        _get_sparse_cache_lens(
            token_pos, sparse_capacity=sparse_capacity, block_size=block_size
        ),
    )
    # token_pos: (bs * num_draft_tokens,) causal positions;
    # row_lens: one entry per (token, head group), dense rows keep token_pos, others clamp.
    return token_pos, row_lens.repeat_interleave(head_group_num)


def _build_dense_verify_overwrite(
    token_pos: torch.Tensor,
    token_to_bs: torch.Tensor,
    page_table: torch.Tensor,
    *,
    dense_len: int,
    head_group_num: int,
    out_rows: Optional[torch.Tensor] = None,
    out_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The page table can be narrower than the configured dense window.
    num_cols = min(dense_len, page_table.shape[1])
    dense_src = torch.index_select(page_table[:, :num_cols], 0, token_to_bs)
    group_offsets = torch.arange(
        head_group_num, dtype=dense_src.dtype, device=page_table.device
    )
    cols = torch.arange(num_cols, device=page_table.device)
    # Rows at token positions past dense_len get a zero column limit,
    # folding the row gate into the column comparison.
    col_limits = torch.where(token_pos < dense_len, token_pos, 0)
    # Group-interleaved prefix pages plus the mask of entries they replace,
    # in the layout _copy_dense_page_table writes.
    rows = torch.add(
        group_offsets[None, :, None],
        dense_src[:, None, :],
        alpha=head_group_num,
        out=out_rows,
    )
    mask = torch.lt(cols[None, None, :], col_limits[:, None, None], out=out_mask)
    return rows, mask


def _plan_sparse_prefill(
    forward_batch: ForwardBatch,
    metadata: MiniCPMSparseMetadata,
    head_group_num: int,
    heads_per_group: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
) -> None:
    device = metadata.base.cu_seqlens_q.device
    sparse_capacity = sparse_topk * block_size
    sparse_bs_list = []
    sparse_idx = []
    dense_layout = []
    dense_rows = []
    row_q_lens = []
    sparse_cache_seqlens = []
    token_to_bs = []
    token_pos_in_bs = []
    sparse_q_lens = []
    sparse_k_lens = []
    dense_q_lens = []
    max_sparse_cache_len = 0
    query_group_start = 0

    for batch_idx in range(forward_batch.batch_size):
        seq_len = int(forward_batch.seq_lens_cpu[batch_idx])
        query_len = int(forward_batch.extend_seq_lens_cpu[batch_idx])
        prefix_len = int(forward_batch.extend_prefix_lens_cpu[batch_idx])
        row_start = len(row_q_lens)
        if seq_len >= dense_len:
            sparse_batch_idx = len(sparse_bs_list)
            sparse_bs_list.append(batch_idx)
            sparse_q_lens.append(query_len)
            sparse_k_lens.append(seq_len)
            sparse_idx.extend(range(row_start, row_start + query_len * head_group_num))
            row_q_lens.extend([1] * (query_len * head_group_num))
            token_to_bs.extend([sparse_batch_idx] * query_len)
            token_pos_in_bs.extend(range(prefix_len + 1, prefix_len + query_len + 1))
            token_seq_lens = torch.arange(
                prefix_len + 1,
                prefix_len + query_len + 1,
                dtype=torch.int32,
            )
            sparse_cache_seqlens.extend(
                _get_sparse_cache_lens(token_seq_lens, sparse_capacity, block_size)
                .repeat_interleave(head_group_num)
                .tolist()
            )
            max_sparse_cache_len = max(max_sparse_cache_len, sparse_capacity)
        else:
            dense_layout.append((batch_idx, row_start, query_group_start, query_len))
            dense_rows.append((row_start, batch_idx, seq_len))
            dense_q_lens.append(query_len)
            row_q_lens.extend([query_len] * head_group_num)
            sparse_cache_seqlens.extend([seq_len] * head_group_num)
            max_sparse_cache_len = max(max_sparse_cache_len, seq_len)
        query_group_start += query_len * head_group_num

    metadata.sparse_bs_list = sparse_bs_list
    metadata.sparse_idx = sparse_idx
    metadata.dense_layout = dense_layout
    metadata.dense_rows = dense_rows
    metadata.token_to_bs = torch.tensor(token_to_bs, dtype=torch.int32, device=device)
    metadata.token_pos_in_bs = torch.tensor(
        token_pos_in_bs, dtype=torch.int32, device=device
    )
    metadata.seqlen_k_sparse_bs_tensor = torch.tensor(
        sparse_k_lens, dtype=torch.int32, device=device
    )
    metadata.sparse_page_table = torch.zeros(
        (len(row_q_lens), max_sparse_cache_len),
        dtype=metadata.base.page_table.dtype,
        device=metadata.base.page_table.device,
    )
    row_q_lens_tensor = torch.tensor(
        row_q_lens, dtype=metadata.base.cu_seqlens_q.dtype, device=device
    )
    metadata.sparse_cu_seqlens_q = F.pad(
        torch.cumsum(row_q_lens_tensor, dim=0, dtype=torch.int32), (1, 0)
    )
    metadata.sparse_max_seq_len_q = max(dense_q_lens, default=1)
    metadata.sparse_cache_seqlens_int32 = torch.tensor(
        sparse_cache_seqlens,
        dtype=torch.int32,
        device=device,
    )
    metadata.sparse_cu_seqlens_k = F.pad(
        torch.cumsum(metadata.sparse_cache_seqlens_int32, dim=0, dtype=torch.int32),
        (1, 0),
    )
    metadata.cache_seqlens_int32_stage1 = (
        metadata.base.cache_seqlens_int32[sparse_bs_list] - 1
    )

    if sparse_bs_list:
        sparse_q_lens_tensor = torch.tensor(
            sparse_q_lens, dtype=torch.int32, device=device
        )
        metadata.topk_cu_seqlens_q = F.pad(
            torch.cumsum(sparse_q_lens_tensor, dim=0, dtype=torch.int32), (1, 0)
        )
        metadata.topk_cu_seqlens_k = F.pad(
            torch.cumsum(metadata.seqlen_k_sparse_bs_tensor, dim=0, dtype=torch.int32),
            (1, 0),
        )
        metadata.topk_max_seqlen_q = max(sparse_q_lens)
        metadata.topk_max_seqlen_k = max(sparse_k_lens)
        metadata.cu_seqlens_q_adjusted = metadata.topk_cu_seqlens_q * heads_per_group
        metadata.max_seqlen_q_adjusted = metadata.topk_max_seqlen_q * heads_per_group
    else:
        metadata.cu_seqlens_q_adjusted = metadata.base.cu_seqlens_q * heads_per_group
        metadata.max_seqlen_q_adjusted = metadata.base.max_seq_len_q * heads_per_group


def _plan_repeated_segments(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    repeats: int,
    segment_starts: Optional[torch.Tensor] = None,
) -> RepeatedSegmentLayout:
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).repeat_interleave(repeats)
    out_cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    # segment_starts locates each request's chunks in the source buffer
    # when it is not packed (the CUDA graph slab); the output stays packed.
    if segment_starts is None:
        segment_starts = cu_seqlens[:-1]
    # index[i] = segment_start + (i - output_row_start),
    # so each output row adds (segment_start - output_row_start) to a flat arange.
    row_delta = (segment_starts.repeat_interleave(repeats) - out_cu_seqlens[:-1]).to(
        torch.int64
    )
    # total_tokens as output_size keeps repeat_interleave sync-free.
    index = row_delta.repeat_interleave(
        lengths.to(torch.int64), output_size=repeats * total_tokens
    ) + torch.arange(
        repeats * total_tokens, dtype=torch.int64, device=cu_seqlens.device
    )
    return RepeatedSegmentLayout(index=index, cu_seqlens=out_cu_seqlens)


def _plan_sparse_verify(
    forward_batch: ForwardBatch,
    metadata: MiniCPMSparseMetadata,
    *,
    head_group_num: int,
    heads_per_group: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
    num_draft_tokens: int,
) -> None:
    base_metadata = metadata.base
    bs = forward_batch.batch_size
    device = base_metadata.cu_seqlens_q.device
    sparse_capacity = sparse_topk * block_size

    token_to_bs = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32, device=device), num_draft_tokens
    )
    rows = bs * num_draft_tokens * head_group_num
    sparse_cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32, device=device)
    verify_cu_seqlens_q = torch.arange(
        bs * num_draft_tokens + 1, dtype=torch.int32, device=device
    )

    token_pos_in_bs, sparse_cache_seqlens_int32 = _build_sparse_verify_rows(
        forward_batch.seq_lens,
        head_group_num=head_group_num,
        dense_len=dense_len,
        sparse_topk=sparse_topk,
        block_size=block_size,
        num_draft_tokens=num_draft_tokens,
    )
    # Every verify row takes the selection output; the dense rows are then
    # overwritten from the prefix pages, so the mask is uniformly True.
    metadata.sparse_row_mask = torch.ones((rows, 1), dtype=torch.bool, device=device)
    if dense_len > 0:
        metadata.verify_dense_rows, metadata.verify_dense_mask = (
            _build_dense_verify_overwrite(
                token_pos_in_bs,
                token_to_bs,
                base_metadata.page_table,
                dense_len=dense_len,
                head_group_num=head_group_num,
            )
        )
    # The selection fill must write exactly these row lengths, row for row;
    # the FlashInfer plan consumes them.
    _assign_row_metadata(
        metadata,
        sparse_cache_seqlens_int32=sparse_cache_seqlens_int32,
        sparse_cu_seqlens_q=sparse_cu_seqlens_q,
        sparse_cu_seqlens_k=F.pad(
            torch.cumsum(sparse_cache_seqlens_int32, dim=0, dtype=torch.int32),
            (1, 0),
        ),
        sparse_page_table=torch.zeros(
            (rows, max(dense_len, sparse_capacity)),
            dtype=base_metadata.page_table.dtype,
            device=base_metadata.page_table.device,
        ),
        token_to_bs=token_to_bs,
        token_pos_in_bs=token_pos_in_bs,
        cache_seqlens_int32_stage1=token_pos_in_bs - 1,
        cu_seqlens_q_adjusted=verify_cu_seqlens_q * heads_per_group,
        max_seqlen_q_adjusted=heads_per_group,
        topk_cu_seqlens_q=verify_cu_seqlens_q,
        topk_cu_seqlens_k=base_metadata.cu_seqlens_k,
        topk_max_seqlen_q=1,
        topk_max_seqlen_k=base_metadata.max_seq_len_k,
    )


class SparseDecodeMetadata(msgspec.Struct):
    dense_rows: list[tuple[int, int, int]]
    sparse_row_mask: torch.Tensor
    sparse_cache_seqlens_int32: torch.Tensor
    sparse_cu_seqlens_k: torch.Tensor


def _build_sparse_decode_metadata(
    seq_lens_cpu: torch.Tensor,
    base_metadata: FlashAttentionMetadata,
    *,
    head_group_num: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
) -> SparseDecodeMetadata:
    cache_seqlens = base_metadata.cache_seqlens_int32

    seq_lens_cpu = torch.as_tensor(
        seq_lens_cpu, dtype=cache_seqlens.dtype, device="cpu"
    )
    is_sparse = seq_lens_cpu >= dense_len
    sparse_cache_seqlens_cpu = torch.where(
        is_sparse,
        _get_sparse_cache_lens(seq_lens_cpu, sparse_topk * block_size, block_size),
        seq_lens_cpu,
    ).repeat_interleave(head_group_num)

    sparse_cache_seqlens_int32 = sparse_cache_seqlens_cpu.to(
        device=cache_seqlens.device
    )
    sparse_row_mask = is_sparse.repeat_interleave(head_group_num)[:, None].to(
        device=cache_seqlens.device
    )
    sparse_cu_seqlens_k = F.pad(
        torch.cumsum(sparse_cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    return SparseDecodeMetadata(
        dense_rows=_plan_dense_rows(
            seq_lens_cpu, dense_len=dense_len, head_group_num=head_group_num
        ),
        sparse_row_mask=sparse_row_mask,
        sparse_cache_seqlens_int32=sparse_cache_seqlens_int32,
        sparse_cu_seqlens_k=sparse_cu_seqlens_k,
    )


def _plan_sparse_decode(
    forward_batch: ForwardBatch,
    metadata: MiniCPMSparseMetadata,
    *,
    head_group_num: int,
    heads_per_group: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
) -> None:
    base_metadata = metadata.base
    sparse_capacity = sparse_topk * block_size
    decode_metadata = _build_sparse_decode_metadata(
        seq_lens_cpu=forward_batch.seq_lens_cpu,
        base_metadata=base_metadata,
        head_group_num=head_group_num,
        dense_len=dense_len,
        sparse_topk=sparse_topk,
        block_size=block_size,
    )

    sparse_rows = head_group_num * forward_batch.batch_size
    metadata.dense_rows = decode_metadata.dense_rows
    metadata.sparse_row_mask = decode_metadata.sparse_row_mask
    _assign_row_metadata(
        metadata,
        sparse_cache_seqlens_int32=decode_metadata.sparse_cache_seqlens_int32,
        sparse_cu_seqlens_q=torch.arange(
            0,
            sparse_rows + 1,
            dtype=torch.int32,
            device=base_metadata.cu_seqlens_q.device,
        ),
        sparse_cu_seqlens_k=decode_metadata.sparse_cu_seqlens_k,
        sparse_page_table=torch.zeros(
            (
                sparse_rows,
                max(dense_len, sparse_capacity),
            ),
            dtype=base_metadata.page_table.dtype,
            device=base_metadata.page_table.device,
        ),
        token_to_bs=torch.arange(
            0,
            forward_batch.batch_size,
            dtype=torch.int32,
            device=base_metadata.page_table.device,
        ),
        token_pos_in_bs=base_metadata.cache_seqlens_int32,
        cache_seqlens_int32_stage1=base_metadata.cache_seqlens_int32 - 1,
        cu_seqlens_q_adjusted=base_metadata.cu_seqlens_q * heads_per_group,
        max_seqlen_q_adjusted=base_metadata.max_seq_len_q * heads_per_group,
        topk_cu_seqlens_q=base_metadata.cu_seqlens_q,
        topk_cu_seqlens_k=base_metadata.cu_seqlens_k,
        topk_max_seqlen_q=1,
        topk_max_seqlen_k=base_metadata.max_seq_len_k,
    )
