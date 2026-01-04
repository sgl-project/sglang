# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Top-K Attention Token Extraction for Interpretability.

This module computes the top-k attention tokens for each batch element during
decode phase. It's designed for interpretability/visualization without
materializing the full attention matrix.
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_attention_scores_kernel(
    Q,  # Query tensor [batch, num_heads, head_dim]
    K_Buffer,  # Key buffer from KV cache
    kv_indptr,  # [batch + 1] - start/end indices into kv_indices
    kv_indices,  # Flattened KV cache indices
    Scores_Out,  # Output: attention scores [batch, num_heads, max_seq_len]
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_ss,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
):
    """
    Compute attention scores (Q @ K^T * scale) for each batch/head.
    Writes scores to output buffer for subsequent top-k selection.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    block_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Get KV range for this batch element
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    kv_len = kv_end - kv_start

    # Compute which KV positions this block handles
    block_start = block_id * BLOCK_N
    if block_start >= kv_len:
        return

    offs_n = block_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < kv_len

    # Load query vector
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk
    q = tl.load(
        Q + cur_batch * stride_qb + cur_head * stride_qh + offs_d,
        mask=mask_d,
        other=0.0,
    )

    # Load KV indices for this block
    kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)

    # Load K vectors
    k = tl.load(
        K_Buffer + kv_loc[:, None] * stride_kb + cur_kv_head * stride_kh + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Compute attention scores: Q @ K^T
    scores = tl.sum(q[None, :] * k, axis=1) * sm_scale

    # Mask out invalid positions
    scores = tl.where(mask_n, scores, float("-inf"))

    # Store scores
    tl.store(
        Scores_Out + cur_batch * stride_sb + cur_head * stride_sh + offs_n,
        scores,
        mask=mask_n,
    )


def compute_topk_attention_tokens(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    top_k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k attention tokens for each batch element.

    This function computes attention scores Q @ K^T for the current query
    against all keys in the KV cache, then returns the top-k positions
    and their normalized attention scores.

    Args:
        q: Query tensor [batch, num_heads, head_dim]
        k_buffer: Key buffer from KV cache [total_tokens, num_kv_heads, head_dim]
        kv_indptr: CSR-style indptr for KV cache access [batch + 1]
        kv_indices: Flattened KV cache indices
        sm_scale: Softmax scale factor (typically 1/sqrt(head_dim))
        top_k: Number of top attention tokens to return

    Returns:
        topk_scores: [batch, top_k] - Softmax-normalized attention scores
        topk_indices: [batch, top_k] - Token positions in the sequence
    """
    batch_size, num_heads, head_dim = q.shape
    device = q.device
    dtype = q.dtype

    # Determine max sequence length from kv_indptr
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    max_seq_len = seq_lens.max().item()

    if max_seq_len == 0:
        # No KV cache yet, return empty results
        return (
            torch.zeros((batch_size, top_k), dtype=torch.float32, device=device),
            torch.zeros((batch_size, top_k), dtype=torch.int64, device=device),
        )

    # For small sequences, use direct PyTorch computation
    # For large sequences, use chunked Triton kernel
    use_triton = max_seq_len > 1024

    if use_triton:
        return _compute_topk_triton(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k, max_seq_len
        )
    else:
        return _compute_topk_pytorch(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k
        )


def _compute_topk_pytorch(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch fallback for computing top-k attention tokens.
    Used for smaller sequences where kernel launch overhead dominates.
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    device = q.device

    topk_scores_list = []
    topk_indices_list = []

    for b in range(batch_size):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        seq_len = kv_end - kv_start

        if seq_len == 0:
            topk_scores_list.append(torch.zeros(top_k, device=device))
            topk_indices_list.append(torch.zeros(top_k, dtype=torch.int64, device=device))
            continue

        # Get KV cache positions for this sequence
        kv_pos = kv_indices[kv_start:kv_end]

        # Gather keys: [seq_len, num_kv_heads, head_dim]
        keys = k_buffer[kv_pos]

        # Expand keys for GQA: [seq_len, num_heads, head_dim]
        if kv_group_num > 1:
            keys = keys.repeat_interleave(kv_group_num, dim=1)

        # Query for this batch: [num_heads, head_dim]
        query = q[b]

        # Compute attention scores: [num_heads, seq_len]
        # scores = (query @ keys.transpose(-1, -2)) * sm_scale
        scores = torch.einsum("hd,shd->hs", query, keys) * sm_scale

        # Average across heads for interpretability
        scores_avg = scores.mean(dim=0)  # [seq_len]

        # Get top-k
        actual_k = min(top_k, seq_len)
        topk_vals, topk_idx = torch.topk(scores_avg, actual_k)

        # Apply softmax to normalize scores
        topk_probs = torch.softmax(topk_vals, dim=0)

        # Pad if needed
        if actual_k < top_k:
            padding = top_k - actual_k
            topk_probs = torch.cat([topk_probs, torch.zeros(padding, device=device)])
            topk_idx = torch.cat([topk_idx, torch.zeros(padding, dtype=torch.int64, device=device)])

        # Convert local indices to sequence positions
        topk_positions = topk_idx  # These are already 0-indexed positions in the sequence

        topk_scores_list.append(topk_probs)
        topk_indices_list.append(topk_positions)

    return torch.stack(topk_scores_list), torch.stack(topk_indices_list)


def _compute_topk_triton(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    top_k: int,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated top-k attention computation for large sequences.

    Uses a chunked approach:
    1. Compute attention scores in blocks using Triton
    2. Use PyTorch topk on the resulting scores
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    device = q.device

    BLOCK_N = 128
    num_blocks = (max_seq_len + BLOCK_N - 1) // BLOCK_N

    # Allocate score buffer
    scores_buffer = torch.full(
        (batch_size, num_heads, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    Lk = head_dim
    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    grid = (batch_size, num_heads, num_blocks)

    _topk_attention_scores_kernel[grid](
        q,
        k_buffer,
        kv_indptr,
        kv_indices,
        scores_buffer,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        scores_buffer.stride(0),
        scores_buffer.stride(1),
        scores_buffer.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        Lk=Lk,
        num_warps=4,
        num_stages=2,
    )

    # Average across heads
    scores_avg = scores_buffer.mean(dim=1)  # [batch, max_seq_len]

    # Get top-k for each batch element
    topk_vals, topk_idx = torch.topk(scores_avg, min(top_k, max_seq_len), dim=-1)

    # Normalize with softmax
    topk_probs = torch.softmax(topk_vals, dim=-1)

    # Pad if needed
    if topk_idx.shape[-1] < top_k:
        padding = top_k - topk_idx.shape[-1]
        topk_probs = torch.cat([
            topk_probs,
            torch.zeros((batch_size, padding), device=device)
        ], dim=-1)
        topk_idx = torch.cat([
            topk_idx,
            torch.zeros((batch_size, padding), dtype=torch.int64, device=device)
        ], dim=-1)

    return topk_probs, topk_idx
