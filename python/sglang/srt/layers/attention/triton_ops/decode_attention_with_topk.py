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
Memory-efficient top-k attention extraction for interpretability.

This module provides efficient top-k attention extraction WITHOUT
materializing the full attention matrix. It uses a chunked approach:

1. For each chunk of keys, compute Q @ K^T scores
2. Keep only the top-k per chunk (small buffer)
3. Merge across chunks to get final top-k

Memory: O(batch × heads × k × num_chunks) during computation
        O(batch × k) final output

For 1M context with chunk_size=4096, num_chunks=244, k=10:
  Intermediate: batch × 64 × 10 × 244 × 4 = ~6MB (vs 4GB for full matrix!)
  Final: batch × 10 × 8 = 80 bytes per sequence

This is called AFTER the main attention forward, not integrated into it,
to keep the performance-critical attention kernel simple.
"""

import logging
from typing import Optional, Tuple, List

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _compute_chunk_max_kernel(
    Q,                    # [batch, num_heads, head_dim]
    K_Buffer,             # [total_kv, num_kv_heads, head_dim]
    kv_indptr,           # [batch + 1]
    kv_indices,          # [total_kv]
    Chunk_Max_Scores,    # [batch, num_heads, num_chunks]
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_outb,
    stride_outh,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    Lk: tl.constexpr,
):
    """
    Compute max attention score for each chunk of keys.

    Grid: (batch, num_heads, num_chunks)
    Each program handles one (batch, head, chunk) tuple.
    Stores only the max score per chunk for memory efficiency.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_chunk = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Get sequence range for this batch
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    seq_len = kv_end - kv_start

    # Chunk boundaries
    chunk_start = cur_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, seq_len)

    # Output location
    out_offset = cur_batch * stride_outb + cur_head * stride_outh + cur_chunk

    # Early exit for out-of-range chunks
    if chunk_start >= seq_len:
        tl.store(Chunk_Max_Scores + out_offset, -1e9)
        return

    # Load query once
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk
    q = tl.load(
        Q + cur_batch * stride_qb + cur_head * stride_qh + offs_d,
        mask=mask_d,
        other=0.0
    )

    # Accumulate all scores for this chunk, then take max at the end
    # Process the entire chunk as a single block for simplicity
    # (chunk_size should be reasonably small, e.g., 1024-2048)
    offs_n = tl.arange(0, CHUNK_SIZE)
    mask_n = (chunk_start + offs_n) < chunk_end

    # Get KV cache locations
    kv_loc = tl.load(
        kv_indices + kv_start + chunk_start + offs_n,
        mask=mask_n,
        other=0
    )

    # Load keys
    k = tl.load(
        K_Buffer + kv_loc[:, None] * stride_kb + cur_kv_head * stride_kh + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0
    )

    # Compute attention scores: q @ k^T
    scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
    scores = tl.where(mask_n, scores, -1e9)

    # Get max score for this chunk
    max_score = tl.max(scores)

    # Store chunk max score
    tl.store(Chunk_Max_Scores + out_offset, max_score)


def compute_topk_attention_chunked(
    q: torch.Tensor,              # [batch, num_heads, head_dim]
    k_buffer: torch.Tensor,       # [total_kv, num_kv_heads, head_dim]
    kv_indptr: torch.Tensor,      # [batch + 1]
    kv_indices: torch.Tensor,     # [total_kv]
    sm_scale: float,
    top_k: int = 10,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k attention positions using chunked approach.

    Two-phase algorithm:
    1. Triton kernel: Compute max score per chunk (memory-efficient)
    2. PyTorch: Rescan top-k chunks to find exact positions

    Memory: O(batch × heads × num_chunks) intermediate
    For 1M context: ~125KB vs ~256MB with full matrix

    Returns:
        topk_scores: [batch, top_k] - normalized attention scores (averaged across heads)
        topk_indices: [batch, top_k] - sequence positions
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    device = q.device

    # Compute max sequence length
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    max_seq_len = seq_lens.max().item()

    if max_seq_len == 0:
        return (
            torch.zeros((batch_size, top_k), dtype=torch.float32, device=device),
            torch.zeros((batch_size, top_k), dtype=torch.int64, device=device),
        )

    # For small sequences, use direct PyTorch (simpler, no kernel overhead)
    if max_seq_len <= chunk_size * 2:
        return _compute_topk_pytorch(
            q, k_buffer, kv_indptr, kv_indices, sm_scale, top_k
        )

    # Chunked approach for large sequences
    num_chunks = (max_seq_len + chunk_size - 1) // chunk_size

    # Phase 1: Use Triton kernel to compute max score per chunk
    chunk_max_scores = torch.empty(
        (batch_size, num_heads, num_chunks),
        dtype=torch.float32,
        device=device
    )

    Lk = head_dim
    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    grid = (batch_size, num_heads, num_chunks)

    _compute_chunk_max_kernel[grid](
        q,
        k_buffer,
        kv_indptr,
        kv_indices,
        chunk_max_scores,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        chunk_max_scores.stride(0),
        chunk_max_scores.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CHUNK_SIZE=chunk_size,
        Lk=Lk,
        num_warps=4,
    )

    # Phase 2: Find top-k chunks and rescan them for exact positions
    # Average scores across heads for interpretability
    avg_chunk_scores = chunk_max_scores.mean(dim=1)  # [batch, num_chunks]

    # Get top-k chunks (get extra in case we need to deduplicate)
    k_chunks = min(top_k * 2, num_chunks)
    _, topk_chunk_idx = torch.topk(avg_chunk_scores, k_chunks, dim=-1)

    # Rescan top chunks in PyTorch to get exact positions and scores
    return _rescan_top_chunks(
        q, k_buffer, kv_indptr, kv_indices,
        sm_scale, topk_chunk_idx, chunk_size, top_k
    )


def _rescan_top_chunks(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    topk_chunk_idx: torch.Tensor,  # [batch, k_chunks]
    chunk_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rescan the top chunks to find exact token positions.
    Only processes the top chunks, keeping memory bounded.
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    device = q.device
    k_chunks = topk_chunk_idx.shape[1]

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

        # Gather keys only from top chunks (memory efficient)
        all_scores = []
        all_positions = []

        for chunk_idx in topk_chunk_idx[b].tolist():
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            if chunk_start >= seq_len:
                continue

            # Get positions in this chunk
            positions = torch.arange(chunk_start, chunk_end, device=device)
            kv_pos = kv_indices[kv_start + chunk_start:kv_start + chunk_end]

            # Gather keys: [chunk_len, num_kv_heads, head_dim]
            keys = k_buffer[kv_pos]

            # Expand for GQA
            if kv_group_num > 1:
                keys = keys.repeat_interleave(kv_group_num, dim=1)

            # Query: [num_heads, head_dim]
            query = q[b]

            # Attention scores: [num_heads, chunk_len]
            scores = torch.einsum("hd,shd->hs", query.float(), keys.float()) * sm_scale

            # Average across heads
            scores_avg = scores.mean(dim=0)  # [chunk_len]

            all_scores.append(scores_avg)
            all_positions.append(positions)

        if len(all_scores) == 0:
            topk_scores_list.append(torch.zeros(top_k, device=device))
            topk_indices_list.append(torch.zeros(top_k, dtype=torch.int64, device=device))
            continue

        # Concatenate all chunk results
        all_scores = torch.cat(all_scores)
        all_positions = torch.cat(all_positions)

        # Get top-k from the candidate positions
        actual_k = min(top_k, len(all_scores))
        topk_vals, topk_idx = torch.topk(all_scores, actual_k)
        topk_positions = all_positions[topk_idx]

        # Softmax normalize
        topk_probs = torch.softmax(topk_vals, dim=0)

        # Pad if needed
        if actual_k < top_k:
            padding = top_k - actual_k
            topk_probs = torch.cat([topk_probs, torch.zeros(padding, device=device)])
            topk_positions = torch.cat([topk_positions, torch.zeros(padding, dtype=torch.int64, device=device)])

        topk_scores_list.append(topk_probs)
        topk_indices_list.append(topk_positions)

    return torch.stack(topk_scores_list), torch.stack(topk_indices_list)


def _compute_topk_pytorch(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation for small sequences.
    More precise than chunked, used when seq_len is manageable.
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

        # Get KV cache positions
        kv_pos = kv_indices[kv_start:kv_end]

        # Gather keys: [seq_len, num_kv_heads, head_dim]
        keys = k_buffer[kv_pos]

        # Expand for GQA
        if kv_group_num > 1:
            keys = keys.repeat_interleave(kv_group_num, dim=1)

        # Query: [num_heads, head_dim]
        query = q[b]

        # Attention scores: [num_heads, seq_len]
        scores = torch.einsum("hd,shd->hs", query, keys) * sm_scale

        # Average across heads
        scores_avg = scores.mean(dim=0)  # [seq_len]

        # Top-k
        actual_k = min(top_k, seq_len)
        topk_vals, topk_idx = torch.topk(scores_avg, actual_k)

        # Softmax normalize
        topk_probs = torch.softmax(topk_vals, dim=0)

        # Pad if needed
        if actual_k < top_k:
            padding = top_k - actual_k
            topk_probs = torch.cat([topk_probs, torch.zeros(padding, device=device)])
            topk_idx = torch.cat([topk_idx, torch.zeros(padding, dtype=torch.int64, device=device)])

        topk_scores_list.append(topk_probs)
        topk_indices_list.append(topk_idx)

    return torch.stack(topk_scores_list), torch.stack(topk_indices_list)


# =============================================================================
# HIGH-LEVEL API FOR INTEGRATION
# =============================================================================

class TopKAttentionCapture:
    """
    High-level API for capturing top-k attention during decode.

    Usage in triton_backend.py:
        capture = TopKAttentionCapture(top_k=10)

        # After attention forward:
        if forward_batch.capture_attention_tokens:
            topk_info = capture.extract(
                q, k_buffer, kv_indptr, kv_indices, sm_scale
            )
            forward_batch.attention_token_info = topk_info
    """

    def __init__(self, top_k: int = 10, chunk_size: int = 2048):
        self.top_k = top_k
        self.chunk_size = chunk_size

    def extract(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
    ) -> dict:
        """
        Extract top-k attention positions.

        Returns dict with:
            - scores: [batch, top_k] normalized
            - indices: [batch, top_k] positions
        """
        scores, indices = compute_topk_attention_chunked(
            q, k_buffer, kv_indptr, kv_indices,
            sm_scale, self.top_k, self.chunk_size
        )

        return {
            "scores": scores,
            "indices": indices,
        }

    def format_for_response(self, topk_info: dict, layer_id: int = -1) -> List[dict]:
        """
        Format for API response.

        Returns list of dicts per batch matching frontend schema:
        {token_positions: [...], attention_scores: [...], layer_id: N}
        """
        scores = topk_info["scores"]
        indices = topk_info["indices"]
        batch_size = scores.shape[0]

        result = []
        for b in range(batch_size):
            result.append({
                "token_positions": indices[b].cpu().tolist(),
                "attention_scores": scores[b].cpu().tolist(),
                "layer_id": layer_id,
            })
        return result


# =============================================================================
# TESTING
# =============================================================================

def test_topk_attention():
    """Quick test of the top-k extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 1000

    # Create test data
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    k_buffer = torch.randn(seq_len * batch_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    kv_indptr = torch.tensor([0, seq_len, seq_len * 2], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len * batch_size, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Test
    capture = TopKAttentionCapture(top_k=10)
    result = capture.extract(q, k_buffer, kv_indptr, kv_indices, sm_scale)

    print(f"Top-k scores shape: {result['scores'].shape}")
    print(f"Top-k indices shape: {result['indices'].shape}")
    print(f"Sample scores: {result['scores'][0]}")
    print(f"Sample indices: {result['indices'][0]}")

    # Format for API
    formatted = capture.format_for_response(result)
    print(f"Formatted response: {formatted[0]}")


if __name__ == "__main__":
    test_topk_attention()
