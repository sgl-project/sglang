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
DeepEP-based Waterfill Load Balancing for Shared Expert.

This module implements waterfill load balancing where shared expert is treated
as the 9th routed expert and dispatched through DeepEP.

Key Design:
1. Each token's shared expert can ONLY be sent to:
   - A rank it already routes to (no extra communication)
   - Or source rank (local computation, marked with LOCAL_SHARED_MARKER)

2. Virtual expert ID = target_rank * experts_per_rank
   - This ensures DeepEP routes to the correct rank
   - LOCAL_SHARED_MARKER (-1) means compute locally, don't dispatch

3. On receiver side:
   - Identify tokens whose 9th expert is for this rank
   - Compute shared expert separately from routed experts
   - Merge outputs before combine

4. Shared expert weight = 1.0 / routed_scaling_factor

5. Small batch optimization:
   - If batch size < MIN_BATCH_FOR_BALANCE, all shared experts compute locally
   - Avoids fragmented computation across ranks
"""

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

DEEPEP_WATERFILL_DEBUG = os.environ.get("SGLANG_DEEPEP_WATERFILL_DEBUG", "0") == "1"

# Marker for local shared expert computation (won't be dispatched)
LOCAL_SHARED_MARKER = -1

# Global counter for periodic logging
_waterfill_log_counter = [0]
_WATERFILL_LOG_INTERVAL = 100  # Log every N calls

# Local preference factor: only send to remote if remote_count * factor < local_count
# This avoids unnecessary remote communication when load is balanced
# Set to 1.0 to disable local preference (original behavior)
# Set to 1.2 to prefer local unless remote is 20% less loaded
LOCAL_PREFERENCE_FACTOR = 1.2

# Try to import Triton for GPU-optimized kernels
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============== Triton Kernels (GPU-optimized) ==============


if HAS_TRITON:

    @triton.jit
    def _waterfill_expand_topk_fused_kernel(
        # Inputs
        topk_ids_ptr,  # [num_tokens, topk]
        topk_weights_ptr,  # [num_tokens, topk]
        routed_counts_ptr,  # [world_size]
        # Outputs
        expanded_ids_ptr,  # [num_tokens, topk+1]
        expanded_weights_ptr,  # [num_tokens, topk+1]
        local_mask_ptr,  # [num_tokens]
        # Scalars
        num_tokens,
        topk: tl.constexpr,
        old_experts_per_rank,  # Original experts per rank (e.g., 32)
        new_experts_per_rank,  # New experts per rank (e.g., 33)
        world_size,
        source_rank,
        shared_weight,
        local_marker,  # LOCAL_SHARED_MARKER = -1
        local_pref_numer,  # Local preference numerator (e.g., 6 for 1.2x)
        local_pref_denom,  # Local preference denominator (e.g., 5 for 1.2x)
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused Triton kernel for waterfill assignment + topk expansion with expert ID remapping.

        Expert ID remapping: old_id -> old_id + (old_id // old_experts_per_rank)
        Shared expert ID: target_rank * new_experts_per_rank + old_experts_per_rank

        For each token:
        1. Find all ranks it routes to (from topk_ids)
        2. Select the rank with minimum routed_count (waterfill)
           - With local preference: only choose remote if remote_count * numerator/denom < local_count
        3. Remap routed expert IDs and expand to include shared expert
        4. Set local_mask for tokens computed locally

        This kernel fuses assign_shared_destination + expand_topk_with_shared_expert
        into a single kernel pass, reducing memory traffic and kernel launch overhead.
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        # ===== Step 1: Waterfill - find best destination rank =====
        # Initialize with source rank (always a candidate)
        source_count = tl.load(routed_counts_ptr + source_rank)
        best_count = tl.where(mask, source_count, 2**30)
        best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)

        # Check each routed expert and update if better
        for k in range(topk):
            # Load expert ID
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid = expert_id >= 0

            # Compute target rank from ORIGINAL expert ID
            target_rank = expert_id // old_experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)

            # Load routed count for this rank
            target_count = tl.load(
                routed_counts_ptr + target_rank, mask=mask & valid, other=2**30
            )

            # Update if this rank has significantly lower count (waterfill with local preference)
            # Only prefer remote if: target_count * numerator < best_count * denom
            # This is equivalent to: target_count * (numerator/denom) < best_count
            # For numerator=6, denom=5: target_count * 1.2 < best_count (20% threshold)
            better = (
                (target_count * local_pref_numer < best_count * local_pref_denom)
                & valid
                & mask
            )
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)

        # ===== Step 2: Compute shared expert ID and local mask =====
        is_local = best_rank == source_rank

        # Shared expert ID: target_rank * new_experts_per_rank + old_experts_per_rank
        # This places shared expert at the END of each rank's expert range
        # NOTE: For local shared expert, we use the REAL shared expert ID (not local_marker=-1)
        # This ensures local shared expert is also computed in MoE layer
        local_shared_id = source_rank * new_experts_per_rank + old_experts_per_rank
        remote_shared_id = best_rank * new_experts_per_rank + old_experts_per_rank
        shared_expert_id = tl.where(
            is_local,
            tl.full([BLOCK_SIZE], local_shared_id, dtype=tl.int64),
            remote_shared_id,
        ).to(tl.int64)

        # ===== Step 3: Copy and remap topk_ids, copy topk_weights =====
        # Remap: old_id -> old_id + (old_id // old_experts_per_rank)
        for k in range(topk):
            old_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            # Only remap valid IDs (>= 0)
            valid_id = old_id >= 0
            # new_id = old_id + (old_id // old_experts_per_rank)
            new_id = tl.where(
                valid_id, old_id + (old_id // old_experts_per_rank), old_id
            )
            tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

        # Copy topk_weights columns
        for k in range(topk):
            val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
            tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

        # ===== Step 4: Write 9th column (shared expert) =====
        tl.store(
            expanded_ids_ptr + token_idx * (topk + 1) + topk,
            shared_expert_id,
            mask=mask,
        )
        tl.store(
            expanded_weights_ptr + token_idx * (topk + 1) + topk,
            shared_weight,
            mask=mask,
        )

        # ===== Step 5: Write local mask =====
        tl.store(local_mask_ptr + token_idx, is_local, mask=mask)

    def waterfill_expand_topk_fused(
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
        num_experts: int,
        world_size: int,
        source_rank: int,
        shared_weight: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fused waterfill assignment + topk expansion using Triton.

        This is a single kernel that does:
        1. Waterfill: For each token, find the least loaded rank among its routed ranks
        2. Expand topk from [N, 8] to [N, 9] with shared expert info

        Returns:
            expanded_topk_ids: [N, 9]
            expanded_topk_weights: [N, 9]
            local_shared_mask: [N] boolean
        """
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = num_experts // world_size
        device = topk_ids.device

        if num_tokens == 0:
            return (
                torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=device),
                torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )

        # Pre-allocate outputs
        expanded_topk_ids = torch.empty(
            num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
        )
        expanded_topk_weights = torch.empty(
            num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
        )
        local_shared_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)

        # Launch fused kernel
        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Convert LOCAL_PREFERENCE_FACTOR to integer ratio to avoid float in kernel
        # 1.2 = 6/5, 1.0 = 5/5 (disabled)
        local_pref_numer = int(LOCAL_PREFERENCE_FACTOR * 5)
        local_pref_denom = 5

        _waterfill_expand_topk_fused_kernel[grid](
            topk_ids,
            topk_weights,
            routed_counts,
            expanded_topk_ids,
            expanded_topk_weights,
            local_shared_mask,
            num_tokens,
            topk,
            experts_per_rank,
            world_size,
            source_rank,
            shared_weight,
            LOCAL_SHARED_MARKER,
            local_pref_numer,
            local_pref_denom,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask

    @triton.jit
    def _count_destinations_kernel(
        destination_ptr,  # [num_tokens] - destination rank for each token
        counts_ptr,  # [world_size] - output counts (atomic add)
        num_tokens,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Count tokens per destination rank using atomic operations."""
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        dest = tl.load(destination_ptr + token_idx, mask=mask, other=0)

        # Use atomic add to count
        # Note: This creates contention but is simpler than reduction
        for i in range(BLOCK_SIZE):
            if tl.arange(0, BLOCK_SIZE)[i] < num_tokens - pid * BLOCK_SIZE:
                d = tl.load(destination_ptr + pid * BLOCK_SIZE + i)
                tl.atomic_add(counts_ptr + d, 1)

    @triton.jit
    def _masked_scatter_add_kernel(
        output_ptr,  # [N, H] - output tensor to add to
        input_ptr,  # [num_selected, H] - packed input tensor
        prefix_ptr,  # [N] - exclusive prefix sum of mask
        mask_ptr,  # [N] - boolean mask
        num_tokens,
        hidden_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Scatter-add packed input to output using mask, without explicit indices.

        For each position where mask[i] is True:
            output[i, :] += input[prefix[i], :]

        prefix[i] = number of True values in mask[:i] (exclusive prefix sum)
        """
        token_idx = tl.program_id(0)
        if token_idx >= num_tokens:
            return

        is_selected = tl.load(mask_ptr + token_idx)
        if not is_selected:
            return

        # Get packed index from exclusive prefix sum
        packed_idx = tl.load(prefix_ptr + token_idx)

        # Process hidden dimension in blocks
        for h_start in range(0, hidden_size, BLOCK_H):
            h_idx = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_idx < hidden_size

            # Load from packed input
            input_val = tl.load(
                input_ptr + packed_idx * hidden_size + h_idx, mask=h_mask, other=0.0
            )

            # Load current output
            output_val = tl.load(
                output_ptr + token_idx * hidden_size + h_idx, mask=h_mask, other=0.0
            )

            # Store sum
            tl.store(
                output_ptr + token_idx * hidden_size + h_idx,
                output_val + input_val,
                mask=h_mask,
            )

    @triton.jit
    def _identify_shared_expert_kernel(
        recv_topk_ids_ptr,  # [num_tokens, topk+1] - received topk IDs
        output_mask_ptr,  # [num_tokens] - output boolean mask
        num_tokens,
        topk_plus_one,  # topk + 1 = 9
        experts_per_rank,
        current_rank,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel to identify shared expert tokens.

        A token needs shared expert on this rank if its 9th column (virtual expert ID)
        maps to current_rank. Tokens with LOCAL_SHARED_MARKER (-1) are skipped.
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        # Load 9th column (virtual expert ID)
        virtual_id = tl.load(
            recv_topk_ids_ptr + token_idx * topk_plus_one + (topk_plus_one - 1),
            mask=mask,
            other=-1,
        ).to(tl.int64)

        # Check if valid (>= 0) and maps to current rank
        valid = virtual_id >= 0
        target_rank = virtual_id // experts_per_rank
        is_for_this_rank = valid & (target_rank == current_rank)

        # Store result
        tl.store(output_mask_ptr + token_idx, is_for_this_rank, mask=mask)

    @triton.jit
    def _count_routed_per_rank_kernel(
        topk_ids_ptr,  # [num_tokens, topk]
        counts_ptr,  # [world_size] output (atomic add)
        num_tokens,
        topk: tl.constexpr,
        experts_per_rank,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Count routed tokens per rank using Triton.
        Uses block-level histogram to minimize atomic contention.
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        # For each rank, count tokens in this block that route to it
        for r in range(world_size):
            rank_count = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

            for k in range(topk):
                expert_id = tl.load(
                    topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
                ).to(tl.int64)
                valid = expert_id >= 0
                target_rank = expert_id // experts_per_rank
                target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
                # Use int64 for consistency with output type
                rank_count += tl.where(
                    mask & valid & (target_rank == r),
                    tl.full([BLOCK_SIZE], 1, dtype=tl.int64),
                    tl.zeros([BLOCK_SIZE], dtype=tl.int64),
                )

            block_total = tl.sum(rank_count)
            if block_total > 0:
                tl.atomic_add(counts_ptr + r, block_total)

    @triton.jit
    def _waterfill_expand_with_histogram_kernel(
        # Inputs
        topk_ids_ptr,  # [num_tokens, topk]
        topk_weights_ptr,  # [num_tokens, topk]
        routed_counts_ptr,  # [world_size]
        # Outputs
        expanded_ids_ptr,  # [num_tokens, topk+1]
        expanded_weights_ptr,  # [num_tokens, topk+1]
        local_mask_ptr,  # [num_tokens]
        dest_counts_ptr,  # [world_size] - output histogram (atomic)
        # Scalars
        num_tokens,
        topk: tl.constexpr,
        old_experts_per_rank,  # Original experts per rank (e.g., 32)
        new_experts_per_rank,  # New experts per rank (e.g., 33)
        world_size: tl.constexpr,
        source_rank,
        shared_weight,
        local_marker,
        local_pref_numer,
        local_pref_denom,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused waterfill + expand + histogram kernel with expert ID remapping.

        Expert ID remapping: old_id -> old_id + (old_id // old_experts_per_rank)
        This ensures each rank's expert range is [r*new_epr, (r+1)*new_epr-1]
        with shared expert at position (r+1)*new_epr - 1.

        Uses block-level histogram accumulation to minimize atomic contention.
        Each block computes a local histogram, then does world_size atomic adds.
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        # ===== Step 1: Waterfill - find best destination rank =====
        source_count = tl.load(routed_counts_ptr + source_rank)
        best_count = tl.where(mask, source_count, 2**30)
        best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)

        for k in range(topk):
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid = expert_id >= 0

            # Use OLD experts_per_rank for rank calculation from original expert IDs
            target_rank = expert_id // old_experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)

            target_count = tl.load(
                routed_counts_ptr + target_rank, mask=mask & valid, other=2**30
            )

            better = (
                (target_count * local_pref_numer < best_count * local_pref_denom)
                & valid
                & mask
            )
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)

        # ===== Step 2: Compute shared expert ID and local mask =====
        is_local = best_rank == source_rank

        # Shared expert ID = target_rank * new_experts_per_rank + old_experts_per_rank
        # This puts shared expert at the end of each rank's expert range
        # NOTE: For local shared expert, we use the REAL shared expert ID (not local_marker=-1)
        # This ensures local shared expert is also computed in MoE layer
        local_shared_id = source_rank * new_experts_per_rank + old_experts_per_rank
        remote_shared_id = best_rank * new_experts_per_rank + old_experts_per_rank
        shared_expert_id = tl.where(
            is_local,
            tl.full([BLOCK_SIZE], local_shared_id, dtype=tl.int64),
            remote_shared_id,
        ).to(tl.int64)

        dest_rank = tl.where(is_local, source_rank, best_rank).to(tl.int32)

        # ===== Step 3: Copy and remap topk_ids, copy topk_weights =====
        # Remap routed expert IDs: old_id -> old_id + (old_id // old_experts_per_rank)
        for k in range(topk):
            old_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            # Only remap valid IDs (>= 0)
            valid_id = old_id >= 0
            # new_id = old_id + (old_id // old_experts_per_rank)
            new_id = tl.where(
                valid_id, old_id + (old_id // old_experts_per_rank), old_id
            )
            tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

        for k in range(topk):
            val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
            tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

        # ===== Step 4: Write 9th column (shared expert) =====
        tl.store(
            expanded_ids_ptr + token_idx * (topk + 1) + topk,
            shared_expert_id,
            mask=mask,
        )
        tl.store(
            expanded_weights_ptr + token_idx * (topk + 1) + topk,
            shared_weight,
            mask=mask,
        )

        # ===== Step 5: Write local mask =====
        tl.store(local_mask_ptr + token_idx, is_local, mask=mask)

        # ===== Step 6: Block-level histogram with minimal atomics =====
        # Count destinations per rank within this block using sum reduction
        for r in range(world_size):
            rank_count = tl.sum(tl.where(mask & (dest_rank == r), 1, 0))
            if rank_count > 0:
                tl.atomic_add(dest_counts_ptr + r, rank_count)

    @triton.jit
    def _sparse_redirect_kernel(
        expanded_ids_ptr,  # [num_tokens, topk+1] - in/out
        local_mask_ptr,  # [num_tokens] - in/out
        dest_counts_ptr,  # [world_size] - destination counts
        num_tokens,
        topk_plus_one,
        old_experts_per_rank,  # Original experts per rank (e.g., 32)
        new_experts_per_rank,  # New experts per rank (e.g., 33)
        source_rank,
        min_tokens_per_rank,
        local_marker,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Redirect sparse remote destinations to local.

        In new layout, shared expert ID = rank * new_experts_per_rank + old_experts_per_rank
        So dest_rank = (shared_id - old_experts_per_rank) // new_experts_per_rank
                     = shared_id // new_experts_per_rank (since shared_id % new_epr == old_epr)
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        shared_expert_id = tl.load(
            expanded_ids_ptr + token_idx * topk_plus_one + (topk_plus_one - 1),
            mask=mask,
            other=-1,
        ).to(tl.int64)
        is_local = tl.load(local_mask_ptr + token_idx, mask=mask, other=True)

        # Use tl.full to create int64 constants (Python int doesn't have .to())
        src_rank_vec = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)
        # For shared expert: dest_rank = shared_expert_id // new_experts_per_rank
        dest_rank = tl.where(
            is_local, src_rank_vec, shared_expert_id // new_experts_per_rank
        )
        dest_rank = tl.minimum(tl.maximum(dest_rank, 0), 7)

        dest_count = tl.load(dest_counts_ptr + dest_rank, mask=mask, other=0)
        is_sparse_remote = (dest_count < min_tokens_per_rank) & ~is_local

        local_marker_vec = tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64)
        new_shared_id = tl.where(is_sparse_remote, local_marker_vec, shared_expert_id)
        new_is_local = is_local | is_sparse_remote

        tl.store(
            expanded_ids_ptr + token_idx * topk_plus_one + (topk_plus_one - 1),
            new_shared_id,
            mask=mask,
        )
        tl.store(local_mask_ptr + token_idx, new_is_local, mask=mask)

    def waterfill_prepare_dispatch_fused(
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
        num_routed_experts: int,
        world_size: int,
        source_rank: int,
        shared_weight: float,
        min_tokens_per_rank: int = 128,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fully fused waterfill using Triton with integrated histogram and expert ID remapping.

        Expert ID remapping: old_id -> old_id + (old_id // old_experts_per_rank)
        This maps original expert IDs to new layout where each rank has one extra expert slot
        for the shared expert.

        Single kernel does: waterfill + expand + histogram counting + ID remapping.
        Second kernel (if needed): sparse redirect.

        Returns:
            expanded_topk_ids: [N, 9] with remapped expert IDs
            expanded_topk_weights: [N, 9]
            local_shared_mask: [N] boolean
        """
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        old_experts_per_rank = num_routed_experts // world_size  # Original: 32
        new_experts_per_rank = old_experts_per_rank + 1  # New: 33
        device = topk_ids.device

        if num_tokens == 0:
            return (
                torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=device),
                torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )

        # Pre-allocate outputs
        expanded_topk_ids = torch.empty(
            num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
        )
        expanded_topk_weights = torch.empty(
            num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
        )
        local_shared_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)

        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        local_pref_numer = int(LOCAL_PREFERENCE_FACTOR * 5)
        local_pref_denom = 5

        if min_tokens_per_rank > 0:
            # Use fused kernel with histogram
            dest_counts = torch.zeros(world_size, dtype=torch.int32, device=device)

            _waterfill_expand_with_histogram_kernel[grid](
                topk_ids,
                topk_weights,
                routed_counts,
                expanded_topk_ids,
                expanded_topk_weights,
                local_shared_mask,
                dest_counts,
                num_tokens,
                topk,
                old_experts_per_rank,
                new_experts_per_rank,
                world_size,
                source_rank,
                shared_weight,
                LOCAL_SHARED_MARKER,
                local_pref_numer,
                local_pref_denom,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Launch sparse redirect kernel
            # Pass new_experts_per_rank so it correctly computes local shared expert ID
            _sparse_redirect_kernel[grid](
                expanded_topk_ids,
                local_shared_mask,
                dest_counts,
                num_tokens,
                topk + 1,
                old_experts_per_rank,
                new_experts_per_rank,
                source_rank,
                min_tokens_per_rank,
                LOCAL_SHARED_MARKER,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # No sparse handling needed, use simple fused kernel
            _waterfill_expand_topk_fused_kernel[grid](
                topk_ids,
                topk_weights,
                routed_counts,
                expanded_topk_ids,
                expanded_topk_weights,
                local_shared_mask,
                num_tokens,
                topk,
                old_experts_per_rank,
                new_experts_per_rank,
                world_size,
                source_rank,
                shared_weight,
                LOCAL_SHARED_MARKER,
                local_pref_numer,
                local_pref_denom,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask

    def identify_shared_expert_tokens_triton(
        recv_topk_ids: Tensor,
        num_experts: int,
        world_size: int,
        current_rank: int,
    ) -> Tensor:
        """
        Triton-optimized identify_shared_expert_tokens.

        Returns boolean mask (avoids nonzero).
        """
        num_tokens = recv_topk_ids.shape[0]
        topk_plus_one = recv_topk_ids.shape[1]
        experts_per_rank = num_experts // world_size
        device = recv_topk_ids.device

        if num_tokens == 0:
            return torch.empty(0, dtype=torch.bool, device=device)

        output_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)

        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _identify_shared_expert_kernel[grid](
            recv_topk_ids,
            output_mask,
            num_tokens,
            topk_plus_one,
            experts_per_rank,
            current_rank,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output_mask

    def count_routed_per_rank_triton(
        topk_ids: Tensor,
        num_experts: int,
        world_size: int,
    ) -> Tensor:
        """
        Triton-optimized count of routed tokens per rank.

        Replaces PyTorch bincount with a Triton kernel using
        block-level histogram to minimize atomic contention.
        """
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = num_experts // world_size
        device = topk_ids.device

        if num_tokens == 0:
            return torch.zeros(world_size, dtype=torch.int64, device=device)

        # Output histogram (atomic adds)
        counts = torch.zeros(world_size, dtype=torch.int64, device=device)

        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _count_routed_per_rank_kernel[grid](
            topk_ids,
            counts,
            num_tokens,
            topk,
            experts_per_rank,
            world_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return counts

    def masked_scatter_add_triton(
        output: Tensor,
        input: Tensor,
        mask: Tensor,
    ) -> None:
        """
        Scatter-add packed input to output using mask (in-place).

        Equivalent to:
            indices = mask.nonzero(as_tuple=True)[0]
            output.index_add_(0, indices, input)

        But avoids the expensive nonzero() call by using prefix sum.

        Args:
            output: [N, H] tensor to add to
            input: [num_selected, H] packed tensor where num_selected = mask.sum()
            mask: [N] boolean mask
        """
        num_tokens = output.shape[0]
        hidden_size = output.shape[1]

        if input.shape[0] == 0:
            return

        # Compute exclusive prefix sum of mask (int64 for indexing)
        mask_int = mask.to(torch.int64)
        # Exclusive prefix sum: prefix[i] = sum(mask[:i])
        prefix = torch.zeros(num_tokens + 1, dtype=torch.int64, device=mask.device)
        torch.cumsum(mask_int, dim=0, out=prefix[1:])
        prefix = prefix[:-1]  # Now prefix[i] = count of True in mask[:i]

        BLOCK_H = min(hidden_size, 256)
        grid = (num_tokens,)

        _masked_scatter_add_kernel[grid](
            output,
            input,
            prefix,
            mask,
            num_tokens,
            hidden_size,
            BLOCK_H=BLOCK_H,
        )

    def assign_shared_destination_triton(
        topk_ids: Tensor,
        routed_counts: Tensor,
        num_experts: int,
        world_size: int,
        source_rank: int,
    ) -> Tensor:
        """Triton-optimized shared destination assignment (standalone version)."""
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = num_experts // world_size
        device = topk_ids.device

        if num_tokens == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        # Use the fused kernel but only extract destination
        # This is less efficient than standalone, but kept for API compatibility
        expanded_ids, _, local_mask = waterfill_expand_topk_fused(
            topk_ids,
            torch.zeros(
                num_tokens, topk, dtype=torch.float32, device=device
            ),  # dummy weights
            routed_counts,
            num_experts,
            world_size,
            source_rank,
            0.0,  # dummy weight
        )

        # Extract destination from 9th column
        virtual_ids = expanded_ids[:, -1]
        destination = torch.where(
            local_mask,
            torch.full_like(virtual_ids, source_rank),
            virtual_ids // experts_per_rank,
        )

        return destination.to(torch.int64)


# ============== PyTorch Implementation ==============


def count_routed_per_rank_pytorch(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
) -> Tensor:
    """Count routed tokens per rank using PyTorch ops."""
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    valid_mask = topk_ids >= 0
    rank_ids = torch.where(
        valid_mask,
        torch.clamp(topk_ids // experts_per_rank, 0, world_size - 1),
        torch.full_like(topk_ids, world_size),  # Invalid -> out of range
    )

    flat_ranks = rank_ids.flatten()
    counts = torch.bincount(flat_ranks, minlength=world_size + 1)[:world_size]

    return counts.to(torch.int64)


def assign_shared_destination_pytorch(
    topk_ids: Tensor,
    routed_counts: Tensor,
    num_experts: int,
    world_size: int,
    source_rank: int,
) -> Tensor:
    """
    Assign shared expert destination for each token using waterfill.

    Strategy:
    1. For each token, find all ranks it routes to
    2. Add source_rank as a candidate (local computation option)
    3. Select the rank with lowest routed count

    Returns:
        destination: [num_tokens] destination rank for each token's shared expert
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Compute rank_ids: [num_tokens, topk]
    # For invalid expert IDs (< 0), use world_size as placeholder (will be filtered)
    valid_mask = topk_ids >= 0
    rank_ids = torch.where(
        valid_mask,
        torch.clamp(topk_ids // experts_per_rank, 0, world_size - 1),
        torch.full_like(topk_ids, world_size),  # Invalid -> out of range
    )

    # OPTIMIZED: Build candidate mask using scatter (vectorized, no loop)
    # Flatten rank_ids and create row indices
    # Shape: [num_tokens * topk]
    flat_rank_ids = rank_ids.flatten()
    row_indices = (
        torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, topk).flatten()
    )

    # Create candidate_mask using scatter
    # Note: use world_size+1 columns to handle invalid entries, then slice
    candidate_mask = torch.zeros(
        num_tokens, world_size + 1, dtype=torch.bool, device=device
    )
    candidate_mask[row_indices, flat_rank_ids] = True
    candidate_mask = candidate_mask[:, :world_size]  # Remove invalid column

    # Source rank is always a candidate
    candidate_mask[:, source_rank] = True

    # Select rank with minimum count among candidates (waterfill with local preference)
    # Apply local preference: scale remote counts by LOCAL_PREFERENCE_FACTOR
    # This makes local more attractive unless remote is significantly less loaded
    INF = routed_counts.max() * 10 + 1
    scaled_counts = routed_counts.unsqueeze(0) * LOCAL_PREFERENCE_FACTOR
    # Don't scale local rank
    scaled_counts[:, source_rank] = routed_counts[source_rank].float()
    candidate_counts = torch.where(candidate_mask, scaled_counts, INF)
    destination = candidate_counts.argmin(dim=1)

    return destination.to(torch.int64)


def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_destination: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Expand topk_ids/weights from [N, 8] to [N, 9] with shared expert as real expert.

    KEY CHANGE: Shared expert is now a real expert ID (not virtual).

    Expert ID layout (per rank):
    - [0, old_experts_per_rank-1]: routed experts
    - [old_experts_per_rank]: shared expert

    Expert ID remapping:
    - Routed expert j (old) -> j + (j // old_experts_per_rank) (new)
    - Shared expert for rank i -> i * new_experts_per_rank + old_experts_per_rank

    The 9th column contains:
    - Real shared expert ID: target_rank * new_experts_per_rank + old_experts_per_rank
    - This ensures DeepEP dispatches the token to the correct rank AND
      num_recv_tokens_per_expert correctly counts shared expert tokens.

    Returns:
        expanded_topk_ids: [N, 9] with remapped routed IDs and real shared expert ID
        expanded_topk_weights: [N, 9]
        local_shared_mask: [N] boolean mask for tokens with local shared expert
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Old and new experts per rank
    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1  # +1 for shared expert

    # Identify local vs remote shared expert
    local_shared_mask = shared_destination == source_rank

    # OPTIMIZED: Pre-allocate output tensors
    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )

    # Remap routed expert IDs: old_id -> old_id + (old_id // old_experts_per_rank)
    # This shifts each rank's experts to make room for shared expert
    # Example: rank 0 [0-31] -> [0-31], rank 1 [32-63] -> [33-64], rank 2 [64-95] -> [66-97], ...
    valid_mask = topk_ids >= 0
    old_ranks = torch.where(
        valid_mask, topk_ids // old_experts_per_rank, torch.zeros_like(topk_ids)
    )
    remapped_ids = torch.where(
        valid_mask,
        topk_ids + old_ranks,  # old_id + (old_id // old_experts_per_rank)
        topk_ids,  # keep -1 or invalid IDs unchanged
    )
    expanded_topk_ids[:, :topk] = remapped_ids

    # Compute real shared expert IDs: target_rank * new_experts_per_rank + old_experts_per_rank
    # This places shared expert at the end of each rank's expert range
    shared_expert_ids = shared_destination * new_experts_per_rank + old_experts_per_rank
    expanded_topk_ids[:, topk] = shared_expert_ids.to(topk_ids.dtype)

    # OPTIMIZED: Pre-allocate weights tensor
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = topk_weights
    expanded_topk_weights[:, topk] = shared_weight

    return expanded_topk_ids, expanded_topk_weights, local_shared_mask


# ============== Main API ==============


class DeepEPWaterfillBalancer:
    """
    Waterfill load balancer for DeepEP-based shared expert dispatch.

    This class implements the waterfill algorithm that assigns each token's
    shared expert computation to the least loaded rank among:
    1. Ranks it already routes to (no extra communication)
    2. Source rank (local computation)

    KEY DESIGN: Shared expert is fused as a real routed expert (not virtual ID).
    - num_experts is expanded: original + world_size (one shared per rank)
    - experts_per_rank = (num_routed_experts + world_size) // world_size
    - Each rank has: 32 routed experts + 1 shared expert = 33 experts
    - Expert IDs are remapped: old_id -> old_id + (old_id // old_experts_per_rank)
    - Shared expert ID for rank i = i * new_experts_per_rank + old_experts_per_rank

    This ensures num_recv_tokens_per_expert correctly counts shared expert tokens,
    and DeepGEMM processes the correct number of tokens without garbage data.
    """

    # Minimum batch size to enable waterfill balancing
    # Below this threshold, all shared experts are computed locally
    MIN_BATCH_FOR_BALANCE = 64

    # Minimum tokens to send to a remote rank for shared expert
    # If a rank would receive fewer tokens than this, compute locally instead
    # Set to 128 to ensure good tile utilization (typical tile size is 128)
    MIN_TOKENS_PER_RANK = 128

    def __init__(
        self,
        num_routed_experts: int,
        world_size: int,
        rank: int,
        routed_scaling_factor: float = 1.0,
    ):
        # Store original routed expert count
        self.num_routed_experts = num_routed_experts
        self.world_size = world_size
        self.rank = rank

        # Original experts per rank (before adding shared experts)
        self.old_experts_per_rank = num_routed_experts // world_size

        # New layout: each rank has old_experts_per_rank + 1 (shared) experts
        self.new_experts_per_rank = self.old_experts_per_rank + 1

        # Total experts including fused shared experts
        self.num_experts = self.new_experts_per_rank * world_size

        # For backward compatibility
        self.experts_per_rank = self.new_experts_per_rank

        self.routed_scaling_factor = routed_scaling_factor
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )

        # Shared expert ID for this rank
        # Layout: [routed_0, routed_1, ..., routed_31, shared] for each rank
        # So shared expert ID = rank * new_experts_per_rank + old_experts_per_rank
        self.my_shared_expert_id = (
            self.rank * self.new_experts_per_rank + self.old_experts_per_rank
        )

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank from local topk_ids.

        Uses Triton kernel on GPU for better performance, falls back to PyTorch on CPU.

        Note: topk_ids contains ORIGINAL expert IDs (0-255), so we use
        num_routed_experts to calculate experts_per_rank for rank assignment.
        """
        if HAS_TRITON and topk_ids.is_cuda:
            return count_routed_per_rank_triton(
                topk_ids, self.num_routed_experts, self.world_size
            )
        else:
            return count_routed_per_rank_pytorch(
                topk_ids, self.num_routed_experts, self.world_size
            )

    def assign_shared_destination(
        self, topk_ids: Tensor, routed_counts: Tensor
    ) -> Tensor:
        """Assign shared expert destination for each token using waterfill.

        Uses Triton kernel on GPU for better performance, falls back to PyTorch on CPU.

        Note: topk_ids contains ORIGINAL expert IDs (0-255), so we use
        num_routed_experts to calculate experts_per_rank for rank assignment.
        """
        # Use Triton kernel on GPU if available
        if HAS_TRITON and topk_ids.is_cuda:
            return assign_shared_destination_triton(
                topk_ids,
                routed_counts,
                self.num_routed_experts,
                self.world_size,
                self.rank,
            )
        else:
            return assign_shared_destination_pytorch(
                topk_ids,
                routed_counts,
                self.num_routed_experts,
                self.world_size,
                self.rank,
            )

    def prepare_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare expanded topk for dispatch with shared expert as 9th expert.

        Uses fused Triton kernel on GPU for maximum performance.

        Optimizations:
        1. Fused kernel: waterfill + expand in single GPU pass
        2. If batch size < MIN_BATCH_FOR_BALANCE, all shared experts compute locally
        3. If a remote rank would receive < MIN_TOKENS_PER_RANK, compute locally instead

        Returns:
            expanded_topk_ids: [N, 9] with virtual expert ID or LOCAL_SHARED_MARKER
            expanded_topk_weights: [N, 9] with shared_weight in 9th column
            local_shared_mask: [N] boolean mask for tokens with local shared expert
        """
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        device = topk_ids.device

        if num_tokens == 0:
            # Empty batch
            return (
                torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=device),
                torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )

        # Small batch optimization: all shared experts compute locally
        if num_tokens < self.MIN_BATCH_FOR_BALANCE:
            if DEEPEP_WATERFILL_DEBUG:
                print(
                    f"[DeepEP Waterfill] rank={self.rank} "
                    f"tokens={num_tokens} < MIN_BATCH={self.MIN_BATCH_FOR_BALANCE}, "
                    f"all shared experts computed locally"
                )
            # Fast path: all local, no waterfill needed
            # Still need to remap expert IDs to new layout
            expanded_topk_ids = torch.empty(
                num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
            )

            # Remap routed expert IDs: old_id -> old_id + (old_id // old_experts_per_rank)
            valid_mask = topk_ids >= 0
            old_ranks = torch.where(
                valid_mask,
                topk_ids // self.old_experts_per_rank,
                torch.zeros_like(topk_ids),
            )
            remapped_ids = torch.where(valid_mask, topk_ids + old_ranks, topk_ids)
            expanded_topk_ids[:, :topk] = remapped_ids

            # Local shared expert ID
            expanded_topk_ids[:, topk] = self.my_shared_expert_id

            expanded_topk_weights = torch.empty(
                num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
            )
            expanded_topk_weights[:, :topk] = topk_weights
            expanded_topk_weights[:, topk] = self.shared_weight

            local_shared_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            return expanded_topk_ids, expanded_topk_weights, local_shared_mask

        # ===== Use Fully Fused Triton Kernel on GPU =====
        # This combines waterfill + expand + sparse handling in minimal kernel launches
        if HAS_TRITON and topk_ids.is_cuda:
            expanded_topk_ids, expanded_topk_weights, local_shared_mask = (
                waterfill_prepare_dispatch_fused(
                    topk_ids,
                    topk_weights,
                    routed_counts,
                    self.num_routed_experts,  # Use num_routed_experts (original count)
                    self.world_size,
                    self.rank,
                    self.shared_weight,
                    self.MIN_TOKENS_PER_RANK,
                )
            )
        else:
            # Fallback to PyTorch implementation
            shared_destination = assign_shared_destination_pytorch(
                topk_ids,
                routed_counts,
                self.num_routed_experts,
                self.world_size,
                self.rank,
            )
            expanded_topk_ids, expanded_topk_weights, local_shared_mask = (
                expand_topk_with_shared_expert(
                    topk_ids,
                    topk_weights,
                    shared_destination,
                    self.num_routed_experts,
                    self.world_size,
                    self.rank,
                    self.shared_weight,
                )
            )

            # PyTorch fallback: post-processing for sparse handling
            # Note: shared expert IDs are now real IDs, not virtual
            if self.MIN_TOKENS_PER_RANK > 0:
                shared_ids = expanded_topk_ids[:, -1]
                # Extract destination rank from real shared expert ID
                # shared_id = target_rank * new_experts_per_rank + old_experts_per_rank
                dest_from_shared = shared_ids // self.new_experts_per_rank
                dest_counts = torch.bincount(
                    dest_from_shared.to(torch.int64), minlength=self.world_size
                )
                sparse_ranks_mask = dest_counts < self.MIN_TOKENS_PER_RANK
                sparse_ranks_mask[self.rank] = False
                token_goes_to_sparse = (
                    sparse_ranks_mask[dest_from_shared.long()] & ~local_shared_mask
                )
                # Redirect sparse tokens to local shared expert
                expanded_topk_ids[:, -1] = torch.where(
                    token_goes_to_sparse,
                    torch.tensor(
                        self.my_shared_expert_id,
                        dtype=expanded_topk_ids.dtype,
                        device=expanded_topk_ids.device,
                    ),
                    expanded_topk_ids[:, -1],
                )
                local_shared_mask = local_shared_mask | token_goes_to_sparse

        # Periodic logging (only when DEBUG enabled to avoid sync)
        if DEEPEP_WATERFILL_DEBUG:
            global _waterfill_log_counter
            _waterfill_log_counter[0] += 1
            if _waterfill_log_counter[0] % _WATERFILL_LOG_INTERVAL == 1:
                num_local = local_shared_mask.sum().item()
                num_remote = num_tokens - num_local
                print(
                    f"[DeepEP Waterfill] rank={self.rank} "
                    f"call={_waterfill_log_counter[0]} "
                    f"tokens={num_tokens} "
                    f"local={num_local} remote={num_remote}"
                )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask


def identify_shared_expert_tokens(
    recv_topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    current_rank: int,
    return_mask: bool = False,
) -> Tensor:
    """
    Identify which received tokens need shared expert computation on this rank.

    A token needs shared expert here if its 9th column (virtual expert ID)
    maps to current_rank. Tokens with LOCAL_SHARED_MARKER (-1) are skipped
    (they were computed locally on source rank).

    Uses Triton kernel on GPU for better performance, falls back to PyTorch on CPU.

    Args:
        recv_topk_ids: [N, 9] received topk IDs with virtual expert in 9th column
        num_experts: total number of experts
        world_size: number of ranks
        current_rank: this rank's ID
        return_mask: if True, return boolean mask instead of indices (avoids nonzero)

    Returns:
        If return_mask=False: shared_indices - indices of tokens needing shared expert
        If return_mask=True: shared_mask - boolean mask for shared expert tokens
    """
    # Use Triton kernel on GPU for mask computation
    if HAS_TRITON and recv_topk_ids.is_cuda:
        shared_mask = identify_shared_expert_tokens_triton(
            recv_topk_ids, num_experts, world_size, current_rank
        )
        if return_mask:
            return shared_mask
        return shared_mask.nonzero(as_tuple=True)[0]

    # PyTorch fallback
    experts_per_rank = num_experts // world_size

    # 9th column contains virtual expert ID or LOCAL_SHARED_MARKER
    virtual_expert_ids = recv_topk_ids[:, -1]

    # Skip LOCAL_SHARED_MARKER tokens (they stay on source rank)
    valid_mask = virtual_expert_ids >= 0

    # Check if virtual ID maps to current rank
    target_ranks = virtual_expert_ids // experts_per_rank
    shared_mask = valid_mask & (target_ranks == current_rank)

    if return_mask:
        return shared_mask

    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    return shared_indices


def compute_local_shared_expert(
    hidden_states: Tensor,
    local_shared_mask: Tensor,
    shared_expert_fn,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Compute shared expert locally for tokens marked as local.

    Uses boolean indexing for efficient token selection.

    Local shared expert output is NOT weighted by 1/rsf because it will be
    added AFTER the routed_scaling_factor multiplication.

    Args:
        hidden_states: [N, H] input hidden states
        local_shared_mask: [N] boolean mask for local shared expert tokens
        shared_expert_fn: function to compute shared expert

    Returns:
        local_shared_output: [num_local, H] output (or None if no local tokens)
        local_indices: [num_local] indices of local tokens (or None)
    """
    # Boolean indexing for efficient token selection
    local_hidden = hidden_states[local_shared_mask]

    # Early exit if no local tokens (shape check, no CPU-GPU sync)
    if local_hidden.shape[0] == 0:
        return None, None

    local_output = shared_expert_fn(local_hidden)

    # Compute indices for index_add_ in caller
    local_indices = local_shared_mask.nonzero(as_tuple=True)[0]

    return local_output, local_indices


def compute_local_shared_expert_inplace(
    hidden_states: Tensor,
    local_shared_mask: Tensor,
    shared_expert_fn,
    output: Tensor,
) -> bool:
    """
    Compute shared expert locally and add to output in-place.

    Uses index_add_ which is faster than boolean indexing for scatter.
    The nonzero call is unavoidable for index_add_, but we skip the .any() check.

    Args:
        hidden_states: [N, H] input hidden states
        local_shared_mask: [N] boolean mask for local shared expert tokens
        shared_expert_fn: function to compute shared expert
        output: [N, H] output tensor to add results to (modified in-place)

    Returns:
        has_local: True if there were local tokens to process
    """
    # Boolean indexing for gather (efficient)
    local_hidden = hidden_states[local_shared_mask]

    if local_hidden.shape[0] == 0:
        return False

    local_output = shared_expert_fn(local_hidden)

    # Use index_add_ which is faster than boolean scatter
    local_indices = local_shared_mask.nonzero(as_tuple=True)[0]
    output.index_add_(0, local_indices, local_output)

    return True


def compute_remote_shared_expert_inplace(
    recv_hidden: Tensor,
    recv_topk_ids: Tensor,
    recv_topk_weights: Tensor,
    num_experts: int,
    world_size: int,
    current_rank: int,
    shared_expert_fn,
    output: Tensor,
    apply_weight: bool = True,
) -> bool:
    """
    Identify and compute remote shared expert tokens in-place.

    Combines identify + compute + scatter into one function to reduce overhead.
    Uses index_add_ for efficient scatter.

    Args:
        recv_hidden: [M, H] received hidden states
        recv_topk_ids: [M, 9] received topk IDs with virtual expert in 9th column
        recv_topk_weights: [M, 9] received topk weights
        num_experts: total number of experts
        world_size: number of ranks
        current_rank: this rank's ID
        shared_expert_fn: function to compute shared expert
        output: [M, H] output tensor to add results to (modified in-place)
        apply_weight: whether to apply the weight from 9th column

    Returns:
        has_remote: True if there were remote shared tokens to process
    """
    if recv_hidden.shape[0] == 0:
        return False

    experts_per_rank = num_experts // world_size

    # Compute shared_mask directly
    virtual_expert_ids = recv_topk_ids[:, -1]
    valid_mask = virtual_expert_ids >= 0
    target_ranks = virtual_expert_ids // experts_per_rank
    shared_mask = valid_mask & (target_ranks == current_rank)

    # Get indices for index_add_
    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    if shared_indices.shape[0] == 0:
        return False

    # Gather hidden states
    remote_hidden = recv_hidden[shared_indices]

    # Compute shared expert
    remote_output = shared_expert_fn(remote_hidden)

    # Apply weight if needed
    if apply_weight:
        weights = recv_topk_weights[shared_indices, -1].unsqueeze(-1)
        remote_output = remote_output * weights

    # Use index_add_ for efficient scatter
    output.index_add_(0, shared_indices, remote_output)

    return True
