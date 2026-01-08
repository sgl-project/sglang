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
        topk_ids_ptr,           # [num_tokens, topk]
        topk_weights_ptr,       # [num_tokens, topk]
        routed_counts_ptr,      # [world_size]
        # Outputs
        expanded_ids_ptr,       # [num_tokens, topk+1]
        expanded_weights_ptr,   # [num_tokens, topk+1]
        local_mask_ptr,         # [num_tokens]
        # Scalars
        num_tokens,
        topk: tl.constexpr,
        experts_per_rank,
        world_size,
        source_rank,
        shared_weight,
        local_marker,           # LOCAL_SHARED_MARKER = -1
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused Triton kernel for waterfill assignment + topk expansion.
        
        For each token:
        1. Find all ranks it routes to (from topk_ids)
        2. Select the rank with minimum routed_count (waterfill)
        3. Expand topk_ids/weights to include shared expert
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
        best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)
        
        # Check each routed expert and update if better
        for k in range(topk):
            # Load expert ID
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, 
                mask=mask, 
                other=-1
            )
            valid = expert_id >= 0
            
            # Compute target rank from expert ID
            target_rank = expert_id // experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            
            # Load routed count for this rank
            target_count = tl.load(
                routed_counts_ptr + target_rank, 
                mask=mask & valid, 
                other=2**30
            )
            
            # Update if this rank has lower count (waterfill)
            better = (target_count < best_count) & valid & mask
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)
        
        # ===== Step 2: Compute virtual expert ID and local mask =====
        is_local = (best_rank == source_rank)
        
        # Virtual expert ID: dest * experts_per_rank, or local_marker if local
        virtual_id = tl.where(
            is_local,
            local_marker,
            best_rank * experts_per_rank
        )
        
        # ===== Step 3: Copy original topk_ids and topk_weights =====
        # Copy topk_ids columns
        for k in range(topk):
            val = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=0)
            tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, val, mask=mask)
        
        # Copy topk_weights columns
        for k in range(topk):
            val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
            tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)
        
        # ===== Step 4: Write 9th column (shared expert) =====
        tl.store(
            expanded_ids_ptr + token_idx * (topk + 1) + topk, 
            virtual_id, 
            mask=mask
        )
        tl.store(
            expanded_weights_ptr + token_idx * (topk + 1) + topk,
            shared_weight,
            mask=mask
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
        expanded_topk_ids = torch.empty(num_tokens, topk + 1, dtype=topk_ids.dtype, device=device)
        expanded_topk_weights = torch.empty(num_tokens, topk + 1, dtype=topk_weights.dtype, device=device)
        local_shared_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)
        
        # Launch fused kernel
        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
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
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return expanded_topk_ids, expanded_topk_weights, local_shared_mask


    @triton.jit
    def _count_destinations_kernel(
        destination_ptr,    # [num_tokens] - destination rank for each token
        counts_ptr,         # [world_size] - output counts (atomic add)
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
            torch.zeros(num_tokens, topk, dtype=torch.float32, device=device),  # dummy weights
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
    row_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, topk).flatten()
    
    # Create candidate_mask using scatter
    # Note: use world_size+1 columns to handle invalid entries, then slice
    candidate_mask = torch.zeros(num_tokens, world_size + 1, dtype=torch.bool, device=device)
    candidate_mask[row_indices, flat_rank_ids] = True
    candidate_mask = candidate_mask[:, :world_size]  # Remove invalid column
    
    # Source rank is always a candidate
    candidate_mask[:, source_rank] = True

    # Select rank with minimum count among candidates (waterfill)
    INF = routed_counts.max() + 1
    candidate_counts = torch.where(candidate_mask, routed_counts.unsqueeze(0), INF)
    destination = candidate_counts.argmin(dim=1)

    return destination.to(torch.int64)


def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_destination: Tensor,
    num_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Expand topk_ids/weights from [N, 8] to [N, 9] with shared expert info.

    The 9th column contains:
    - LOCAL_SHARED_MARKER (-1): if destination == source_rank (compute locally)
    - virtual_expert_id: if destination != source_rank (dispatch to target rank)
    
    virtual_expert_id = target_rank * experts_per_rank
    This ensures DeepEP dispatches the token to the correct rank.
    
    Returns:
        expanded_topk_ids: [N, 9]
        expanded_topk_weights: [N, 9]
        local_shared_mask: [N] boolean mask for tokens with local shared expert
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    experts_per_rank = num_experts // world_size

    # Identify local vs remote shared expert
    local_shared_mask = shared_destination == source_rank
    
    # OPTIMIZED: Pre-allocate output tensors to avoid cat overhead
    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_ids[:, :topk] = topk_ids
    
    # Compute virtual expert IDs: dest * experts_per_rank for remote, -1 for local
    # Use in-place operations where possible
    virtual_expert_ids = shared_destination * experts_per_rank
    virtual_expert_ids[local_shared_mask] = LOCAL_SHARED_MARKER
    expanded_topk_ids[:, topk] = virtual_expert_ids.to(topk_ids.dtype)
    
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

    The shared expert is encoded as a virtual 9th expert in topk_ids.
    Local computation is marked with LOCAL_SHARED_MARKER (-1).
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
        num_experts: int,
        world_size: int,
        rank: int,
        routed_scaling_factor: float = 1.0,
    ):
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.experts_per_rank = num_experts // world_size
        self.routed_scaling_factor = routed_scaling_factor
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank from local topk_ids."""
        return count_routed_per_rank_pytorch(
            topk_ids, self.num_experts, self.world_size
        )

    def assign_shared_destination(
        self, topk_ids: Tensor, routed_counts: Tensor
    ) -> Tensor:
        """Assign shared expert destination for each token using waterfill.
        
        Uses Triton kernel on GPU for better performance, falls back to PyTorch on CPU.
        """
        # Use Triton kernel on GPU if available
        if HAS_TRITON and topk_ids.is_cuda:
            return assign_shared_destination_triton(
                topk_ids, routed_counts, self.num_experts, self.world_size, self.rank
            )
        else:
            return assign_shared_destination_pytorch(
                topk_ids, routed_counts, self.num_experts, self.world_size, self.rank
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
            expanded_topk_ids = torch.empty(num_tokens, topk + 1, dtype=topk_ids.dtype, device=device)
            expanded_topk_ids[:, :topk] = topk_ids
            expanded_topk_ids[:, topk] = LOCAL_SHARED_MARKER
            
            expanded_topk_weights = torch.empty(num_tokens, topk + 1, dtype=topk_weights.dtype, device=device)
            expanded_topk_weights[:, :topk] = topk_weights
            expanded_topk_weights[:, topk] = self.shared_weight
            
            local_shared_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            return expanded_topk_ids, expanded_topk_weights, local_shared_mask

        # ===== Use Fused Triton Kernel on GPU =====
        if HAS_TRITON and topk_ids.is_cuda:
            expanded_topk_ids, expanded_topk_weights, local_shared_mask = waterfill_expand_topk_fused(
                topk_ids,
                topk_weights,
                routed_counts,
                self.num_experts,
                self.world_size,
                self.rank,
                self.shared_weight,
            )
        else:
            # Fallback to PyTorch implementation
            shared_destination = assign_shared_destination_pytorch(
                topk_ids, routed_counts, self.num_experts, self.world_size, self.rank
            )
            expanded_topk_ids, expanded_topk_weights, local_shared_mask = expand_topk_with_shared_expert(
                topk_ids, topk_weights, shared_destination,
                self.num_experts, self.world_size, self.rank, self.shared_weight,
            )

        # ===== Post-processing: Handle sparse destinations (vectorized) =====
        # This is done on GPU with minimal CPU sync
        
        # Extract destinations from virtual IDs
        virtual_ids = expanded_topk_ids[:, -1]
        
        # Compute destination for each token
        dest_from_virtual = torch.where(
            local_shared_mask,
            torch.full_like(virtual_ids, self.rank),
            virtual_ids // self.experts_per_rank,
        )
        
        # Count tokens per destination rank
        dest_counts = torch.bincount(dest_from_virtual.to(torch.int64), minlength=self.world_size)
        
        # Find sparse remote ranks (those receiving < MIN_TOKENS_PER_RANK)
        sparse_ranks_mask = dest_counts < self.MIN_TOKENS_PER_RANK
        sparse_ranks_mask[self.rank] = False  # Don't touch local
        
        # VECTORIZED: Redirect all sparse remote tokens to local in one shot
        # Check which tokens go to sparse ranks
        token_goes_to_sparse = sparse_ranks_mask[dest_from_virtual.long()] & ~local_shared_mask
        
        if token_goes_to_sparse.any():
            expanded_topk_ids[token_goes_to_sparse, -1] = LOCAL_SHARED_MARKER
            local_shared_mask = local_shared_mask | token_goes_to_sparse
            
            if DEEPEP_WATERFILL_DEBUG:
                print(
                    f"[DeepEP Waterfill] rank={self.rank} "
                    f"redirected {token_goes_to_sparse.sum().item()} sparse tokens to local"
                )

        # VECTORIZED: Handle case where local count is too small
        # Move all local to best remote rank
        local_count = local_shared_mask.sum()
        has_sparse_local = (local_count > 0) & (local_count < self.MIN_TOKENS_PER_RANK)
        
        if has_sparse_local:
            # Find best remote rank (one with most tokens)
            remote_dest_counts = dest_counts.clone()
            remote_dest_counts[self.rank] = -1  # Exclude local
            best_remote_rank = remote_dest_counts.argmax()
            
            if remote_dest_counts[best_remote_rank] > 0:
                # Redirect all local to best remote
                expanded_topk_ids[local_shared_mask, -1] = best_remote_rank * self.experts_per_rank
                local_shared_mask = torch.zeros_like(local_shared_mask)
                
                if DEEPEP_WATERFILL_DEBUG:
                    print(
                        f"[DeepEP Waterfill] rank={self.rank} "
                        f"local_count={local_count.item()} < MIN={self.MIN_TOKENS_PER_RANK}, "
                        f"redirecting to rank {best_remote_rank.item()}"
                    )

        if DEEPEP_WATERFILL_DEBUG:
            num_local = local_shared_mask.sum().item()
            num_remote = num_tokens - num_local
            print(
                f"[DeepEP Waterfill] rank={self.rank} "
                f"tokens={num_tokens} "
                f"local_shared={num_local} remote_shared={num_remote}"
            )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask


def identify_shared_expert_tokens(
    recv_topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    current_rank: int,
) -> Tensor:
    """
    Identify which received tokens need shared expert computation on this rank.

    A token needs shared expert here if its 9th column (virtual expert ID)
    maps to current_rank. Tokens with LOCAL_SHARED_MARKER (-1) are skipped
    (they were computed locally on source rank).

    Returns:
        shared_indices: indices of tokens needing shared expert computation
    """
    experts_per_rank = num_experts // world_size

    # 9th column contains virtual expert ID or LOCAL_SHARED_MARKER
    virtual_expert_ids = recv_topk_ids[:, -1]
    
    # Skip LOCAL_SHARED_MARKER tokens (they stay on source rank)
    valid_mask = virtual_expert_ids >= 0
    
    # Check if virtual ID maps to current rank
    target_ranks = virtual_expert_ids // experts_per_rank
    shared_mask = valid_mask & (target_ranks == current_rank)
    
    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    return shared_indices


def compute_local_shared_expert(
    hidden_states: Tensor,
    local_shared_mask: Tensor,
    shared_expert_fn,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Compute shared expert locally for tokens marked as local.
    
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
    if not local_shared_mask.any():
        return None, None
    
    local_indices = local_shared_mask.nonzero(as_tuple=True)[0]
    local_hidden = hidden_states[local_indices]
    local_output = shared_expert_fn(local_hidden)
    
    # NO weight applied here - local shared is added after rsf multiplication
    return local_output, local_indices
