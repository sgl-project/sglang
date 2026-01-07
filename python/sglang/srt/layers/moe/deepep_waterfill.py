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

This module implements waterfill load balancing for shared expert computation
using DeepEP communication. The key idea is to treat shared expert as the 9th
expert and dispatch it through DeepEP along with routed experts.

Design principles:
1. Each token's shared expert can be sent to:
   - One of the ranks it already routes to (no extra communication)
   - Or stay at source rank for local computation
2. Waterfill algorithm selects the lowest-loaded rank from candidates
3. Shared expert weight = 1.0 / routed_scaling_factor (for correct combine)
"""

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Environment variables
DEEPEP_WATERFILL_DEBUG = os.environ.get("SGLANG_DEEPEP_WATERFILL_DEBUG", "0") == "1"

# Special expert ID for shared expert (assuming 256 routed experts)
SHARED_EXPERT_ID = 256


# ============== Triton Kernels ==============

if HAS_TRITON:

    @triton.jit
    def _count_routed_per_rank_kernel(
        topk_ids_ptr,  # [num_tokens, topk]
        counts_ptr,  # [world_size] output
        num_tokens,
        topk: tl.constexpr,
        experts_per_rank: tl.constexpr,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Count routed tokens per rank.
        Each token contributes to multiple ranks based on its topk expert selections.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE

        # Local histogram
        local_hist = tl.zeros([8], dtype=tl.int32)
        offs = tl.arange(0, 8)

        for i in range(BLOCK_SIZE):
            token_idx = block_start + i
            if token_idx < num_tokens:
                base_ptr = topk_ids_ptr + token_idx * topk
                for k in range(topk):
                    expert_id = tl.load(base_ptr + k)
                    if expert_id >= 0:  # Skip invalid experts
                        rank_id = expert_id // experts_per_rank
                        rank_id = tl.minimum(rank_id, world_size - 1)
                        local_hist = tl.where(offs == rank_id, local_hist + 1, local_hist)

        # Atomic add to global histogram
        for r in range(world_size):
            count = tl.sum(tl.where(offs == r, local_hist, 0))
            if count > 0:
                tl.atomic_add(counts_ptr + r, count)

    @triton.jit
    def _assign_shared_destination_kernel(
        topk_ids_ptr,  # [num_tokens, topk]
        routed_counts_ptr,  # [world_size] global routed counts
        destination_ptr,  # [num_tokens] output: destination rank for shared expert
        num_tokens,
        topk: tl.constexpr,
        experts_per_rank: tl.constexpr,
        world_size: tl.constexpr,
        source_rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Assign shared expert destination for each token.

        For each token:
        1. Extract candidate ranks (routed ranks + source_rank)
        2. Select the rank with lowest routed count
        """
        pid = tl.program_id(0)
        token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = token_idx < num_tokens

        # Load global routed counts
        rank_offs = tl.arange(0, 8)
        counts = tl.load(routed_counts_ptr + rank_offs, mask=rank_offs < world_size, other=0x7FFFFFFF)

        for i in range(BLOCK_SIZE):
            tid = pid * BLOCK_SIZE + i
            if tid < num_tokens:
                base_ptr = topk_ids_ptr + tid * topk

                # Build candidate mask: ranks this token routes to + source_rank
                candidate_mask = tl.zeros([8], dtype=tl.int32)
                candidate_mask = tl.where(rank_offs == source_rank, 1, candidate_mask)

                for k in range(topk):
                    expert_id = tl.load(base_ptr + k)
                    if expert_id >= 0:
                        rank_id = expert_id // experts_per_rank
                        rank_id = tl.minimum(rank_id, world_size - 1)
                        candidate_mask = tl.where(rank_offs == rank_id, 1, candidate_mask)

                # Find minimum count among candidates
                candidate_counts = tl.where(candidate_mask == 1, counts, 0x7FFFFFFF)
                min_count = tl.min(candidate_counts)

                # Select first rank with minimum count
                is_min = (candidate_counts == min_count).to(tl.int32)
                cumsum = tl.cumsum(is_min, axis=0)
                first_min_mask = (is_min == 1) & (cumsum == 1)
                dest_rank = tl.sum(tl.where(first_min_mask, rank_offs, 0))

                tl.store(destination_ptr + tid, dest_rank)


# ============== PyTorch Implementation ==============


def count_routed_per_rank_pytorch(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
) -> Tensor:
    """
    Count routed tokens per rank using PyTorch ops.

    Args:
        topk_ids: [num_tokens, topk] tensor of expert IDs
        num_experts: Total number of routed experts
        world_size: Number of ranks

    Returns:
        counts: [world_size] tensor of token counts per rank
    """
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    # Convert expert IDs to rank IDs
    valid_mask = topk_ids >= 0
    rank_ids = torch.where(
        valid_mask, topk_ids // experts_per_rank, torch.full_like(topk_ids, world_size)
    )
    rank_ids = torch.clamp(rank_ids, 0, world_size)

    # Count tokens per rank
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
    Assign shared expert destination for each token using PyTorch ops.

    Strategy:
    1. For each token, find all ranks it routes to
    2. Add source_rank as a candidate (local computation option)
    3. Select the rank with lowest routed count

    Args:
        topk_ids: [num_tokens, topk] tensor of expert IDs
        routed_counts: [world_size] tensor of global routed token counts
        num_experts: Total number of routed experts
        world_size: Number of ranks
        source_rank: Current rank ID

    Returns:
        destination: [num_tokens] tensor of destination ranks for shared expert
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Build candidate mask for each token: [num_tokens, world_size]
    # candidate_mask[i, r] = 1 if token i can send shared expert to rank r
    candidate_mask = torch.zeros(num_tokens, world_size, dtype=torch.bool, device=device)

    # Source rank is always a candidate
    candidate_mask[:, source_rank] = True

    # Add routed ranks as candidates
    valid_mask = topk_ids >= 0
    rank_ids = torch.where(
        valid_mask,
        torch.clamp(topk_ids // experts_per_rank, 0, world_size - 1),
        torch.zeros_like(topk_ids),
    )

    # Scatter to mark routed ranks
    for k in range(topk):
        token_indices = torch.arange(num_tokens, device=device)
        valid = valid_mask[:, k]
        ranks = rank_ids[:, k]
        candidate_mask[token_indices[valid], ranks[valid]] = True

    # Select rank with minimum count among candidates
    # Set non-candidate ranks to infinity
    INF = routed_counts.max() + 1
    candidate_counts = torch.where(candidate_mask, routed_counts.unsqueeze(0), INF)

    # Select minimum count rank
    destination = candidate_counts.argmin(dim=1)

    return destination


def expand_topk_for_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_destination: Tensor,
    shared_expert_id: int,
    shared_weight: float,
    source_rank: int,
) -> Tuple[Tensor, Tensor]:
    """
    Expand topk_ids and topk_weights to include shared expert.

    Args:
        topk_ids: [num_tokens, topk] original expert IDs
        topk_weights: [num_tokens, topk] original expert weights
        shared_destination: [num_tokens] destination ranks for shared expert
        shared_expert_id: Expert ID for shared expert (e.g., 256)
        shared_weight: Weight for shared expert (1.0 / routed_scaling_factor)
        source_rank: Current rank ID

    Returns:
        expanded_topk_ids: [num_tokens, topk+1]
        expanded_topk_weights: [num_tokens, topk+1]
    """
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device

    # Create expanded tensors
    expanded_topk_ids = torch.cat(
        [topk_ids, torch.full((num_tokens, 1), -1, dtype=topk_ids.dtype, device=device)],
        dim=1,
    )
    expanded_topk_weights = torch.cat(
        [topk_weights, torch.zeros((num_tokens, 1), dtype=topk_weights.dtype, device=device)],
        dim=1,
    )

    # Set shared expert ID and weight for tokens that will be dispatched
    # Tokens staying at source_rank will have shared_expert_id, others will use -1
    # Actually, all tokens need shared expert computed, so we set the ID
    # The destination is encoded in the expert_id: shared_expert_id + destination_rank
    # Or we can use a separate mechanism

    # For simplicity, use shared_expert_id for all tokens
    # The destination is determined by which rank receives the token
    expanded_topk_ids[:, -1] = shared_expert_id
    expanded_topk_weights[:, -1] = shared_weight

    return expanded_topk_ids, expanded_topk_weights


# ============== Main API ==============


class DeepEPWaterfillBalancer:
    """
    Waterfill load balancer for DeepEP-based shared expert dispatch.

    Usage:
        balancer = DeepEPWaterfillBalancer(num_experts=256, world_size=8, rank=0)
        expanded_topk = balancer.prepare_dispatch(topk_ids, topk_weights, routed_counts)
    """

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        num_experts: int,
        world_size: int,
        rank: int,
        routed_scaling_factor: float = 1.0,
        use_triton: bool = True,
    ):
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.routed_scaling_factor = routed_scaling_factor
        self.shared_weight = 1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        self.use_triton = use_triton and HAS_TRITON

        # Shared expert ID
        self.shared_expert_id = SHARED_EXPERT_ID

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank from local topk_ids."""
        if self.use_triton and topk_ids.shape[0] > 0:
            return self._count_routed_triton(topk_ids)
        else:
            return count_routed_per_rank_pytorch(
                topk_ids, self.num_experts, self.world_size
            )

    def _count_routed_triton(self, topk_ids: Tensor) -> Tensor:
        """Triton implementation of routed token counting."""
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = self.num_experts // self.world_size
        device = topk_ids.device

        counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)

        BLOCK_SIZE = 64
        num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

        _count_routed_per_rank_kernel[(num_blocks,)](
            topk_ids,
            counts,
            num_tokens,
            topk,
            experts_per_rank,
            self.world_size,
            BLOCK_SIZE,
        )

        return counts.to(torch.int64)

    def assign_shared_destination(
        self, topk_ids: Tensor, routed_counts: Tensor
    ) -> Tensor:
        """
        Assign shared expert destination for each token.

        Args:
            topk_ids: [num_tokens, topk] local expert IDs
            routed_counts: [world_size] global routed token counts (after AllReduce)

        Returns:
            destination: [num_tokens] destination ranks for shared expert
        """
        if self.use_triton and topk_ids.shape[0] > self.MIN_BATCH_FOR_BALANCE:
            return self._assign_destination_triton(topk_ids, routed_counts)
        else:
            return assign_shared_destination_pytorch(
                topk_ids, routed_counts, self.num_experts, self.world_size, self.rank
            )

    def _assign_destination_triton(
        self, topk_ids: Tensor, routed_counts: Tensor
    ) -> Tensor:
        """Triton implementation of destination assignment."""
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = self.num_experts // self.world_size
        device = topk_ids.device

        destination = torch.empty(num_tokens, dtype=torch.int32, device=device)

        BLOCK_SIZE = 1
        num_blocks = num_tokens

        _assign_shared_destination_kernel[(num_blocks,)](
            topk_ids,
            routed_counts.to(torch.int32),
            destination,
            num_tokens,
            topk,
            experts_per_rank,
            self.world_size,
            self.rank,
            BLOCK_SIZE,
        )

        return destination.to(torch.int64)

    def prepare_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare expanded topk for dispatch with shared expert.

        Args:
            topk_ids: [num_tokens, topk] original expert IDs
            topk_weights: [num_tokens, topk] original expert weights
            routed_counts: [world_size] global routed token counts

        Returns:
            expanded_topk_ids: [num_tokens, topk+1]
            expanded_topk_weights: [num_tokens, topk+1]
            shared_destination: [num_tokens] destination ranks
        """
        # Assign shared expert destination
        shared_destination = self.assign_shared_destination(topk_ids, routed_counts)

        # Expand topk to include shared expert
        expanded_topk_ids, expanded_topk_weights = expand_topk_for_shared_expert(
            topk_ids,
            topk_weights,
            shared_destination,
            self.shared_expert_id,
            self.shared_weight,
            self.rank,
        )

        if DEEPEP_WATERFILL_DEBUG:
            print(
                f"[DeepEP Waterfill] rank={self.rank} "
                f"num_tokens={topk_ids.shape[0]} "
                f"routed_counts={routed_counts.tolist()} "
                f"shared_weight={self.shared_weight:.4f}"
            )

        return expanded_topk_ids, expanded_topk_weights, shared_destination


def split_shared_and_routed_tokens(
    hidden_states: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_expert_id: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Split received tokens into shared expert tokens and routed expert tokens.

    After DeepEP dispatch, each rank receives tokens for its local experts.
    We need to separate:
    - Tokens for shared expert (expert_id == shared_expert_id)
    - Tokens for routed experts (expert_id < shared_expert_id)

    Args:
        hidden_states: [total_recv_tokens, hidden_size] received hidden states
        topk_ids: [total_recv_tokens, topk+1] received expert IDs
        topk_weights: [total_recv_tokens, topk+1] received expert weights
        shared_expert_id: Expert ID for shared expert

    Returns:
        shared_hidden: Hidden states for shared expert
        shared_weights: Weights for shared expert
        routed_hidden: Hidden states for routed experts
        routed_topk_ids: Expert IDs for routed experts
        routed_topk_weights: Weights for routed experts
        shared_indices: Original indices of shared expert tokens
    """
    # Find tokens that have shared expert
    # In expanded topk, the last column is shared expert
    shared_mask = topk_ids[:, -1] == shared_expert_id
    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    # Extract shared expert data
    shared_hidden = hidden_states[shared_indices]
    shared_weights = topk_weights[shared_indices, -1]

    # For routed experts, use original topk (without last column)
    routed_topk_ids = topk_ids[:, :-1]
    routed_topk_weights = topk_weights[:, :-1]

    return (
        shared_hidden,
        shared_weights,
        hidden_states,  # All tokens go through routed path
        routed_topk_ids,
        routed_topk_weights,
        shared_indices,
    )


def merge_shared_and_routed_outputs(
    shared_output: Tensor,
    routed_output: Tensor,
    shared_indices: Tensor,
    shared_weights: Tensor,
) -> Tensor:
    """
    Merge shared expert output with routed expert output.

    Args:
        shared_output: [num_shared, hidden_size] shared expert computation result
        routed_output: [total_tokens, hidden_size] routed expert computation result
        shared_indices: [num_shared] indices of shared expert tokens
        shared_weights: [num_shared] weights for shared expert

    Returns:
        merged_output: [total_tokens, hidden_size] merged output
    """
    # Add shared output to corresponding positions
    # shared_weights is already 1.0 / routed_scaling_factor
    if shared_output.shape[0] > 0:
        routed_output.index_add_(
            0,
            shared_indices,
            shared_output * shared_weights.unsqueeze(-1),
        )

    return routed_output

