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
using DeepEP communication. The key idea is:

1. Each token's shared expert can ONLY be sent to:
   - One of the ranks it already routes to (no extra communication)
   - Or stay at source rank for local computation

2. Waterfill algorithm selects the lowest-loaded rank from these candidates

3. Implementation strategy:
   - For tokens staying local: compute shared expert locally, don't include in dispatch
   - For tokens going remote: encode shared expert as a "virtual expert" on target rank
   - Virtual expert ID = num_routed_experts + target_rank (e.g., 256..263 for 8 ranks)

4. Shared expert weight = 1.0 / routed_scaling_factor (for correct combine)
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

# Marker for tokens that should compute shared expert locally (not dispatch)
LOCAL_SHARED_EXPERT_MARKER = -1


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
    num_routed_experts: int,
    shared_weight: float,
    source_rank: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Expand topk_ids and topk_weights to include shared expert.

    For each token:
    - If destination == source_rank: mark as LOCAL_SHARED_EXPERT_MARKER (-1)
      (will be computed locally, not dispatched)
    - If destination != source_rank: use virtual expert ID = num_routed_experts + dest_rank
      (will be dispatched to dest_rank which will compute shared expert)

    Args:
        topk_ids: [num_tokens, topk] original expert IDs
        topk_weights: [num_tokens, topk] original expert weights
        shared_destination: [num_tokens] destination ranks for shared expert
        num_routed_experts: Number of routed experts (e.g., 256)
        shared_weight: Weight for shared expert (1.0 / routed_scaling_factor)
        source_rank: Current rank ID

    Returns:
        expanded_topk_ids: [num_tokens, topk+1]
        expanded_topk_weights: [num_tokens, topk+1]
        local_shared_mask: [num_tokens] boolean mask for tokens with local shared expert
    """
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device

    # Determine which tokens compute shared expert locally vs remotely
    local_shared_mask = shared_destination == source_rank

    # Create expanded tensors
    expanded_topk_ids = torch.cat(
        [topk_ids, torch.full((num_tokens, 1), LOCAL_SHARED_EXPERT_MARKER, dtype=topk_ids.dtype, device=device)],
        dim=1,
    )
    expanded_topk_weights = torch.cat(
        [topk_weights, torch.full((num_tokens, 1), shared_weight, dtype=topk_weights.dtype, device=device)],
        dim=1,
    )

    # For tokens that send shared expert to remote rank:
    # Set expert ID = num_routed_experts + destination_rank
    # This creates "virtual experts" 256, 257, ..., 263 (for 8 ranks)
    # Each virtual expert will be handled by its corresponding rank
    remote_shared_mask = ~local_shared_mask
    if remote_shared_mask.any():
        virtual_expert_ids = num_routed_experts + shared_destination
        expanded_topk_ids[remote_shared_mask, -1] = virtual_expert_ids[remote_shared_mask]

    # Tokens with local shared expert keep -1 (won't be dispatched for the 9th slot)

    return expanded_topk_ids, expanded_topk_weights, local_shared_mask


# ============== Main API ==============


class DeepEPWaterfillBalancer:
    """
    Waterfill load balancer for DeepEP-based shared expert dispatch.

    The balancer assigns each token's shared expert computation to either:
    1. A rank it already routes to (no extra communication)
    2. The source rank (local computation)

    Virtual expert IDs for shared expert: num_routed_experts + rank_id
    E.g., for 256 routed experts and 8 ranks: virtual IDs are 256, 257, ..., 263

    Usage:
        balancer = DeepEPWaterfillBalancer(num_experts=256, world_size=8, rank=0)
        expanded_topk, local_mask = balancer.prepare_dispatch(topk_ids, topk_weights, routed_counts)
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

        # Virtual expert IDs for shared expert on each rank
        # rank 0 -> num_experts + 0, rank 1 -> num_experts + 1, etc.
        self.shared_expert_base_id = num_experts

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
            local_shared_mask: [num_tokens] boolean mask for local shared expert tokens
        """
        # Assign shared expert destination using waterfill
        shared_destination = self.assign_shared_destination(topk_ids, routed_counts)

        # Expand topk to include shared expert (with correct virtual expert IDs)
        expanded_topk_ids, expanded_topk_weights, local_shared_mask = expand_topk_for_shared_expert(
            topk_ids,
            topk_weights,
            shared_destination,
            self.num_experts,  # num_routed_experts
            self.shared_weight,
            self.rank,
        )

        if DEEPEP_WATERFILL_DEBUG:
            num_local = local_shared_mask.sum().item()
            num_remote = (~local_shared_mask).sum().item()
            print(
                f"[DeepEP Waterfill] rank={self.rank} "
                f"num_tokens={topk_ids.shape[0]} "
                f"local_shared={num_local} remote_shared={num_remote} "
                f"routed_counts={routed_counts.tolist()} "
                f"shared_weight={self.shared_weight:.4f}"
            )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask


def identify_received_shared_tokens(
    recv_topk_ids: Tensor,
    num_routed_experts: int,
    current_rank: int,
) -> Tuple[Tensor, Tensor]:
    """
    Identify received tokens that need shared expert computation on this rank.

    After DeepEP dispatch, this rank receives tokens from all source ranks.
    We need to identify tokens that were assigned to compute shared expert here.

    Virtual expert ID for this rank = num_routed_experts + current_rank

    Args:
        recv_topk_ids: [total_recv_tokens, topk+1] received expert IDs
        num_routed_experts: Number of routed experts (e.g., 256)
        current_rank: Current rank ID

    Returns:
        shared_mask: [total_recv_tokens] boolean mask for tokens needing shared expert
        shared_indices: [num_shared] indices of tokens needing shared expert
    """
    # Virtual expert ID for shared expert on this rank
    virtual_shared_id = num_routed_experts + current_rank

    # Check if the last column (shared expert slot) matches our virtual ID
    shared_mask = recv_topk_ids[:, -1] == virtual_shared_id
    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    return shared_mask, shared_indices


def merge_shared_output_inplace(
    routed_output: Tensor,
    shared_output: Tensor,
    shared_indices: Tensor,
    shared_weights: Tensor,
) -> Tensor:
    """
    Merge shared expert output into routed expert output in-place.

    Args:
        routed_output: [total_tokens, hidden_size] routed expert computation result (modified in-place)
        shared_output: [num_shared, hidden_size] shared expert computation result
        shared_indices: [num_shared] indices where to add shared output
        shared_weights: [num_shared] weights for shared expert (already = 1.0 / routed_scaling_factor)

    Returns:
        routed_output: [total_tokens, hidden_size] merged output
    """
    if shared_output is not None and shared_output.shape[0] > 0:
        # shared_weights is 1.0 / routed_scaling_factor
        # After combine's routed_scaling_factor multiplication:
        # shared contribution = shared_output * shared_weights * routed_scaling_factor
        #                     = shared_output * (1/rsf) * rsf = shared_output (correct!)
        routed_output.index_add_(
            0,
            shared_indices,
            shared_output * shared_weights.unsqueeze(-1),
        )

    return routed_output


def compute_local_shared_expert(
    hidden_states: Tensor,
    local_shared_mask: Tensor,
    shared_expert_fn,
    shared_weight: float,
) -> Tuple[Optional[Tensor], Tensor]:
    """
    Compute shared expert for tokens that stay local.

    Args:
        hidden_states: [num_tokens, hidden_size] input hidden states
        local_shared_mask: [num_tokens] boolean mask for tokens with local shared expert
        shared_expert_fn: Function to compute shared expert (e.g., self.shared_experts)
        shared_weight: Weight for shared expert (1.0 / routed_scaling_factor)

    Returns:
        local_shared_output: [num_tokens, hidden_size] or None if no local tokens
                             Output is already weighted and shaped for direct addition
        local_shared_indices: [num_local] indices of local shared expert tokens
    """
    local_indices = local_shared_mask.nonzero(as_tuple=True)[0]

    if local_indices.shape[0] == 0:
        return None, local_indices

    # Compute shared expert for local tokens
    local_hidden = hidden_states[local_indices]
    local_output = shared_expert_fn(local_hidden)

    # Weight the output (will be combined later without additional weighting)
    local_output = local_output * shared_weight

    return local_output, local_indices

