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
   - Or source rank (local computation)

2. Virtual expert ID = target_rank * experts_per_rank
   - This ensures DeepEP routes to the correct rank
   - No need to increase num_experts

3. On receiver side:
   - Identify tokens whose 9th expert is for this rank
   - Compute shared expert separately from routed experts
   - Merge outputs before combine

4. Shared expert weight = 1.0 / routed_scaling_factor
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

DEEPEP_WATERFILL_DEBUG = os.environ.get("SGLANG_DEEPEP_WATERFILL_DEBUG", "0") == "1"


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
        torch.full_like(topk_ids, world_size),
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
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Build candidate mask: [num_tokens, world_size]
    candidate_mask = torch.zeros(num_tokens, world_size, dtype=torch.bool, device=device)
    candidate_mask[:, source_rank] = True  # Source rank is always a candidate

    # Add routed ranks as candidates
    valid_mask = topk_ids >= 0
    rank_ids = torch.where(
        valid_mask,
        torch.clamp(topk_ids // experts_per_rank, 0, world_size - 1),
        torch.zeros_like(topk_ids),
    )

    for k in range(topk):
        token_indices = torch.arange(num_tokens, device=device)
        valid = valid_mask[:, k]
        ranks = rank_ids[:, k]
        candidate_mask[token_indices[valid], ranks[valid]] = True

    # Select rank with minimum count among candidates
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
    shared_weight: float,
) -> Tuple[Tensor, Tensor]:
    """
    Expand topk_ids/weights from [N, 8] to [N, 9] with shared expert info.

    The 9th column contains a virtual expert ID that routes to the target rank:
    virtual_expert_id = target_rank * experts_per_rank

    This ensures DeepEP dispatches the token to the correct rank without
    needing to increase num_experts in the MoE runner.
    """
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device
    experts_per_rank = num_experts // world_size

    # Virtual expert ID = target_rank * experts_per_rank
    # This ID will be in the range [0, num_experts) and routes to target_rank
    virtual_expert_ids = (shared_destination * experts_per_rank).unsqueeze(1)

    expanded_topk_ids = torch.cat(
        [topk_ids, virtual_expert_ids.to(topk_ids.dtype)], dim=1
    )

    shared_weights_col = torch.full(
        (num_tokens, 1), shared_weight, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights = torch.cat([topk_weights, shared_weights_col], dim=1)

    return expanded_topk_ids, expanded_topk_weights


# ============== Main API ==============


class DeepEPWaterfillBalancer:
    """
    Waterfill load balancer for DeepEP-based shared expert dispatch.

    This class implements the waterfill algorithm that assigns each token's
    shared expert computation to the least loaded rank among:
    1. Ranks it already routes to (no extra communication)
    2. Source rank (local computation)

    The shared expert is encoded as a virtual 9th expert in topk_ids.
    """

    MIN_BATCH_FOR_BALANCE = 64

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
        """Assign shared expert destination for each token using waterfill."""
        return assign_shared_destination_pytorch(
            topk_ids, routed_counts, self.num_experts, self.world_size, self.rank
        )

    def prepare_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare expanded topk for dispatch with shared expert as 9th expert.

        Returns:
            expanded_topk_ids: [N, 9] with virtual expert ID in 9th column
            expanded_topk_weights: [N, 9] with shared_weight in 9th column
        """
        shared_destination = self.assign_shared_destination(topk_ids, routed_counts)

        expanded_topk_ids, expanded_topk_weights = expand_topk_with_shared_expert(
            topk_ids,
            topk_weights,
            shared_destination,
            self.num_experts,
            self.world_size,
            self.shared_weight,
        )

        if DEEPEP_WATERFILL_DEBUG:
            # Count how many tokens go to each rank for shared expert
            dest_counts = torch.bincount(
                shared_destination, minlength=self.world_size
            ).tolist()
            print(
                f"[DeepEP Waterfill] rank={self.rank} "
                f"tokens={topk_ids.shape[0]} "
                f"routed_counts={routed_counts.tolist()} "
                f"shared_dest_counts={dest_counts}"
            )

        return expanded_topk_ids, expanded_topk_weights


def identify_shared_expert_tokens(
    recv_topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    current_rank: int,
) -> Tensor:
    """
    Identify which received tokens need shared expert computation on this rank.

    A token needs shared expert here if its 9th column (virtual expert ID)
    maps to current_rank.

    Returns:
        shared_indices: indices of tokens needing shared expert computation
    """
    experts_per_rank = num_experts // world_size

    # 9th column contains virtual expert ID = target_rank * experts_per_rank
    virtual_expert_ids = recv_topk_ids[:, -1]
    target_ranks = virtual_expert_ids // experts_per_rank

    shared_mask = target_ranks == current_rank
    shared_indices = shared_mask.nonzero(as_tuple=True)[0]

    return shared_indices
