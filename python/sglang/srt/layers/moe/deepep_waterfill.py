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

Shared expert is treated as the 9th routed expert (topk=9) and dispatched
through DeepEP. Each token's shared expert is assigned to the least-loaded
rank among its routed destinations. Expert IDs are remapped to a per-rank
layout of (old_experts_per_rank + 1) slots. See DeepEPWaterfillBalancer for details.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

# Marker value for invalid/padded tokens that should not dispatch shared expert.
# DeepEP treats expert_id < 0 as invalid, so these tokens are safely ignored.
LOCAL_SHARED_MARKER = -1

# Local preference factor used by waterfill assignment.
# Set to 1.0 to disable the bias and use pure argmin over routed_counts.
LOCAL_PREFERENCE_FACTOR = 1.1


import triton
import triton.language as tl

# ============== Triton Kernels (GPU-optimized) ==============


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
    routed_counts_ptr,  # [world_size]  (effective load per rank)
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
    precomputed_target_total,  # Pre-computed target total load per rank
    ALLOW_ALL_RANKS: tl.constexpr,
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

    r_idx = tl.arange(0, world_size)
    routed_vec = tl.load(
        routed_counts_ptr + r_idx, mask=r_idx < world_size, other=0
    ).to(tl.int64)
    total_effective_k = tl.sum(routed_vec)
    total_tokens_global_k = total_effective_k // topk
    derived_target_total = (
        total_effective_k + total_tokens_global_k + world_size - 1
    ) // world_size
    # Use precomputed_target_total when provided (> 0); otherwise fall back
    # to the derived value.  The dynamic path passes a pre-computed target
    # that accounts for DP-attention load, while the static path passes 0.
    target_total = tl.where(
        precomputed_target_total > 0,
        precomputed_target_total,
        derived_target_total,
    )

    # ===== Step 1: Select destination rank for shared expert =====
    # Prefer balanced total load (routed + shared) by sampling destination among
    # candidate ranks (routed ranks + source rank) with probability proportional
    # to (target_total - routed_counts[r]). If all candidate weights are zero, fall back to the
    # legacy argmin(routed_counts) logic.
    source_count = tl.load(routed_counts_ptr + source_rank)
    best_count = tl.where(mask, source_count, 2**30)
    best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    src_rank_i32 = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)

    if ALLOW_ALL_RANKS:
        candidate_mask = tl.full([BLOCK_SIZE], (1 << world_size) - 1, dtype=tl.int32)
        for r in range(world_size):
            target_count = tl.load(routed_counts_ptr + r).to(tl.int64)
            better = (
                target_count * local_pref_numer < best_count * local_pref_denom
            ) & mask
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(
                better, tl.full([BLOCK_SIZE], r, dtype=tl.int64), best_rank
            )
    else:
        candidate_mask = (tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << src_rank_i32).to(
            tl.int32
        )

    for k in range(topk):
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        valid = expert_id >= 0
        has_valid = has_valid | valid

        if not ALLOW_ALL_RANKS:
            # Use OLD experts_per_rank for rank calculation from original expert IDs
            target_rank = expert_id // old_experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            target_rank_i32 = target_rank.to(tl.int32)
            shift_amt = tl.where(valid, target_rank_i32, 0)
            bit = tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << shift_amt
            candidate_mask = tl.where(
                valid & mask, candidate_mask | bit, candidate_mask
            )

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

    # Total weight per token across candidate ranks.
    total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for r in range(world_size):
        present = ((candidate_mask >> r) & 1) == 1
        routed_r = tl.load(routed_counts_ptr + r).to(tl.int64)
        w = tl.where(target_total > routed_r, target_total - routed_r, 0).to(tl.int32)
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)
        # Apply local preference (scale down remote weights).
        w_vec = tl.where(
            src_rank_i32 == r,
            w_vec,
            (w_vec * local_pref_denom) // local_pref_numer,
        )
        total_w += tl.where(present, w_vec, 0)

    # Deterministic per-token draw in [0, total_w).
    token_seed = token_idx.to(tl.uint32) ^ (
        src_rank_i32.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full([BLOCK_SIZE], 1664525, dtype=tl.uint32) + tl.full(
        [BLOCK_SIZE], 1013904223, dtype=tl.uint32
    )
    u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)

    chosen = src_rank_i32
    cum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for r in range(world_size):
        present = ((candidate_mask >> r) & 1) == 1
        routed_r = tl.load(routed_counts_ptr + r).to(tl.int64)
        w = tl.where(target_total > routed_r, target_total - routed_r, 0).to(tl.int32)
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)
        w_vec = tl.where(
            src_rank_i32 == r,
            w_vec,
            (w_vec * local_pref_denom) // local_pref_numer,
        )
        w_vec = tl.where(present, w_vec, 0)
        pick = (total_w > 0) & present & (u >= cum) & (u < (cum + w_vec))
        chosen = tl.where(pick, r, chosen)
        cum += w_vec

    best_rank = tl.where(total_w > 0, chosen.to(tl.int64), best_rank)

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
    # Padded / invalid tokens (all routed experts are -1) should not dispatch shared expert.
    shared_expert_id = tl.where(
        has_valid,
        shared_expert_id,
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),
    )

    dest_rank = tl.where(is_local, source_rank, best_rank).to(tl.int32)

    # ===== Step 3: Copy and remap topk_ids, copy topk_weights =====
    # Remap routed expert IDs: old_id -> old_id + (old_id // old_experts_per_rank)
    for k in range(topk):
        old_id = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1).to(
            tl.int64
        )
        # Only remap valid IDs (>= 0)
        valid_id = old_id >= 0
        # new_id = old_id + (old_id // old_experts_per_rank)
        new_id = tl.where(valid_id, old_id + (old_id // old_experts_per_rank), old_id)
        tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

    for k in range(topk):
        val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        val = tl.where(expert_id >= 0, val, 0.0)
        tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

    # ===== Step 4: Write 9th column (shared expert) =====
    tl.store(
        expanded_ids_ptr + token_idx * (topk + 1) + topk,
        shared_expert_id,
        mask=mask,
    )
    tl.store(
        expanded_weights_ptr + token_idx * (topk + 1) + topk,
        tl.where(has_valid, shared_weight, 0.0),
        mask=mask,
    )

    # ===== Step 5: Write local mask =====
    tl.store(local_mask_ptr + token_idx, is_local, mask=mask)

    # ===== Step 6: Block-level histogram with minimal atomics =====
    # Count destinations per rank within this block using sum reduction
    for r in range(world_size):
        rank_count = tl.sum(tl.where(mask & has_valid & (dest_rank == r), 1, 0))
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
    world_size,
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
    dest_rank = tl.minimum(tl.maximum(dest_rank, 0), world_size - 1)

    dest_count = tl.load(dest_counts_ptr + dest_rank, mask=mask, other=0)
    is_sparse_remote = (dest_count < min_tokens_per_rank) & ~is_local

    # Redirect sparse remote destinations to local shared expert ID.
    local_shared_id = source_rank * new_experts_per_rank + old_experts_per_rank
    local_shared_id_vec = tl.full([BLOCK_SIZE], local_shared_id, dtype=tl.int64)
    new_shared_id = tl.where(is_sparse_remote, local_shared_id_vec, shared_expert_id)
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
    allow_all_ranks: bool = False,
    target_total: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Fully fused waterfill using Triton with integrated histogram and expert ID remapping.

    Expert ID remapping: old_id -> old_id + (old_id // old_experts_per_rank)
    This maps original expert IDs to new layout where each rank has one extra expert slot
    for the shared expert.

    Single kernel does: waterfill + expand + histogram counting + ID remapping.

    Returns:
        expanded_topk_ids: [N, 9] with remapped expert IDs
        expanded_topk_weights: [N, 9]
        local_shared_mask: [N] boolean
        dest_counts: [world_size] histogram of shared expert destinations (local to this rank)
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
            torch.zeros(world_size, dtype=torch.int32, device=device),
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

    # Always use fused kernel with histogram; sparse redirect is applied outside
    # (after global reduction of dest_counts) in DeepEPWaterfillBalancer.prepare_dispatch.
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
        target_total,
        allow_all_ranks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expanded_topk_ids, expanded_topk_weights, local_shared_mask, dest_counts


def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_destination: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Expand topk from [N, 8] to [N, 9] with shared expert as real expert.

    Remaps routed IDs: old_id -> old_id + (old_id // old_epr).
    Shared expert for rank i -> i * new_epr + old_epr.

    Returns (expanded_topk_ids, expanded_topk_weights, local_shared_mask).
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Old and new experts per rank
    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1  # +1 for shared expert

    # Identify local vs remote shared expert
    local_shared_mask = shared_destination == source_rank
    # Tokens with no valid routed experts (e.g. padded region) should NOT dispatch shared expert.
    has_any_valid = (topk_ids >= 0).any(dim=1)

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
    expanded_topk_ids[:, topk] = torch.where(
        has_any_valid,
        shared_expert_ids.to(topk_ids.dtype),
        torch.full(
            (num_tokens,), LOCAL_SHARED_MARKER, dtype=topk_ids.dtype, device=device
        ),
    )

    # OPTIMIZED: Pre-allocate weights tensor
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = topk_weights
    expanded_topk_weights[:, topk] = torch.where(
        has_any_valid,
        torch.full(
            (num_tokens,), float(shared_weight), dtype=topk_weights.dtype, device=device
        ),
        torch.zeros((num_tokens,), dtype=topk_weights.dtype, device=device),
    )
    # For invalid tokens, force all weights to 0 for safety.
    if (~has_any_valid).any():
        expanded_topk_weights[~has_any_valid, :topk] = 0.0

    # Local shared mask is only meaningful for tokens that actually dispatch shared expert.
    local_shared_mask = local_shared_mask & has_any_valid

    return expanded_topk_ids, expanded_topk_weights, local_shared_mask


# ============== Main API ==============


def compute_static_rank_load(
    logical_count: Tensor,
    physical_to_logical_map: Tensor,
    world_size: int,
) -> Tensor:
    """Compute per-layer static rank load from EPLB historical statistics.

    Given historical ``logical_count`` (average expert utilisation from the EPLB
    recorder) and the current ``physical_to_logical_map``, this function produces
    a ``[num_layers, world_size]`` float tensor where entry ``[l, r]`` estimates
    the *relative* workload that EP rank ``r`` will carry in MoE layer ``l``.

    The returned tensor is suitable for :pymethod:`DeepEPWaterfillBalancer.set_static_weights`.
    It allows the forward path to **skip the runtime all_reduce** and use these
    pre-computed weights for waterfill sampling instead.

    **Expert replication handling** (critical for correctness):
    Multiple physical experts may map to the same logical expert (EPLB
    replication).  We divide each logical expert's historical count by its
    replica count so that load is split evenly across all physical copies.

    Args:
        logical_count: ``[num_layers, num_logical_experts]`` float/int tensor.
            Average token count per logical expert across recent history.
            If the raw recording has shape ``[num_samples, num_layers, num_logical_experts]``,
            caller should average over samples first.
        physical_to_logical_map: ``[num_layers, num_physical_experts]`` int tensor.
            Maps each physical expert slot to its logical expert id.
        world_size: Number of EP ranks.

    Returns:
        ``[num_layers, world_size]`` float64 tensor with per-rank workload estimates.
    """
    num_layers, num_physical_experts = physical_to_logical_map.shape
    num_logical_experts = logical_count.shape[-1]
    experts_per_rank = num_physical_experts // world_size

    device = physical_to_logical_map.device
    logical_count = logical_count.to(device=device, dtype=torch.float64)
    physical_to_logical_map = physical_to_logical_map.to(device=device)

    # Step 1: Compute replica count per logical expert per layer.
    # replica_counts[l, e] = number of physical experts mapped to logical expert e in layer l.
    ones = torch.ones(
        num_layers, num_physical_experts, dtype=torch.float64, device=device
    )
    replica_counts = torch.zeros(
        num_layers, num_logical_experts, dtype=torch.float64, device=device
    )
    replica_counts.scatter_add_(1, physical_to_logical_map.long(), ones)
    # Avoid division by zero for unused logical experts.
    replica_counts = replica_counts.clamp(min=1.0)

    # Step 2: Per-physical-expert load = logical_count[logical_id] / replica_count[logical_id].
    # Gather logical counts for each physical expert position.
    mapped_logical_ids = (
        physical_to_logical_map.long()
    )  # [num_layers, num_physical_experts]
    physical_load = torch.gather(
        logical_count, 1, mapped_logical_ids
    )  # [num_layers, num_phys]
    physical_replica = torch.gather(
        replica_counts, 1, mapped_logical_ids
    )  # [num_layers, num_phys]
    physical_load = (
        physical_load / physical_replica
    )  # [num_layers, num_physical_experts]

    # Step 3: Aggregate per rank (sum across experts_per_rank experts per rank).
    # Reshape to [num_layers, world_size, experts_per_rank] and sum the last dim.
    per_rank_load = physical_load.view(num_layers, world_size, experts_per_rank).sum(
        dim=2
    )

    return per_rank_load  # [num_layers, world_size]


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

    # Minimum global shared tokens for a rank to accept *remote* shared-expert dispatch.
    # If after aggregating destinations across all ranks a destination rank would get
    # < this many shared tokens, we redirect those remote shared tokens back to their
    # source ranks (i.e., that rank does not receive remote shared expert work).
    #
    # Note: shared expert compute uses 128-token blocks; <64 tokens would waste >50% padding.
    MIN_TOKENS_PER_RANK = 0

    def __init__(
        self,
        num_routed_experts: int,
        world_size: int,
        rank: int,
        routed_scaling_factor: float = 1.0,
        static_rank_load: Optional[Tensor] = None,
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

        # Static per-rank load derived from EPLB historical statistics.
        # Shape: [world_size], dtype float64/int64. When set, forward_deepep_waterfill
        # can skip the runtime all_reduce and use these weights directly.
        self.static_rank_load: Optional[Tensor] = static_rank_load

        # Pre-allocated buffers to avoid per-layer tensor allocations in the
        # hot path.  Lazily initialised on first use (device may not be known
        # at __init__ time).
        self._counts_buf: Optional[Tensor] = None  # [world_size], int64

    # -------- Static weight helpers --------

    def has_static_weights(self) -> bool:
        """Return True if static EPLB-derived weights are available."""
        return self.static_rank_load is not None

    def set_static_weights(self, static_rank_load: Tensor) -> None:
        """Replace static per-rank load weights (e.g. after EPLB rebalance)."""
        assert static_rank_load.shape == (
            self.world_size,
        ), f"Expected shape ({self.world_size},), got {static_rank_load.shape}"
        self.static_rank_load = static_rank_load.to(dtype=torch.float64)
        w = self.static_rank_load
        w_sum = w.sum().clamp(min=1.0)
        self._static_rank_load_normalized = w / w_sum

    def estimate_global_counts(
        self,
        local_routed_counts: Tensor,
        topk: int,
    ) -> Tuple[Tensor, Tensor]:
        """Estimate global routed counts and local_tokens_per_rank without all_reduce.

        Uses ``self.static_rank_load`` to scale the locally-observed total into
        per-rank estimates, removing the need for the runtime ``all_reduce``.
        All operations stay on GPU — no ``.item()`` or GPU→CPU sync.

        Args:
            local_routed_counts: ``[world_size]`` int64 – routed counts from this rank.
            topk: Number of routed experts per token (e.g. 8).

        Returns:
            estimated_global_routed: ``[world_size]`` int64.
            estimated_local_tokens: ``[world_size]`` int64 (uniform assumption).
        """
        assert self.static_rank_load is not None
        device = local_routed_counts.device

        local_total_routed = local_routed_counts.sum()
        estimated_global_total = local_total_routed * self.world_size

        w = self._static_rank_load_normalized
        estimated_global_routed = (w * estimated_global_total.double()).to(torch.int64)

        local_num_tokens = local_total_routed // max(topk, 1)
        estimated_local_tokens = local_num_tokens.expand(self.world_size).to(
            torch.int64
        )

        return estimated_global_routed, estimated_local_tokens

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank from local topk_ids.

        Uses Triton kernel on GPU for better performance.

        Note: topk_ids contains ORIGINAL expert IDs (0-255), so we use
        num_routed_experts to calculate experts_per_rank for rank assignment.
        """
        # Reuse pre-allocated buffer to avoid per-layer torch.zeros allocation.
        if self._counts_buf is None:
            self._counts_buf = torch.zeros(
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        buf = self._counts_buf
        buf.zero_()
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = self.num_routed_experts // self.world_size
        if num_tokens == 0:
            return buf
        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _count_routed_per_rank_kernel[grid](
            topk_ids,
            buf,
            num_tokens,
            topk,
            experts_per_rank,
            self.world_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return buf

    def prepare_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
        local_tokens_per_rank: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare expanded topk for dispatch with shared expert as 9th expert.

        Uses fused Triton kernel on GPU for maximum performance.

        Args:
            topk_ids: [N, topk] routed expert IDs
            topk_weights: [N, topk] routed expert weights
            routed_counts: [world_size] global routed token count per rank
            local_tokens_per_rank: [world_size] number of tokens each EP rank
                processes from DP attention.  When provided, waterfill uses
                ``routed_counts + local_tokens_per_rank`` as the effective load
                per rank so that shared expert tokens are steered away from
                ranks that already carry a heavy DP-attention load.

        Optimizations:
        1. Fused kernel: waterfill + expand + per-rank histogram in single GPU pass
        2. If batch size < MIN_BATCH_FOR_BALANCE, all shared experts compute locally
        3. Global sparse redirect: if a destination rank would get < MIN_TOKENS_PER_RANK
           shared tokens (after aggregating across all ranks), redirect those remote shared
           tokens back to their source ranks to avoid tiny shards / padding waste.

        Returns:
            expanded_topk_ids: [N, 9] with remapped expert IDs (shared expert as 9th)
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
            # Fast path: all local, no waterfill needed.
            # Still need to remap expert IDs to new layout and handle padded/invalid tokens.
            shared_destination = torch.full(
                (num_tokens,), self.rank, dtype=torch.int64, device=device
            )
            return expand_topk_with_shared_expert(
                topk_ids,
                topk_weights,
                shared_destination,
                self.num_routed_experts,
                self.world_size,
                self.rank,
                self.shared_weight,
            )

        # Effective per-rank load for waterfill weighting.
        # When local_tokens_per_rank is provided (DP-attention aware mode),
        # we add it to routed_counts so that shared expert tokens are steered
        # away from ranks that already carry heavy DP-attention load.
        routed_counts_i64 = routed_counts.to(torch.int64)
        if local_tokens_per_rank is not None:
            effective_load = routed_counts_i64 + local_tokens_per_rank.to(torch.int64)
        else:
            effective_load = routed_counts_i64

        # Compute target_total and allow_all_ranks WITHOUT GPU→CPU sync.
        # When using static weights, always allow dispatch to any rank (EPLB
        # already balances routed load, so the mild-imbalance condition is
        # almost always satisfied).  For the dynamic path, keep the original
        # logic but compute target_total entirely on GPU (single .item() at
        # the very end, reducing 3 syncs to 1).
        if self.has_static_weights():
            # Static path: zero GPU→CPU syncs.
            # Pass target_total=0 so the kernel derives it from routed_counts.
            # allow_all_ranks=True since EPLB keeps routed load balanced.
            allow_all_ranks = True
            target_total = 0
        else:
            # Dynamic path: keep original logic (3 → 1 sync).
            total_routed_t = routed_counts_i64.sum()
            total_tokens_global_t = total_routed_t // topk
            total_effective_t = effective_load.sum()
            max_effective_t = effective_load.max()
            target_total = int(
                (
                    (total_effective_t + total_tokens_global_t + self.world_size - 1)
                    // self.world_size
                ).item()
            )
            allow_all_ranks = bool((max_effective_t <= target_total).item())

        expanded_topk_ids, expanded_topk_weights, local_shared_mask, dest_counts = (
            waterfill_prepare_dispatch_fused(
                topk_ids,
                topk_weights,
                effective_load,
                self.num_routed_experts,
                self.world_size,
                self.rank,
                self.shared_weight,
                allow_all_ranks=allow_all_ranks,
                target_total=target_total,
            )
        )

        if self.MIN_TOKENS_PER_RANK > 0:
            # Globalize dest_counts across EP ranks, then redirect sparse remote destinations.
            try:
                import torch.distributed as dist

                if dist.is_initialized() and self.world_size > 1:
                    from sglang.srt.distributed.parallel_state import get_moe_ep_group

                    dist.all_reduce(
                        dest_counts,
                        op=dist.ReduceOp.SUM,
                        group=get_moe_ep_group().device_group,
                    )
            except Exception:
                # If distributed is not available/initialized, fall back to local counts.
                pass

            BLOCK_SIZE = 256
            grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _sparse_redirect_kernel[grid](
                expanded_topk_ids,
                local_shared_mask,
                dest_counts,
                num_tokens,
                topk + 1,
                self.old_experts_per_rank,
                self.new_experts_per_rank,
                self.world_size,
                self.rank,
                self.MIN_TOKENS_PER_RANK,
                LOCAL_SHARED_MARKER,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return expanded_topk_ids, expanded_topk_weights, local_shared_mask
