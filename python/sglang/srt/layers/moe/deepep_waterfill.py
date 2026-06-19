# Copyright 2023-2026 SGLang Team
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
"""DeepEP Waterfill: shared expert as 9th routed expert, dispatched to least-loaded rank."""

import logging
import os
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import get_fused_shared_expert_replicas_per_rank

LOCAL_SHARED_MARKER = -1  # Invalid expert ID; DeepEP ignores expert_id < 0.
_LOCAL_PREF_NUMER = 11  # local-rank preference = 11/10
_LOCAL_PREF_DENOM = 10

logger = logging.getLogger(__name__)


class WaterfillDispatchPlan(NamedTuple):
    """Inputs needed by the fused DeepEP Waterfill expansion path."""

    # Effective rank load consumed by the fused kernel.
    rank_load: Tensor
    allow_all_ranks: bool
    target_total: int


def _empty_expanded(topk_ids: Tensor, topk_weights: Tensor):
    """Return empty expanded tensors for zero-token batches."""
    topk, d = topk_ids.shape[1], topk_ids.device
    return (
        torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=d),
        torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=d),
    )


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
    """Count routed tokens per rank using block-level histogram."""
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    for r in range(world_size):
        rank_count = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

        for k in range(topk):
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid = expert_id >= 0
            target_rank = expert_id // experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            rank_count += tl.where(
                mask & valid & (target_rank == r),
                tl.full([BLOCK_SIZE], 1, dtype=tl.int64),
                tl.zeros([BLOCK_SIZE], dtype=tl.int64),
            )

        block_total = tl.sum(rank_count)
        if block_total > 0:
            tl.atomic_add(counts_ptr + r, block_total)


@triton.jit
def _count_routed_per_rank2_kernel(
    topk_ids_ptr,  # [num_tokens, topk]
    counts_ptr,  # [2] output (atomic add)
    num_tokens,
    topk: tl.constexpr,
    experts_per_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Specialized TP/EP=2 counter.

    B200 Mega-MoE uses EP2 in the target setup. The generic kernel rereads the
    same TopK IDs once per rank; this path reads each routed ID once and derives
    rank1_count = valid_count - rank0_count.
    """
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    rank0_count = tl.full((), 0, dtype=tl.int64)
    valid_count = tl.full((), 0, dtype=tl.int64)

    for k in range(topk):
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        valid = mask & (expert_id >= 0)
        target_rank = expert_id // experts_per_rank
        one = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
        zero = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        valid_count += tl.sum(tl.where(valid, one, zero))
        rank0_count += tl.sum(tl.where(valid & (target_rank == 0), one, zero))

    if valid_count > 0:
        tl.atomic_add(counts_ptr, rank0_count)
        tl.atomic_add(counts_ptr + 1, valid_count - rank0_count)


@triton.jit
def _count_routed_per_rank2_single_block_kernel(
    topk_ids_ptr,  # [num_tokens, topk]
    counts_ptr,  # [2] output
    num_tokens,
    topk: tl.constexpr,
    experts_per_rank,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-program EP2 counter for medium prefill batches.

    This avoids the separate counts_buf.zero_() launch and all atomics. It is
    only used for bounded token counts where one program can cover the batch.
    """
    token_idx = tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    rank0_count = tl.full((), 0, dtype=tl.int64)
    valid_count = tl.full((), 0, dtype=tl.int64)

    for k in range(topk):
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        valid = mask & (expert_id >= 0)
        target_rank = expert_id // experts_per_rank
        one = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
        zero = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        valid_count += tl.sum(tl.where(valid, one, zero))
        rank0_count += tl.sum(tl.where(valid & (target_rank == 0), one, zero))

    tl.store(counts_ptr, rank0_count)
    tl.store(counts_ptr + 1, valid_count - rank0_count)


@triton.jit
def _waterfill_expand_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    rank_load_ptr,
    expanded_ids_ptr,
    expanded_weights_ptr,
    num_tokens,
    topk: tl.constexpr,
    old_experts_per_rank,
    new_experts_per_rank,
    shared_replicas_per_rank: tl.constexpr,
    world_size: tl.constexpr,
    source_rank,
    shared_weight,
    local_marker,
    local_pref_numer,
    local_pref_denom,
    remote_cost_tokens,
    precomputed_target_total,
    ALLOW_ALL_RANKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused waterfill + expand. ID remap: old_id -> old_id + old_id // old_epr."""
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    r_idx = tl.arange(0, world_size)
    rank_load_vec = tl.load(rank_load_ptr + r_idx, mask=r_idx < world_size, other=0).to(
        tl.int64
    )
    total_effective_k = tl.sum(rank_load_vec)
    total_tokens_global_k = total_effective_k // topk
    derived_target_total = (
        total_effective_k + total_tokens_global_k + world_size - 1
    ) // world_size
    target_total = tl.where(
        precomputed_target_total > 0,
        precomputed_target_total,
        derived_target_total,
    )

    # Step 1: Copy/remap routed experts and select destination rank for the
    # shared expert. Static Waterfill allows every rank as a shared target, so
    # keep that path lean: routed TopK only needs to be read once.
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    src_rank_i32 = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)
    best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)

    if ALLOW_ALL_RANKS:
        for k in range(topk):
            old_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid_id = old_id >= 0
            has_valid = has_valid | valid_id
            old_rank = old_id // old_experts_per_rank
            new_id = tl.where(
                valid_id,
                old_id + old_rank * shared_replicas_per_rank,
                old_id,
            )
            tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

            val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
            val = tl.where(valid_id, val, 0.0)
            tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

        total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for r in range(world_size):
            rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
            adjusted_load_r = rank_load_r + tl.where(
                src_rank_i32 == r, 0, remote_cost_tokens
            ).to(tl.int64)
            w = tl.where(
                target_total > adjusted_load_r, target_total - adjusted_load_r, 0
            ).to(tl.int32)
            w_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + w
            w_vec = tl.where(
                src_rank_i32 == r,
                w_vec,
                (w_vec * local_pref_denom) // local_pref_numer,
            )
            total_w += w_vec

        token_seed = token_idx.to(tl.uint32) ^ (
            src_rank_i32.to(tl.uint32)
            * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
        )
        token_seed = token_seed * tl.full(
            [BLOCK_SIZE], 1664525, dtype=tl.uint32
        ) + tl.full([BLOCK_SIZE], 1013904223, dtype=tl.uint32)
        u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)

        chosen = src_rank_i32
        cum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for r in range(world_size):
            rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
            adjusted_load_r = rank_load_r + tl.where(
                src_rank_i32 == r, 0, remote_cost_tokens
            ).to(tl.int64)
            w = tl.where(
                target_total > adjusted_load_r, target_total - adjusted_load_r, 0
            ).to(tl.int32)
            w_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + w
            w_vec = tl.where(
                src_rank_i32 == r,
                w_vec,
                (w_vec * local_pref_denom) // local_pref_numer,
            )
            pick = (total_w > 0) & (u >= cum) & (u < (cum + w_vec))
            chosen = tl.where(pick, r, chosen)
            cum += w_vec

        best_rank = chosen.to(tl.int64)
    else:
        source_count = tl.load(rank_load_ptr + source_rank)
        best_count = tl.where(mask, source_count, 2**30)
        candidate_mask = (tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << src_rank_i32).to(
            tl.int32
        )

        for k in range(topk):
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid = expert_id >= 0
            has_valid = has_valid | valid
            target_rank = expert_id // old_experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            target_rank_i32 = target_rank.to(tl.int32)
            shift_amt = tl.where(valid, target_rank_i32, 0)
            bit = tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << shift_amt
            candidate_mask = tl.where(
                valid & mask, candidate_mask | bit, candidate_mask
            )

            target_count = tl.load(
                rank_load_ptr + target_rank, mask=mask & valid, other=2**30
            )
            adjusted_target_count = target_count + tl.where(
                target_rank_i32 == src_rank_i32, 0, remote_cost_tokens
            ).to(tl.int64)
            adjusted_best_count = best_count

            better = (
                (
                    adjusted_target_count * local_pref_numer
                    < adjusted_best_count * local_pref_denom
                )
                & valid
                & mask
            )
            best_count = tl.where(better, adjusted_target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)

        total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for r in range(world_size):
            present = ((candidate_mask >> r) & 1) == 1
            rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
            adjusted_load_r = rank_load_r + tl.where(
                src_rank_i32 == r, 0, remote_cost_tokens
            ).to(tl.int64)
            w = tl.where(
                target_total > adjusted_load_r, target_total - adjusted_load_r, 0
            ).to(tl.int32)
            w_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + w
            w_vec = tl.where(
                src_rank_i32 == r,
                w_vec,
                (w_vec * local_pref_denom) // local_pref_numer,
            )
            total_w += tl.where(present, w_vec, 0)

        token_seed = token_idx.to(tl.uint32) ^ (
            src_rank_i32.to(tl.uint32)
            * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
        )
        token_seed = token_seed * tl.full(
            [BLOCK_SIZE], 1664525, dtype=tl.uint32
        ) + tl.full([BLOCK_SIZE], 1013904223, dtype=tl.uint32)
        u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)

        chosen = src_rank_i32
        cum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for r in range(world_size):
            present = ((candidate_mask >> r) & 1) == 1
            rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
            adjusted_load_r = rank_load_r + tl.where(
                src_rank_i32 == r, 0, remote_cost_tokens
            ).to(tl.int64)
            w = tl.where(
                target_total > adjusted_load_r, target_total - adjusted_load_r, 0
            ).to(tl.int32)
            w_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + w
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

        for k in range(topk):
            old_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid_id = old_id >= 0
            new_id = tl.where(
                valid_id,
                old_id + (old_id // old_experts_per_rank) * shared_replicas_per_rank,
                old_id,
            )
            tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

            val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
            val = tl.where(valid_id, val, 0.0)
            tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

    # Step 2: Compute shared expert ID and local mask.
    is_local = best_rank == source_rank
    replica_seed = token_idx.to(tl.uint32) ^ (
        best_rank.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x85EBCA6B, dtype=tl.uint32)
    )
    replica_seed = replica_seed * tl.full(
        [BLOCK_SIZE], 1103515245, dtype=tl.uint32
    ) + tl.full([BLOCK_SIZE], 12345, dtype=tl.uint32)
    shared_replica = (replica_seed % shared_replicas_per_rank).to(tl.int64)
    local_shared_id = (
        source_rank * new_experts_per_rank + old_experts_per_rank + shared_replica
    )
    remote_shared_id = (
        best_rank * new_experts_per_rank + old_experts_per_rank + shared_replica
    )
    shared_expert_id = tl.where(is_local, local_shared_id, remote_shared_id).to(
        tl.int64
    )
    shared_expert_id = tl.where(
        has_valid,
        shared_expert_id,
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),
    )

    # Step 3: Write shared expert column.
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


@triton.jit
def _waterfill_expand_rank2_static_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    rank_load_ptr,
    expanded_ids_ptr,
    expanded_weights_ptr,
    num_tokens,
    topk: tl.constexpr,
    old_experts_per_rank,
    new_experts_per_rank,
    shared_replicas_per_rank: tl.constexpr,
    source_rank: tl.constexpr,
    shared_weight,
    local_marker,
    local_pref_numer,
    local_pref_denom,
    remote_cost_tokens,
    ONE_WAY_REMOTE_SHARED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """EP2 static Waterfill expansion.

    Static mode allows both ranks as shared-expert targets. For EP2 the generic
    weighted-random rank loop reduces to a two-way threshold, so avoid the
    world-size loop and candidate-mask bookkeeping in the common B200 path.
    """
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    load0 = tl.load(rank_load_ptr).to(tl.int64)
    load1 = tl.load(rank_load_ptr + 1).to(tl.int64)
    total_effective = load0 + load1
    total_tokens = total_effective // topk
    target_total = (total_effective + total_tokens + 1) // 2
    if source_rank == 0:
        adjusted_load0 = load0
        adjusted_load1 = load1 + remote_cost_tokens
    else:
        adjusted_load0 = load0 + remote_cost_tokens
        adjusted_load1 = load1
    deficit0 = tl.where(
        target_total > adjusted_load0, target_total - adjusted_load0, 0
    ).to(tl.int32)
    deficit1 = tl.where(
        target_total > adjusted_load1, target_total - adjusted_load1, 0
    ).to(tl.int32)

    if ONE_WAY_REMOTE_SHARED:
        if source_rank == 0:
            local_baseline = load0 + num_tokens
            remote_budget = tl.maximum(local_baseline - target_total, 0).to(tl.int32)
            remote_budget = tl.minimum(remote_budget, num_tokens).to(tl.int32)
            source_is_heavy = load0 > load1
            weight0 = num_tokens - tl.where(source_is_heavy, remote_budget, 0)
            weight1 = tl.where(source_is_heavy, remote_budget, 0)
        else:
            local_baseline = load1 + num_tokens
            remote_budget = tl.maximum(local_baseline - target_total, 0).to(tl.int32)
            remote_budget = tl.minimum(remote_budget, num_tokens).to(tl.int32)
            source_is_heavy = load1 > load0
            weight0 = tl.where(source_is_heavy, remote_budget, 0)
            weight1 = num_tokens - tl.where(source_is_heavy, remote_budget, 0)
    else:
        if source_rank == 0:
            weight0 = deficit0
            weight1 = (deficit1 * local_pref_denom) // local_pref_numer
        else:
            weight0 = (deficit0 * local_pref_denom) // local_pref_numer
            weight1 = deficit1
    total_w = weight0 + weight1

    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    for k in range(topk):
        old_id = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1).to(
            tl.int64
        )
        valid_id = old_id >= 0
        has_valid = has_valid | valid_id
        old_rank = old_id // old_experts_per_rank
        new_id = tl.where(
            valid_id,
            old_id + old_rank * shared_replicas_per_rank,
            old_id,
        )
        tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

        val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
        val = tl.where(valid_id, val, 0.0)
        tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

    token_seed = token_idx.to(tl.uint32) ^ (
        tl.full([BLOCK_SIZE], source_rank, dtype=tl.uint32)
        * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full([BLOCK_SIZE], 1664525, dtype=tl.uint32) + tl.full(
        [BLOCK_SIZE], 1013904223, dtype=tl.uint32
    )
    u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)
    chosen_rank = tl.where(u < weight0, 0, 1)
    chosen_rank = tl.where(total_w > 0, chosen_rank, source_rank).to(tl.int64)

    replica_seed = token_idx.to(tl.uint32) ^ (
        chosen_rank.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x85EBCA6B, dtype=tl.uint32)
    )
    replica_seed = replica_seed * tl.full(
        [BLOCK_SIZE], 1103515245, dtype=tl.uint32
    ) + tl.full([BLOCK_SIZE], 12345, dtype=tl.uint32)
    shared_replica = (replica_seed % shared_replicas_per_rank).to(tl.int64)
    shared_id = (
        chosen_rank * new_experts_per_rank + old_experts_per_rank + shared_replica
    )
    shared_id = tl.where(
        has_valid,
        shared_id,
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),
    )
    tl.store(
        expanded_ids_ptr + token_idx * (topk + 1) + topk,
        shared_id,
        mask=mask,
    )
    tl.store(
        expanded_weights_ptr + token_idx * (topk + 1) + topk,
        tl.where(has_valid, shared_weight, 0.0),
        mask=mask,
    )


@triton.jit
def _waterfill_expand_rank2_candidate_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    rank_load_ptr,
    expanded_ids_ptr,
    expanded_weights_ptr,
    num_tokens,
    topk: tl.constexpr,
    old_experts_per_rank,
    new_experts_per_rank,
    shared_replicas_per_rank: tl.constexpr,
    source_rank: tl.constexpr,
    shared_weight,
    local_marker,
    local_pref_numer,
    local_pref_denom,
    remote_cost_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    """EP2 Waterfill expansion restricted to routed ranks plus source rank."""
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    load0 = tl.load(rank_load_ptr).to(tl.int64)
    load1 = tl.load(rank_load_ptr + 1).to(tl.int64)
    total_effective = load0 + load1
    total_tokens = total_effective // topk
    target_total = (total_effective + total_tokens + 1) // 2
    if source_rank == 0:
        adjusted_load0 = load0
        adjusted_load1 = load1 + remote_cost_tokens
    else:
        adjusted_load0 = load0 + remote_cost_tokens
        adjusted_load1 = load1
    deficit0 = tl.where(
        target_total > adjusted_load0, target_total - adjusted_load0, 0
    ).to(tl.int32)
    deficit1 = tl.where(
        target_total > adjusted_load1, target_total - adjusted_load1, 0
    ).to(tl.int32)

    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    has_rank0 = tl.full([BLOCK_SIZE], source_rank == 0, dtype=tl.int1)
    has_rank1 = tl.full([BLOCK_SIZE], source_rank == 1, dtype=tl.int1)

    for k in range(topk):
        old_id = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1).to(
            tl.int64
        )
        valid_id = old_id >= 0
        has_valid = has_valid | valid_id
        old_rank = old_id // old_experts_per_rank
        has_rank0 = has_rank0 | (valid_id & (old_rank == 0))
        has_rank1 = has_rank1 | (valid_id & (old_rank == 1))
        new_id = tl.where(
            valid_id,
            old_id + old_rank * shared_replicas_per_rank,
            old_id,
        )
        tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

        val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
        val = tl.where(valid_id, val, 0.0)
        tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

    if source_rank == 0:
        weight0 = deficit0
        weight1 = (deficit1 * local_pref_denom) // local_pref_numer
    else:
        weight0 = (deficit0 * local_pref_denom) // local_pref_numer
        weight1 = deficit1
    weight0 = tl.where(has_rank0, weight0, 0)
    weight1 = tl.where(has_rank1, weight1, 0)
    total_w = weight0 + weight1

    token_seed = token_idx.to(tl.uint32) ^ (
        tl.full([BLOCK_SIZE], source_rank, dtype=tl.uint32)
        * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full([BLOCK_SIZE], 1664525, dtype=tl.uint32) + tl.full(
        [BLOCK_SIZE], 1013904223, dtype=tl.uint32
    )
    u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)
    chosen_rank = tl.where(u < weight0, 0, 1)

    if source_rank == 0:
        remote_better = (
            adjusted_load1 * local_pref_numer < load0 * local_pref_denom
        ) & has_rank1
    else:
        remote_better = (
            adjusted_load0 * local_pref_numer < load1 * local_pref_denom
        ) & has_rank0
    best_rank = tl.where(remote_better, 1 - source_rank, source_rank)
    chosen_rank = tl.where(total_w > 0, chosen_rank, best_rank).to(tl.int64)

    replica_seed = token_idx.to(tl.uint32) ^ (
        chosen_rank.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x85EBCA6B, dtype=tl.uint32)
    )
    replica_seed = replica_seed * tl.full(
        [BLOCK_SIZE], 1103515245, dtype=tl.uint32
    ) + tl.full([BLOCK_SIZE], 12345, dtype=tl.uint32)
    shared_replica = (replica_seed % shared_replicas_per_rank).to(tl.int64)
    shared_id = (
        chosen_rank * new_experts_per_rank + old_experts_per_rank + shared_replica
    )
    shared_id = tl.where(
        has_valid,
        shared_id,
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),
    )
    tl.store(
        expanded_ids_ptr + token_idx * (topk + 1) + topk,
        shared_id,
        mask=mask,
    )
    tl.store(
        expanded_weights_ptr + token_idx * (topk + 1) + topk,
        tl.where(has_valid, shared_weight, 0.0),
        mask=mask,
    )


def materialize_waterfill_dispatch_fused(
    topk_ids: Tensor,
    topk_weights: Tensor,
    rank_load: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
    allow_all_ranks: bool = False,
    target_total: int = 0,
    local_pref_numer: int = _LOCAL_PREF_NUMER,
    local_pref_denom: int = _LOCAL_PREF_DENOM,
    remote_cost_tokens: int = 0,
    shared_replicas_per_rank: int = 1,
    expanded_topk_ids: Optional[Tensor] = None,
    expanded_topk_weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run fused Waterfill rank selection and DeepEP TopK expansion.

    The Triton kernel intentionally selects each token's shared-expert rank and
    writes the expanded DeepEP TopK layout in one pass.
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    old_experts_per_rank = num_routed_experts // world_size
    shared_replicas_per_rank = max(shared_replicas_per_rank, 1)
    new_experts_per_rank = old_experts_per_rank + shared_replicas_per_rank
    device = topk_ids.device

    if num_tokens == 0:
        return _empty_expanded(topk_ids, topk_weights)

    if expanded_topk_ids is None:
        expanded_topk_ids = torch.empty(
            num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
        )
    if expanded_topk_weights is None:
        expanded_topk_weights = torch.empty(
            num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
        )
    BLOCK_SIZE = 256
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if world_size == 2 and allow_all_ranks:
        _waterfill_expand_rank2_static_kernel[grid](
            topk_ids,
            topk_weights,
            rank_load,
            expanded_topk_ids,
            expanded_topk_weights,
            num_tokens,
            topk,
            old_experts_per_rank,
            new_experts_per_rank,
            shared_replicas_per_rank,
            source_rank,
            shared_weight,
            LOCAL_SHARED_MARKER,
            local_pref_numer,
            local_pref_denom,
            remote_cost_tokens,
            envs.SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED.get(),
            BLOCK_SIZE,
        )
        return expanded_topk_ids, expanded_topk_weights

    if world_size == 2 and not allow_all_ranks:
        _waterfill_expand_rank2_candidate_kernel[grid](
            topk_ids,
            topk_weights,
            rank_load,
            expanded_topk_ids,
            expanded_topk_weights,
            num_tokens,
            topk,
            old_experts_per_rank,
            new_experts_per_rank,
            shared_replicas_per_rank,
            source_rank,
            shared_weight,
            LOCAL_SHARED_MARKER,
            local_pref_numer,
            local_pref_denom,
            remote_cost_tokens,
            BLOCK_SIZE,
        )
        return expanded_topk_ids, expanded_topk_weights

    _waterfill_expand_kernel[grid](
        topk_ids,
        topk_weights,
        rank_load,
        expanded_topk_ids,
        expanded_topk_weights,
        num_tokens,
        topk,
        old_experts_per_rank,
        new_experts_per_rank,
        shared_replicas_per_rank,
        world_size,
        source_rank,
        shared_weight,
        LOCAL_SHARED_MARKER,
        local_pref_numer,
        local_pref_denom,
        remote_cost_tokens,
        target_total,
        allow_all_ranks,
        BLOCK_SIZE,
    )

    return expanded_topk_ids, expanded_topk_weights


@torch.compile(dynamic=True)
def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
    shared_replicas_per_rank: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Expand topk [N, 8] → [N, 9] with ID remap; shared expert always local."""
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    old_epr = num_routed_experts // world_size
    shared_replicas_per_rank = max(shared_replicas_per_rank, 1)
    new_epr = old_epr + shared_replicas_per_rank
    has_valid = (topk_ids >= 0).any(dim=1)
    valid_mask = topk_ids >= 0
    old_ranks = torch.where(valid_mask, topk_ids // old_epr, torch.zeros_like(topk_ids))
    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_ids[:, :topk] = torch.where(
        valid_mask, topk_ids + old_ranks * shared_replicas_per_rank, topk_ids
    )

    shared_replica = (
        torch.arange(num_tokens, device=device, dtype=topk_ids.dtype)
        % shared_replicas_per_rank
    )
    shared_id = source_rank * new_epr + old_epr + shared_replica
    expanded_topk_ids[:, topk] = torch.where(has_valid, shared_id, LOCAL_SHARED_MARKER)
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = torch.where(valid_mask, topk_weights, 0.0)
    expanded_topk_weights[:, topk] = torch.where(has_valid, shared_weight, 0.0).to(
        topk_weights.dtype
    )
    return expanded_topk_ids, expanded_topk_weights


class DeepEPWaterfillBalancer:
    """Waterfill load balancer: shared expert fused as real routed expert (topk 8→9)."""

    _stats_calls = 0

    def __init__(
        self,
        num_routed_experts: int,
        world_size: int,
        rank: int,
        layer_id: int,
        routed_scaling_factor: float = 1.0,
    ):
        self.num_routed_experts = num_routed_experts
        self.world_size = world_size
        self.rank = rank
        self.layer_id = layer_id
        self.old_experts_per_rank = num_routed_experts // world_size
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )
        self.shared_replicas_per_rank = get_fused_shared_expert_replicas_per_rank()
        self._counts_buf: Optional[Tensor] = None
        self._expanded_topk_cache: OrderedDict[
            Tuple[object, object, object, int, int], Tuple[Tensor, Tensor]
        ] = OrderedDict()
        self.use_static_waterfill = not envs.SGLANG_DISABLE_STATIC_WATERFILL.get()
        self.static_rank_load: Optional[Tensor] = None
        self.static_rank_load_by_source: Optional[Tensor] = None

    def try_bind_static_rank_load(self) -> None:
        """Bind EPLB-derived per-rank load when logical_count metadata exists."""
        if (
            not self.use_static_waterfill
            or envs.SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD.get()
        ):
            return
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is None or metadata.rank_load is None:
            return
        if self.static_rank_load is not None:
            return
        if self.layer_id is None:
            return
        if self.layer_id >= metadata.rank_load.shape[0]:
            return
        layer_load = metadata.rank_load[self.layer_id]
        if layer_load.sum() <= 0:
            return
        self.static_rank_load = layer_load.to(dtype=torch.int64).contiguous()
        if (
            metadata.rank_load_by_source is not None
            and self.layer_id < metadata.rank_load_by_source.shape[0]
        ):
            layer_load_by_source = metadata.rank_load_by_source[self.layer_id]
            if layer_load_by_source.sum() > 0:
                self.static_rank_load_by_source = layer_load_by_source.to(
                    dtype=torch.int64
                ).contiguous()
        if self.rank == 0:
            logger.info(
                "Bound static Waterfill rank_load for layer %s: %s by_source=%s",
                self.layer_id,
                self.static_rank_load.detach().cpu().tolist(),
                (
                    self.static_rank_load_by_source.detach().cpu().tolist()
                    if self.static_rank_load_by_source is not None
                    else None
                ),
            )

    def _get_expanded_topk_buffers(
        self, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return reusable exact-shape output buffers for Waterfill.

        DeepGEMM's launch path appears sensitive to Tensor/view identity in
        addition to shape. Keep the same Tensor object for repeated shapes so
        Waterfill does not force per-layer launch metadata work every step.
        """
        num_tokens = topk_ids.shape[0]
        topk_plus_one = topk_ids.shape[1] + 1
        key = (
            topk_ids.device,
            topk_ids.dtype,
            topk_weights.dtype,
            num_tokens,
            topk_plus_one,
        )
        cached = self._expanded_topk_cache.get(key)
        if cached is not None:
            self._expanded_topk_cache.move_to_end(key)
            return cached

        cached = (
            torch.empty(
                num_tokens,
                topk_plus_one,
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            ),
            torch.empty(
                num_tokens,
                topk_plus_one,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            ),
        )
        self._expanded_topk_cache[key] = cached

        cache_size = max(envs.SGLANG_WATERFILL_REUSE_TOPK_BUFFER_CACHE_SIZE.get(), 1)
        while len(self._expanded_topk_cache) > cache_size:
            self._expanded_topk_cache.popitem(last=False)
        return cached

    def _maybe_log_stats(
        self,
        num_tokens: int,
        dispatch_plan: WaterfillDispatchPlan,
        expanded_ids: Tensor,
    ) -> None:
        interval = envs.SGLANG_WATERFILL_LOG_STATS_INTERVAL.get()
        if (
            interval <= 0
            or num_tokens == 0
            or (
                self.rank != 0
                and os.environ.get("SGLANG_WATERFILL_LOG_ALL_RANKS") != "1"
            )
        ):
            return

        DeepEPWaterfillBalancer._stats_calls += 1
        if DeepEPWaterfillBalancer._stats_calls % interval != 0:
            return

        valid = expanded_ids >= 0
        experts_per_rank = self.old_experts_per_rank + self.shared_replicas_per_rank
        ranks = torch.where(
            valid,
            expanded_ids // experts_per_rank,
            torch.zeros_like(expanded_ids),
        )
        after_counts = torch.bincount(
            ranks[valid].reshape(-1),
            minlength=self.world_size,
        )[: self.world_size]
        before_counts = dispatch_plan.rank_load
        shared_counts = after_counts - before_counts
        before = before_counts.detach().cpu().tolist()
        shared = shared_counts.detach().cpu().tolist()
        after = after_counts.detach().cpu().tolist()
        routed_ranks = ranks[:, :-1]
        routed_valid = valid[:, :-1]
        shared_ids = expanded_ids[:, -1]
        shared_valid = shared_ids >= 0
        shared_ranks = torch.where(
            shared_valid,
            shared_ids // experts_per_rank,
            torch.full_like(shared_ids, self.rank),
        )
        shared_remote = shared_valid & (shared_ranks != self.rank)
        shared_rank_in_routed = (
            (routed_ranks == shared_ranks[:, None]) & routed_valid
        ).any(dim=1)
        shared_remote_new_rank = shared_remote & ~shared_rank_in_routed
        logger.info(
            "WATERFILL_STATS ep_rank=%s layer=%s tokens=%s static=%s allow_all=%s "
            "source_aware=%s one_way=%s local_pref=%s/%s remote_cost=%s "
            "shared_replicas=%s target_total=%s "
            "before=%s shared=%s after=%s "
            "before_max_min=%s/%s after_max_min=%s/%s "
            "shared_remote=%s shared_remote_new_rank=%s",
            self.rank,
            self.layer_id,
            num_tokens,
            self.use_static_waterfill,
            dispatch_plan.allow_all_ranks,
            envs.SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD.get(),
            envs.SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED.get(),
            max(envs.SGLANG_WATERFILL_LOCAL_PREF_NUMER.get(), 1),
            max(envs.SGLANG_WATERFILL_LOCAL_PREF_DENOM.get(), 1),
            max(envs.SGLANG_WATERFILL_REMOTE_COST_TOKENS.get(), 0),
            self.shared_replicas_per_rank,
            dispatch_plan.target_total,
            before,
            shared,
            after,
            max(before) if before else 0,
            min(before) if before else 0,
            max(after) if after else 0,
            min(after) if after else 0,
            int(shared_remote.sum().item()),
            int(shared_remote_new_rank.sum().item()),
        )

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank via Triton kernel (uses original expert IDs)."""
        if self._counts_buf is None:
            self._counts_buf = torch.zeros(
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        buf = self._counts_buf
        num_tokens = topk_ids.shape[0]
        if num_tokens == 0:
            buf.zero_()
            return buf
        topk = topk_ids.shape[1]
        if self.world_size == 2:
            single_block_max = (
                envs.SGLANG_WATERFILL_RANK2_SINGLE_BLOCK_COUNT_MAX_TOKENS.get()
            )
            if num_tokens <= single_block_max:
                block_size = 1 << (num_tokens - 1).bit_length()
                num_warps = 8 if block_size >= 2048 else 4
                _count_routed_per_rank2_single_block_kernel[(1,)](
                    topk_ids,
                    buf,
                    num_tokens,
                    topk,
                    self.old_experts_per_rank,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
                return buf

            buf.zero_()
            BLOCK_SIZE = 256
            grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _count_routed_per_rank2_kernel[grid](
                topk_ids,
                buf,
                num_tokens,
                topk,
                self.old_experts_per_rank,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return buf

        buf.zero_()
        BLOCK_SIZE = 256
        grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _count_routed_per_rank_kernel[grid](
            topk_ids,
            buf,
            num_tokens,
            topk,
            self.old_experts_per_rank,
            self.world_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return buf

    def _is_low_batch(self, num_tokens: int) -> bool:
        """Return whether waterfill should skip balancing for small batches."""
        return num_tokens < max(envs.SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE.get(), 0)

    def _can_skip_dispatch_plan_for_low_batch(self, num_tokens: int) -> bool:
        """Return whether static mode can skip dispatch-plan setup entirely."""
        return self.use_static_waterfill and self._is_low_batch(num_tokens)

    def _build_static_dispatch_plan(
        self,
        routed_counts: Optional[Tensor],
        device: torch.device,
    ) -> WaterfillDispatchPlan:
        """Build static-mode Waterfill inputs without EP all-reduce."""
        if (
            envs.SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD.get()
            and routed_counts is not None
            and self.static_rank_load_by_source is not None
        ):
            rank_load_by_source = self.static_rank_load_by_source
            if rank_load_by_source.device != device:
                rank_load_by_source = rank_load_by_source.to(
                    device=device, non_blocking=True
                )
                self.static_rank_load_by_source = rank_load_by_source
            other_rank_load = (
                rank_load_by_source.sum(dim=0) - rank_load_by_source[self.rank]
            )
            rank_load = routed_counts + other_rank_load.to(dtype=routed_counts.dtype)
            return WaterfillDispatchPlan(
                rank_load=rank_load,
                allow_all_ranks=envs.SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS.get(),
                target_total=0,
            )

        if self.static_rank_load is not None:
            rank_load = self.static_rank_load
            if rank_load.device != device:
                rank_load = rank_load.to(device=device, non_blocking=True)
                self.static_rank_load = rank_load
            return WaterfillDispatchPlan(
                rank_load=rank_load,
                allow_all_ranks=envs.SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS.get(),
                target_total=0,
            )

        assert routed_counts is not None
        return WaterfillDispatchPlan(
            rank_load=routed_counts,
            allow_all_ranks=envs.SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS.get(),
            target_total=0,
        )

    def _build_dynamic_dispatch_plan(
        self,
        routed_counts: Tensor,
        topk: int,
    ) -> WaterfillDispatchPlan:
        """Build dynamic waterfill inputs from globally reduced routed counts."""
        # Dynamic Waterfill balances against the global routed incoming load.
        # Shared tokens are not part of the initial load; they are the work this
        # pass is about to assign, so they only enter through target_total.
        rank_load = routed_counts
        total_routed_t = routed_counts.sum()
        total_tokens_global_t = total_routed_t // topk
        total_effective_t = rank_load.sum()
        max_effective_t = rank_load.max()
        target_total = int(
            (total_effective_t + total_tokens_global_t + self.world_size - 1)
            // self.world_size
        )
        allow_all_ranks = bool(max_effective_t <= target_total)
        return WaterfillDispatchPlan(
            rank_load=rank_load,
            allow_all_ranks=allow_all_ranks,
            target_total=target_total,
        )

    @staticmethod
    def _all_reduce_dynamic_rank_load(
        local_routed_counts: Tensor,
    ) -> Tensor:
        """Aggregate dynamic load with SGLang EP communication."""
        from sglang.srt.distributed import get_moe_ep_group
        from sglang.srt.distributed.communication_op import (
            moe_expert_parallel_all_reduce,
        )

        group = get_moe_ep_group()
        world = group.world_size
        # SGLang's small custom all-reduce path currently supports fp dtypes
        # but not integer sentinels. Counts are far below fp32's exact-integer
        # range here, so reduce as fp32 and convert back for Waterfill math.
        buf = torch.zeros(world, dtype=torch.float32, device=local_routed_counts.device)
        buf[:world] = local_routed_counts.to(torch.float32)
        buf = moe_expert_parallel_all_reduce(buf)
        return buf.to(local_routed_counts.dtype)

    def _build_dispatch_plan(
        self, topk_ids: Tensor, num_tokens: int
    ) -> Optional[WaterfillDispatchPlan]:
        """Prepare dispatch state for the waterfill selection boundary."""
        if self._is_low_batch(num_tokens):
            return None

        if (
            self.use_static_waterfill
            and envs.SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD.get()
            and self.static_rank_load_by_source is not None
        ):
            local_routed_counts = self.count_local_routed(topk_ids)
            return self._build_static_dispatch_plan(
                local_routed_counts, topk_ids.device
            )

        if self.use_static_waterfill and self.static_rank_load is not None:
            return self._build_static_dispatch_plan(None, topk_ids.device)

        local_routed_counts = self.count_local_routed(topk_ids)
        if self.use_static_waterfill:
            return self._build_static_dispatch_plan(
                local_routed_counts, topk_ids.device
            )

        global_routed_counts = DeepEPWaterfillBalancer._all_reduce_dynamic_rank_load(
            local_routed_counts
        )
        return self._build_dynamic_dispatch_plan(
            global_routed_counts,
            topk=topk_ids.shape[1],
        )

    def build_dispatch_plan_for_topk(
        self, topk_ids: Tensor, num_tokens: int
    ) -> Optional[WaterfillDispatchPlan]:
        return self._build_dispatch_plan(topk_ids, num_tokens)

    def _materialize_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        dispatch_plan: WaterfillDispatchPlan,
    ) -> Tuple[Tensor, Tensor]:
        """Expand TopK using local expansion or fused Waterfill."""
        num_tokens = topk_ids.shape[0]
        if num_tokens == 0:
            return _empty_expanded(topk_ids, topk_weights)

        if self._is_low_batch(num_tokens):
            return expand_topk_with_shared_expert(
                topk_ids,
                topk_weights,
                self.num_routed_experts,
                self.world_size,
                self.rank,
                self.shared_weight,
                self.shared_replicas_per_rank,
            )

        expanded_ids = None
        expanded_weights = None
        if envs.SGLANG_WATERFILL_REUSE_TOPK_BUFFER.get():
            expanded_ids, expanded_weights = self._get_expanded_topk_buffers(
                topk_ids, topk_weights
            )

        return materialize_waterfill_dispatch_fused(
            topk_ids,
            topk_weights,
            dispatch_plan.rank_load,
            self.num_routed_experts,
            self.world_size,
            self.rank,
            self.shared_weight,
            allow_all_ranks=dispatch_plan.allow_all_ranks,
            target_total=dispatch_plan.target_total,
            local_pref_numer=max(envs.SGLANG_WATERFILL_LOCAL_PREF_NUMER.get(), 1),
            local_pref_denom=max(envs.SGLANG_WATERFILL_LOCAL_PREF_DENOM.get(), 1),
            remote_cost_tokens=max(envs.SGLANG_WATERFILL_REMOTE_COST_TOKENS.get(), 0),
            shared_replicas_per_rank=self.shared_replicas_per_rank,
            expanded_topk_ids=expanded_ids,
            expanded_topk_weights=expanded_weights,
        )

    @staticmethod
    def _with_expanded_topk(
        topk_output: StandardTopKOutput,
        expanded_ids: Tensor,
        expanded_weights: Tensor,
    ) -> StandardTopKOutput:
        """Wrap expanded tensors back into SGLang's StandardTopKOutput."""
        return StandardTopKOutput(
            topk_weights=expanded_weights,
            topk_ids=expanded_ids,
            router_logits=topk_output.router_logits,
        )

    def _expand_local_shared(
        self, topk_output: StandardTopKOutput
    ) -> StandardTopKOutput:
        expanded_ids, expanded_weights = expand_topk_with_shared_expert(
            topk_output.topk_ids,
            topk_output.topk_weights,
            self.num_routed_experts,
            self.world_size,
            self.rank,
            self.shared_weight,
            self.shared_replicas_per_rank,
        )
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)

    def expand_topk(
        self, topk_output: StandardTopKOutput, num_tokens: int
    ) -> StandardTopKOutput:
        """Expand topk [N, 8] -> [N, 9] with waterfill-assigned shared expert."""
        if envs.SGLANG_WATERFILL_FORCE_LOCAL_SHARED.get():
            return self._expand_local_shared(topk_output)

        if self._can_skip_dispatch_plan_for_low_batch(num_tokens):
            # Static mode can use local expansion without dispatch-plan setup for
            # small decode-sized batches.
            return self._expand_local_shared(topk_output)

        dispatch_plan = self._build_dispatch_plan(topk_output.topk_ids, num_tokens)
        if dispatch_plan is None:
            if num_tokens == 0:
                expanded_ids, expanded_weights = _empty_expanded(
                    topk_output.topk_ids, topk_output.topk_weights
                )
                return self._with_expanded_topk(
                    topk_output, expanded_ids, expanded_weights
                )
            else:
                return self._expand_local_shared(topk_output)
        expanded_ids, expanded_weights = self._materialize_dispatch(
            topk_output.topk_ids,
            topk_output.topk_weights,
            dispatch_plan,
        )
        self._maybe_log_stats(num_tokens, dispatch_plan, expanded_ids)
        return self._with_expanded_topk(topk_output, expanded_ids, expanded_weights)
