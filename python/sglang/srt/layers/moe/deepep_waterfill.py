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
"""DeepEP Waterfill: shared expert as 9th routed expert, dispatched to least-loaded rank."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput

LOCAL_SHARED_MARKER = -1  # Invalid expert ID; DeepEP ignores expert_id < 0.
LOCAL_PREFERENCE_FACTOR = 1.1  # Bias towards local rank; 1.0 = pure argmin.
_LOCAL_PREF_NUMER = int(LOCAL_PREFERENCE_FACTOR * 5)
_LOCAL_PREF_DENOM = 5


def _empty_expanded(topk_ids: Tensor, topk_weights: Tensor):
    """Return empty expanded tensors for zero-token batches."""
    topk = topk_ids.shape[1]
    device = topk_ids.device
    return (
        torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=device),
        torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=device),
        torch.empty(0, dtype=torch.bool, device=device),
    )


# ============== Triton Kernels ==============


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
    """Fused waterfill + expand + histogram. ID remap: old_id -> old_id + old_id // old_epr."""
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
    target_total = tl.where(
        precomputed_target_total > 0,
        precomputed_target_total,
        derived_target_total,
    )

    # Step 1: Select destination rank for shared expert (waterfill sampling).
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

    total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
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
        total_w += tl.where(present, w_vec, 0)

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

    # Step 2: Compute shared expert ID and local mask.
    is_local = best_rank == source_rank
    local_shared_id = source_rank * new_experts_per_rank + old_experts_per_rank
    remote_shared_id = best_rank * new_experts_per_rank + old_experts_per_rank
    shared_expert_id = tl.where(
        is_local,
        tl.full([BLOCK_SIZE], local_shared_id, dtype=tl.int64),
        remote_shared_id,
    ).to(tl.int64)
    shared_expert_id = tl.where(
        has_valid,
        shared_expert_id,
        tl.full([BLOCK_SIZE], local_marker, dtype=tl.int64),
    )

    dest_rank = tl.where(is_local, source_rank, best_rank).to(tl.int32)

    # Step 3: Copy and remap topk_ids, copy weights.
    for k in range(topk):
        old_id = tl.load(topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1).to(
            tl.int64
        )
        valid_id = old_id >= 0
        new_id = tl.where(valid_id, old_id + (old_id // old_experts_per_rank), old_id)
        tl.store(expanded_ids_ptr + token_idx * (topk + 1) + k, new_id, mask=mask)

    for k in range(topk):
        val = tl.load(topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0)
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        val = tl.where(expert_id >= 0, val, 0.0)
        tl.store(expanded_weights_ptr + token_idx * (topk + 1) + k, val, mask=mask)

    # Step 4: Write shared expert column and local mask.
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
    tl.store(local_mask_ptr + token_idx, is_local, mask=mask)

    # Step 5: Block-level histogram with minimal atomics.
    for r in range(world_size):
        rank_count = tl.sum(tl.where(mask & has_valid & (dest_rank == r), 1, 0))
        if rank_count > 0:
            tl.atomic_add(dest_counts_ptr + r, rank_count)


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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused waterfill + expand + ID remapping via Triton kernel."""
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1
    device = topk_ids.device

    if num_tokens == 0:
        return _empty_expanded(topk_ids, topk_weights)

    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    local_shared_mask = torch.empty(num_tokens, dtype=torch.bool, device=device)

    BLOCK_SIZE = 256
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
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
        _LOCAL_PREF_NUMER,
        _LOCAL_PREF_DENOM,
        target_total,
        allow_all_ranks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expanded_topk_ids, expanded_topk_weights, local_shared_mask


def expand_topk_with_shared_expert(
    topk_ids: Tensor,
    topk_weights: Tensor,
    shared_destination: Tensor,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Expand topk [N, 8] → [N, 9] with ID remap and shared expert placement."""
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    old_epr = num_routed_experts // world_size
    new_epr = old_epr + 1

    has_valid = (topk_ids >= 0).any(dim=1)
    valid_mask = topk_ids >= 0

    # Remap: old_id -> old_id + (old_id // old_epr)
    old_ranks = torch.where(valid_mask, topk_ids // old_epr, torch.zeros_like(topk_ids))
    expanded_topk_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_topk_ids[:, :topk] = torch.where(
        valid_mask, topk_ids + old_ranks, topk_ids
    )

    # Shared expert column
    shared_ids = shared_destination * new_epr + old_epr
    expanded_topk_ids[:, topk] = torch.where(
        has_valid, shared_ids.to(topk_ids.dtype), LOCAL_SHARED_MARKER
    )

    # Weights: copy routed, add shared weight column
    expanded_topk_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    expanded_topk_weights[:, :topk] = topk_weights
    expanded_topk_weights[:, topk] = torch.where(has_valid, shared_weight, 0.0).to(
        topk_weights.dtype
    )
    if (~has_valid).any():
        expanded_topk_weights[~has_valid, :topk] = 0.0

    local_shared_mask = (shared_destination == source_rank) & has_valid
    return expanded_topk_ids, expanded_topk_weights, local_shared_mask


def compute_static_rank_load(
    logical_count: Tensor,
    physical_to_logical_map: Tensor,
    world_size: int,
) -> Tensor:
    """Compute per-layer static rank load [num_layers, world_size] from EPLB statistics."""
    num_layers, num_physical_experts = physical_to_logical_map.shape
    num_logical_experts = logical_count.shape[-1]
    experts_per_rank = num_physical_experts // world_size

    device = physical_to_logical_map.device
    logical_count = logical_count.to(device=device, dtype=torch.float64)
    physical_to_logical_map = physical_to_logical_map.to(device=device)

    ones = torch.ones(
        num_layers, num_physical_experts, dtype=torch.float64, device=device
    )
    replica_counts = torch.zeros(
        num_layers, num_logical_experts, dtype=torch.float64, device=device
    )
    replica_counts.scatter_add_(1, physical_to_logical_map.long(), ones)
    replica_counts = replica_counts.clamp(min=1.0)

    mapped_logical_ids = physical_to_logical_map.long()
    physical_load = torch.gather(logical_count, 1, mapped_logical_ids)
    physical_replica = torch.gather(replica_counts, 1, mapped_logical_ids)
    physical_load = physical_load / physical_replica

    return physical_load.view(num_layers, world_size, experts_per_rank).sum(dim=2)


class DeepEPWaterfillBalancer:
    """Waterfill load balancer: shared expert fused as real routed expert (topk 8→9)."""

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        num_routed_experts: int,
        world_size: int,
        rank: int,
        routed_scaling_factor: float = 1.0,
        static_rank_load: Optional[Tensor] = None,
    ):
        self.num_routed_experts = num_routed_experts
        self.world_size = world_size
        self.rank = rank
        self.old_experts_per_rank = num_routed_experts // world_size
        self.new_experts_per_rank = self.old_experts_per_rank + 1
        self.routed_scaling_factor = routed_scaling_factor
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )
        self.static_rank_load: Optional[Tensor] = static_rank_load
        self._counts_buf: Optional[Tensor] = None

    def has_static_weights(self) -> bool:
        return self.static_rank_load is not None

    def set_static_weights(self, static_rank_load: Tensor) -> None:
        """Replace static per-rank load weights (e.g. after EPLB rebalance)."""
        assert static_rank_load.shape == (
            self.world_size,
        ), f"Expected ({self.world_size},), got {static_rank_load.shape}"
        self.static_rank_load = static_rank_load.to(dtype=torch.float64)

    def count_local_routed(self, topk_ids: Tensor) -> Tensor:
        """Count routed tokens per rank via Triton kernel (uses original expert IDs)."""
        if self._counts_buf is None:
            self._counts_buf = torch.zeros(
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        buf = self._counts_buf
        buf.zero_()
        num_tokens = topk_ids.shape[0]
        if num_tokens == 0:
            return buf
        topk = topk_ids.shape[1]
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

    def prepare_dispatch(
        self,
        topk_ids: Tensor,
        topk_weights: Tensor,
        routed_counts: Tensor,
        local_tokens_per_rank: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Expand topk [N, 8] → [N, 9] with waterfill-assigned shared expert."""
        num_tokens = topk_ids.shape[0]
        device = topk_ids.device

        if num_tokens == 0:
            return _empty_expanded(topk_ids, topk_weights)

        if num_tokens < self.MIN_BATCH_FOR_BALANCE:
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

        routed_counts_i64 = routed_counts.to(torch.int64)
        effective_load = (
            routed_counts_i64 + local_tokens_per_rank.to(torch.int64)
            if local_tokens_per_rank is not None
            else routed_counts_i64
        )
        topk = topk_ids.shape[1]

        if self.has_static_weights():
            allow_all_ranks = True
            target_total = 0
        else:
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

        return waterfill_prepare_dispatch_fused(
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

    def expand_topk(self, topk_output: TopKOutput, num_tokens: int) -> TopKOutput:
        """Expand topk [N, 8] -> [N, 9] with waterfill-assigned shared expert."""
        from sglang.srt.distributed import get_moe_ep_group

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        device = topk_ids.device

        local_routed_counts = self.count_local_routed(topk_ids)

        if self.has_static_weights():
            global_routed_counts = local_routed_counts
            local_tokens_per_rank = None
        else:
            _ep_group = get_moe_ep_group().device_group
            _ep_world = torch.distributed.get_world_size(group=_ep_group)
            _ep_rank = torch.distributed.get_rank(group=_ep_group)
            _fused_buf = torch.zeros(_ep_world * 2, dtype=torch.int64, device=device)
            _fused_buf[:_ep_world] = local_routed_counts
            if not torch.cuda.is_current_stream_capturing():
                _fused_buf[_ep_world + _ep_rank] = num_tokens
            torch.distributed.all_reduce(
                _fused_buf,
                op=torch.distributed.ReduceOp.SUM,
                group=_ep_group,
            )
            global_routed_counts = _fused_buf[:_ep_world]
            if not torch.cuda.is_current_stream_capturing():
                local_tokens_per_rank = _fused_buf[_ep_world:]
            else:
                local_tokens_per_rank = None

        expanded_topk_ids, expanded_topk_weights, local_shared_mask = (
            self.prepare_dispatch(
                topk_ids,
                topk_weights,
                global_routed_counts,
                local_tokens_per_rank=local_tokens_per_rank,
            )
        )

        return StandardTopKOutput(
            topk_weights=expanded_topk_weights,
            topk_ids=expanded_topk_ids,
            router_logits=topk_output.router_logits,
        )
