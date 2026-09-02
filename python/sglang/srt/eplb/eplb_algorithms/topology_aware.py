# Copyright 2026 SGLang Team
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
# ============================================================================

"""Topology-aware expert placement primitives.

The existing EPLB algorithms receive one global count for each logical
expert.  A topology-aware planner needs one extra piece of information: how
many tokens each source rank sent to each expert.  This module deliberately
keeps that distinction explicit by accepting a ``[layers, source_ranks,
experts]`` tensor.

This first version handles the no-replication case.  It seeds the placement
with a load-balanced assignment and then applies communication-improving
pairwise swaps without increasing the baseline's maximum rank load.  The
implementation is deliberately small and pure PyTorch so it can run during a
rebalance without an external optimizer.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.eplb.topology import _validate_rank_cost_matrix


def _validate_inputs(
    tokens_per_source_expert: torch.Tensor,
    rank_cost_matrix: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int | None,
) -> tuple[int, int, int]:
    if tokens_per_source_expert.ndim != 3:
        raise ValueError(
            "tokens_per_source_expert must have shape "
            "[num_layers, num_source_ranks, num_logical_experts]"
        )
    num_layers, num_ranks, num_logical_experts = tokens_per_source_expert.shape
    if num_layers <= 0 or num_ranks <= 0 or num_logical_experts <= 0:
        raise ValueError("tokens_per_source_expert dimensions must be positive")
    if not torch.is_floating_point(tokens_per_source_expert) and not (
        tokens_per_source_expert.dtype == torch.int32
        or tokens_per_source_expert.dtype == torch.int64
    ):
        raise TypeError("tokens_per_source_expert must use an integer or float dtype")
    if torch.any(tokens_per_source_expert < 0) or (
        torch.is_floating_point(tokens_per_source_expert)
        and not torch.isfinite(tokens_per_source_expert).all()
    ):
        raise ValueError("tokens_per_source_expert must be finite and non-negative")

    if not torch.is_floating_point(rank_cost_matrix):
        raise TypeError("rank_cost_matrix must use a floating-point dtype")
    _validate_rank_cost_matrix(rank_cost_matrix, expected_num_ranks=num_ranks)

    if num_physical_experts != num_logical_experts:
        raise NotImplementedError(
            "topology-aware placement currently requires no redundant experts"
        )
    if num_physical_experts <= 0 or num_physical_experts % num_ranks != 0:
        raise ValueError("num_physical_experts must be positive and divisible by ranks")
    expected_local = num_physical_experts // num_ranks
    if num_local_physical_experts is not None and (
        num_local_physical_experts != expected_local
    ):
        raise ValueError(
            "num_local_physical_experts must equal "
            "num_physical_experts // num_source_ranks"
        )
    return num_layers, num_ranks, num_logical_experts


def _assignment_to_maps(
    assignment: torch.Tensor, num_ranks: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ``[layers, experts] -> destination rank`` into EPLB maps."""
    num_layers, num_experts = assignment.shape
    experts_per_rank = num_experts // num_ranks
    physical_to_logical = torch.empty_like(assignment)
    logical_to_physical = torch.empty(
        (num_layers, num_experts, 1), dtype=torch.int64
    )

    for layer in range(num_layers):
        next_slot = [0] * num_ranks
        for logical_expert in range(num_experts):
            destination_rank = int(assignment[layer, logical_expert])
            slot = next_slot[destination_rank]
            if slot >= experts_per_rank:
                raise ValueError("assignment exceeds per-rank expert capacity")
            physical_expert = destination_rank * experts_per_rank + slot
            physical_to_logical[layer, physical_expert] = logical_expert
            logical_to_physical[layer, logical_expert, 0] = physical_expert
            next_slot[destination_rank] += 1
        if next_slot != [experts_per_rank] * num_ranks:
            raise ValueError("assignment does not fill every physical expert slot")

    expert_count = torch.ones((num_layers, num_experts), dtype=torch.int64)
    return physical_to_logical, logical_to_physical, expert_count


def _load_balanced_seed(total_load: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """Build the same kind of load-first, equal-capacity seed as EPLB."""
    num_experts = total_load.numel()
    experts_per_rank = num_experts // num_ranks
    order = sorted(
        range(num_experts), key=lambda expert: (-float(total_load[expert]), expert)
    )
    assignment = torch.empty(num_experts, dtype=torch.int64)
    rank_load = [0.0] * num_ranks
    rank_items = [0] * num_ranks
    for expert in order:
        eligible = [
            rank for rank in range(num_ranks) if rank_items[rank] < experts_per_rank
        ]
        destination_rank = min(eligible, key=lambda rank: (rank_load[rank], rank))
        assignment[expert] = destination_rank
        rank_items[destination_rank] += 1
        rank_load[destination_rank] += float(total_load[expert])
    return assignment


def _improve_topology(
    assignment: torch.Tensor,
    total_load: torch.Tensor,
    communication_cost: torch.Tensor,
    num_ranks: int,
) -> torch.Tensor:
    """Apply deterministic improving swaps within the seed's load envelope."""
    rank_load = [0.0] * num_ranks
    for expert, rank in enumerate(assignment.tolist()):
        rank_load[rank] += float(total_load[expert])
    rank_load = torch.tensor(rank_load, dtype=torch.float64)
    max_seed_load = rank_load.max().item()

    num_experts = assignment.numel()
    expert_ids = torch.arange(num_experts, dtype=torch.int64)
    first_rank = assignment[:, None].expand(num_experts, num_experts)
    second_rank = assignment[None, :].expand(num_experts, num_experts)
    pair_mask = torch.triu(
        torch.ones((num_experts, num_experts), dtype=torch.bool), diagonal=1
    )
    pair_mask &= first_rank != second_rank

    # The rank loads of all non-swapped ranks are already within the seed's
    # envelope.  Therefore a proposed swap is feasible exactly when its two
    # changed ranks stay below that envelope; this avoids an O(num_ranks)
    # max-reduction for every pair.
    new_first_load = (
        rank_load[first_rank]
        - total_load[:, None]
        + total_load[None, :]
    )
    new_second_load = (
        rank_load[second_rank]
        - total_load[None, :]
        + total_load[:, None]
    )
    pair_mask &= new_first_load <= max_seed_load + 1e-12
    pair_mask &= new_second_load <= max_seed_load + 1e-12

    first_ids = expert_ids[:, None]
    second_ids = expert_ids[None, :]
    delta = (
        communication_cost[first_ids, second_rank]
        + communication_cost[second_ids, first_rank]
        - communication_cost[first_ids, first_rank]
        - communication_cost[second_ids, second_rank]
    )

    # Rebalancing runs in the serving process.  A bounded number of swaps keeps
    # the planner predictable for large expert counts while still allowing a
    # full pass over the ranks on the usual 8-way EP setup.
    for _ in range(max(1, num_ranks)):
        feasible_delta = delta.masked_fill(~pair_mask, float("inf"))
        feasible_delta = feasible_delta.masked_fill(feasible_delta >= -1e-12, float("inf"))
        best_flat = int(feasible_delta.argmin())
        if not torch.isfinite(feasible_delta.flatten()[best_flat]):
            return assignment

        first = best_flat // num_experts
        second = best_flat % num_experts
        first_rank_id = int(assignment[first])
        second_rank_id = int(assignment[second])
        assignment[first] = second_rank_id
        assignment[second] = first_rank_id
        rank_load[first_rank_id] = new_first_load[first, second]
        rank_load[second_rank_id] = new_second_load[first, second]

        # Keep the pair tensors synchronized with the changed assignment for
        # the next bounded iteration.
        first_rank = assignment[:, None].expand(num_experts, num_experts)
        second_rank = assignment[None, :].expand(num_experts, num_experts)
        new_first_load = (
            rank_load[first_rank]
            - total_load[:, None]
            + total_load[None, :]
        )
        new_second_load = (
            rank_load[second_rank]
            - total_load[None, :]
            + total_load[:, None]
        )
        pair_mask = torch.triu(
            torch.ones((num_experts, num_experts), dtype=torch.bool), diagonal=1
        )
        pair_mask &= first_rank != second_rank
        pair_mask &= new_first_load <= max_seed_load + 1e-12
        pair_mask &= new_second_load <= max_seed_load + 1e-12
        delta = (
            communication_cost[first_ids, second_rank]
            + communication_cost[second_ids, first_rank]
            - communication_cost[first_ids, first_rank]
            - communication_cost[second_ids, second_rank]
        )

    return assignment


def rebalance_experts_topology_aware(
    tokens_per_source_expert: torch.Tensor,
    rank_cost_matrix: torch.Tensor,
    *,
    num_physical_experts: int | None = None,
    num_local_physical_experts: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place experts using source-to-destination communication costs.

    ``tokens_per_source_expert[layer, source, expert]`` is the number of
    routed tokens originating at ``source`` for ``expert``.  Each logical
    expert is assigned to exactly one destination rank, and every rank gets
    the same number of experts.  A load-balanced seed is improved with
    communication-reducing swaps that keep the seed's maximum rank load as an
    upper bound.  Ties are resolved by expert and rank id, making the result
    deterministic.

    The return value follows ``rebalance_experts``: physical-to-logical map,
    logical-to-physical map, and the per-expert replica count.
    """
    if num_physical_experts is None:
        num_physical_experts = int(tokens_per_source_expert.shape[-1])
    num_layers, num_ranks, num_logical_experts = _validate_inputs(
        tokens_per_source_expert,
        rank_cost_matrix,
        num_physical_experts,
        num_local_physical_experts,
    )

    counts = tokens_per_source_expert.to(dtype=torch.float64, device="cpu")
    costs = rank_cost_matrix.to(dtype=torch.float64, device="cpu")
    assignment = torch.empty(
        (num_layers, num_logical_experts), dtype=torch.int64, device="cpu"
    )
    for layer in range(num_layers):
        total_load = counts[layer].sum(dim=0)
        communication_cost = counts[layer].transpose(0, 1).matmul(costs)
        assignment[layer] = _load_balanced_seed(total_load, num_ranks)
        assignment[layer] = _improve_topology(
            assignment[layer], total_load, communication_cost, num_ranks
        )

    return _assignment_to_maps(assignment, num_ranks)


__all__ = ["rebalance_experts_topology_aware"]
