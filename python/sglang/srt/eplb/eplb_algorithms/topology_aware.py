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

This first version handles the no-replication case.  It starts from the
default node-uniform placement, applies communication-improving pairwise
swaps, and then spends a tiny bounded communication budget on reducing the
busiest destination rank.  The swaps never exceed that placement's
busiest-rank load; keeping the default compute envelope matters because the
MoE runner's kernel shape and occupancy depend on which experts share a rank.
The implementation is deliberately small and pure PyTorch so it can run
during a rebalance without an external optimizer.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.eplb.topology import _validate_rank_cost_matrix

# A tiny communication budget lets the planner remove a critical-rank hot
# spot without allowing the topology objective to drift materially.  The
# default node-uniform fallback below still protects against a net regression.
_MAX_COMM_REGRESSION_RATIO = 5e-4

# Keep part of the bounded local-search budget for the serving critical path.
# Without an explicit reservation, the communication phase can consume every
# swap and leave a hot destination rank untouched.
_COMMUNICATION_PHASE_FRACTION = 0.5


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

    if num_physical_experts > num_logical_experts:
        raise NotImplementedError(
            "topology-aware placement currently requires no redundant experts"
        )
    if num_physical_experts != num_logical_experts:
        raise ValueError(
            "num_physical_experts must match the source-count expert dimension"
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
    logical_to_physical = torch.empty((num_layers, num_experts, 1), dtype=torch.int64)

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


def _node_uniform_seed(num_experts: int, num_ranks: int) -> torch.Tensor:
    """Return the default contiguous expert-to-rank assignment."""
    experts_per_rank = num_experts // num_ranks
    return torch.arange(num_experts, dtype=torch.int64) // experts_per_rank


def _rank_loads(
    assignment: torch.Tensor, total_load: torch.Tensor, num_ranks: int
) -> torch.Tensor:
    """Sum logical-expert traffic for each destination rank."""
    loads = torch.zeros(num_ranks, dtype=torch.float64)
    loads.scatter_add_(0, assignment, total_load.to(torch.float64))
    return loads


def _build_swap_state(
    assignment: torch.Tensor,
    total_load: torch.Tensor,
    communication_cost: torch.Tensor,
    num_ranks: int,
    rank_load: torch.Tensor,
    max_seed_load: float,
    expert_ids: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the pairwise swap tensors for the current assignment."""
    num_experts = assignment.numel()
    first_rank = assignment[:, None].expand(num_experts, num_experts)
    second_rank = assignment[None, :].expand(num_experts, num_experts)
    pair_mask = torch.triu(
        torch.ones(
            (num_experts, num_experts),
            dtype=torch.bool,
            device=assignment.device,
        ),
        diagonal=1,
    )
    pair_mask &= first_rank != second_rank

    # A swap only changes the loads of the two ranks that own its experts.
    # Every other rank is already within the seed envelope.
    new_first_load = rank_load[first_rank] - total_load[:, None] + total_load[None, :]
    new_second_load = rank_load[second_rank] - total_load[None, :] + total_load[:, None]
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
    return (
        first_rank,
        second_rank,
        pair_mask,
        new_first_load,
        new_second_load,
        delta,
    )


def _apply_swap(
    assignment: torch.Tensor,
    rank_load: torch.Tensor,
    new_first_load: torch.Tensor,
    new_second_load: torch.Tensor,
    first: int,
    second: int,
) -> None:
    """Apply one expert swap and update the two affected rank loads."""
    first_rank = int(assignment[first])
    second_rank = int(assignment[second])
    assignment[first] = second_rank
    assignment[second] = first_rank
    rank_load[first_rank] = new_first_load[first, second]
    rank_load[second_rank] = new_second_load[first, second]


def _run_communication_swaps(
    assignment: torch.Tensor,
    total_load: torch.Tensor,
    communication_cost: torch.Tensor,
    num_ranks: int,
    rank_load: torch.Tensor,
    max_seed_load: float,
    expert_ids: torch.Tensor,
    max_swaps: int,
) -> int:
    """Apply the bounded phase that strictly reduces communication cost."""
    used_swaps = 0
    for _ in range(max_swaps):
        (
            _first_rank,
            _second_rank,
            pair_mask,
            new_first_load,
            new_second_load,
            delta,
        ) = _build_swap_state(
            assignment,
            total_load,
            communication_cost,
            num_ranks,
            rank_load,
            max_seed_load,
            expert_ids,
        )
        feasible_delta = delta.masked_fill(~pair_mask, float("inf"))
        feasible_delta = feasible_delta.masked_fill(
            feasible_delta >= -1e-12, float("inf")
        )
        best_flat = int(feasible_delta.argmin())
        if not torch.isfinite(feasible_delta.flatten()[best_flat]):
            break

        first = best_flat // assignment.numel()
        second = best_flat % assignment.numel()
        _apply_swap(
            assignment,
            rank_load,
            new_first_load,
            new_second_load,
            first,
            second,
        )
        used_swaps += 1
    return used_swaps


def _run_load_swaps(
    assignment: torch.Tensor,
    total_load: torch.Tensor,
    communication_cost: torch.Tensor,
    num_ranks: int,
    rank_load: torch.Tensor,
    max_seed_load: float,
    expert_ids: torch.Tensor,
    max_swaps: int,
    used_swaps: int,
    max_comm_regression_ratio: float,
) -> None:
    """Spend the explicit communication budget on critical-rank balancing."""
    if max_comm_regression_ratio == 0.0 or used_swaps >= max_swaps:
        return

    communication_limit = communication_cost[expert_ids, assignment].sum().item() * (
        1.0 + max_comm_regression_ratio
    )
    rank_ids = torch.arange(num_ranks, dtype=torch.int64, device=assignment.device)[
        None, None, :
    ]
    for _ in range(max_swaps - used_swaps):
        (
            first_rank,
            second_rank,
            pair_mask,
            new_first_load,
            new_second_load,
            delta_comm,
        ) = _build_swap_state(
            assignment,
            total_load,
            communication_cost,
            num_ranks,
            rank_load,
            max_seed_load,
            expert_ids,
        )
        pair_mask &= (
            communication_cost[expert_ids, assignment].sum() + delta_comm
            <= communication_limit + 1e-12
        )

        changed_rank = (rank_ids == first_rank[:, :, None]) | (
            rank_ids == second_rank[:, :, None]
        )
        other_max = (
            rank_load[None, None, :]
            .masked_fill(changed_rank, float("-inf"))
            .amax(dim=-1)
        )
        new_max = torch.maximum(
            torch.maximum(new_first_load, new_second_load), other_max
        )
        old_max = rank_load.max()
        old_sumsq = rank_load.square().sum()
        new_sumsq = (
            old_sumsq
            - rank_load[first_rank].square()
            - rank_load[second_rank].square()
            + new_first_load.square()
            + new_second_load.square()
        )
        max_improving = new_max < old_max - 1e-12
        max_tied = (new_max - old_max).abs() <= 1e-12
        pair_mask &= max_improving | (max_tied & (new_sumsq < old_sumsq - 1e-12))

        feasible_max = new_max.masked_fill(~pair_mask, float("inf"))
        best_max = feasible_max.min()
        if not torch.isfinite(best_max):
            break
        tied_mask = pair_mask & ((new_max - best_max).abs() <= 1e-12)
        feasible_sumsq = new_sumsq.masked_fill(~tied_mask, float("inf"))
        best_flat = int(feasible_sumsq.argmin())

        first = best_flat // assignment.numel()
        second = best_flat % assignment.numel()
        _apply_swap(
            assignment,
            rank_load,
            new_first_load,
            new_second_load,
            first,
            second,
        )


def _improve_topology(
    assignment: torch.Tensor,
    total_load: torch.Tensor,
    communication_cost: torch.Tensor,
    num_ranks: int,
    max_allowed_load: float | None = None,
    max_swaps: int | None = None,
    max_comm_regression_ratio: float = 0.0,
) -> torch.Tensor:
    """Apply deterministic swaps within a rank-load envelope.

    The first phase only accepts communication-improving swaps.  An optional
    small communication budget can then be spent on swaps that lower the
    busiest destination rank.  The latter models the serving critical path:
    one overloaded rank can hold up the whole expert-parallel step even when
    the aggregate communication score is marginally better.
    """
    if max_comm_regression_ratio < 0:
        raise ValueError("max_comm_regression_ratio must be non-negative")
    rank_load = _rank_loads(assignment, total_load, num_ranks)
    max_seed_load = (
        rank_load.max().item() if max_allowed_load is None else float(max_allowed_load)
    )
    num_experts = assignment.numel()
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=assignment.device)
    if max_swaps is None:
        # Five rounds per local expert slot reach the useful local-search basin
        # on the observed 8-way EP workloads without a long scheduler pause.
        max_swaps = max(1, 5 * (num_experts // num_ranks))

    communication_swaps = max_swaps
    if max_comm_regression_ratio > 0.0:
        # Reserve the remaining rounds for the critical-rank load phase.  A
        # communication-only search can otherwise consume the entire budget,
        # leaving no opportunity to fix a straggling destination rank.
        communication_swaps = max(1, int(max_swaps * _COMMUNICATION_PHASE_FRACTION))

    used_swaps = _run_communication_swaps(
        assignment,
        total_load,
        communication_cost,
        num_ranks,
        rank_load,
        max_seed_load,
        expert_ids,
        communication_swaps,
    )
    _run_load_swaps(
        assignment,
        total_load,
        communication_cost,
        num_ranks,
        rank_load,
        max_seed_load,
        expert_ids,
        max_swaps,
        used_swaps,
        max_comm_regression_ratio,
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
    the same number of experts.  The default node-uniform layout is improved
    with communication-reducing swaps followed by a tiny, bounded
    load-balancing phase.  The maximum rank load never exceeds the default
    layout's busiest rank, and the default layout is used as a
    communication-cost fallback.  Ties are resolved by expert and rank id,
    making the result deterministic.

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
    node_uniform_assignment = _node_uniform_seed(num_logical_experts, num_ranks)
    expert_ids = torch.arange(num_logical_experts, dtype=torch.int64)
    for layer in range(num_layers):
        total_load = counts[layer].sum(dim=0)
        communication_cost = counts[layer].transpose(0, 1).matmul(costs)
        node_uniform_load = _rank_loads(node_uniform_assignment, total_load, num_ranks)
        max_allowed_load = node_uniform_load.max().item()
        assignment[layer] = _improve_topology(
            node_uniform_assignment.clone(),
            total_load,
            communication_cost,
            num_ranks,
            max_allowed_load=max_allowed_load,
            max_comm_regression_ratio=_MAX_COMM_REGRESSION_RATIO,
        )
        # A bounded local search can stop before finding a good swap sequence.
        # The node-uniform layout is always a valid fallback, so never return a
        # topology placement with a higher communication objective.
        candidate_cost = communication_cost[expert_ids, assignment[layer]].sum()
        node_uniform_cost = communication_cost[
            expert_ids, node_uniform_assignment
        ].sum()
        if candidate_cost > node_uniform_cost:
            assignment[layer] = node_uniform_assignment

    return _assignment_to_maps(assignment, num_ranks)


__all__ = ["rebalance_experts_topology_aware"]
