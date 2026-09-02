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

This first version handles the no-replication case.  It is a small, pure
PyTorch planner that is useful for validating the placement contract before
the runtime recorder and CLI are wired up.
"""

from __future__ import annotations

from typing import Tuple

import torch


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

    if (
        rank_cost_matrix.ndim != 2
        or rank_cost_matrix.shape != (num_ranks, num_ranks)
    ):
        raise ValueError("rank_cost_matrix must be square and match source ranks")
    if not torch.is_floating_point(rank_cost_matrix):
        raise TypeError("rank_cost_matrix must use a floating-point dtype")
    if not torch.isfinite(rank_cost_matrix).all() or torch.any(rank_cost_matrix < 0):
        raise ValueError("rank_cost_matrix must be finite and non-negative")
    if torch.any(torch.diagonal(rank_cost_matrix) != 0):
        raise ValueError("rank_cost_matrix diagonal must be zero")

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
    the same number of experts.  Experts are considered in descending total
    load; each one is assigned to the currently cheapest rank with remaining
    capacity.  Ties are resolved by rank id, making the result deterministic.

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
    experts_per_rank = num_physical_experts // num_ranks

    for layer in range(num_layers):
        total_load = counts[layer].sum(dim=0)
        communication_cost = counts[layer].transpose(0, 1).matmul(costs)
        order = sorted(
            range(num_logical_experts),
            key=lambda expert: (-float(total_load[expert]), expert),
        )
        remaining = [experts_per_rank] * num_ranks
        for expert in order:
            eligible = [rank for rank in range(num_ranks) if remaining[rank] > 0]
            destination_rank = min(
                eligible,
                key=lambda rank: (float(communication_cost[expert, rank]), rank),
            )
            assignment[layer, expert] = destination_rank
            remaining[destination_rank] -= 1

    return _assignment_to_maps(assignment, num_ranks)


__all__ = ["rebalance_experts_topology_aware"]
