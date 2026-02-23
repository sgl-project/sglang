from collections import defaultdict

import torch

from sglang.srt.debug_utils.comparator.unshard.types import (
    ConcatParams,
    UnshardPlan,
    UnshardStep,
)


def unshard_concat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors_by_world_rank: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Execute an unshard plan on actual tensor data.

    Concatenation order is strictly by axis_rank (not world_rank).
    """
    if not plan.steps:
        if len(tensors_by_world_rank) == 1:
            return next(iter(tensors_by_world_rank.values()))
        raise ValueError(
            f"No unshard steps but got {len(tensors_by_world_rank)} tensors"
        )

    axis_rank_lookup: dict[int, dict[int, int]] = {
        id(step): {wr: i for i, wr in enumerate(step.world_ranks_by_axis_rank)}
        for step in plan.steps
    }

    current_tensors = dict(tensors_by_world_rank)

    for step in plan.steps:
        groups: dict[tuple, dict[int, torch.Tensor]] = defaultdict(dict)

        for world_rank, tensor in current_tensors.items():
            group_key = tuple(
                sorted(
                    (s.axis.value, axis_rank_lookup[id(s)].get(world_rank, -1))
                    for s in plan.steps
                    if s is not step
                )
            )
            groups[group_key][world_rank] = tensor

        new_tensors: dict[int, torch.Tensor] = {}
        for members in groups.values():
            ordered = [
                members[wr] for wr in step.world_ranks_by_axis_rank if wr in members
            ]
            if not ordered:
                continue

            merged = _execute_step(step, ordered)
            representative_rank = next(
                wr for wr in step.world_ranks_by_axis_rank if wr in members
            )
            new_tensors[representative_rank] = merged

        current_tensors = new_tensors

    if len(current_tensors) != 1:
        raise ValueError(f"Expected 1 tensor after unshard, got {len(current_tensors)}")

    return next(iter(current_tensors.values()))


def _execute_step(
    step: UnshardStep, ordered_tensors: list[torch.Tensor]
) -> torch.Tensor:
    params = step.params
    if isinstance(params, ConcatParams):
        return unshard_concat(ordered_tensors, dim=params.dim)
    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")
