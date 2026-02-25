import torch

from sglang.srt.debug_utils.comparator.unshard.types import (
    ConcatParams,
    UnshardPlan,
)


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors_by_world_rank: dict[int, torch.Tensor],
) -> torch.Tensor:
    ordered_tensors = [
        tensors_by_world_rank[world_rank]
        for world_rank in plan.world_ranks_by_axis_rank
    ]

    return _apply_unshard(plan, ordered_tensors)


def _apply_unshard(
    plan: UnshardPlan, ordered_tensors: list[torch.Tensor]
) -> torch.Tensor:
    params = plan.params
    if isinstance(params, ConcatParams):
        return _unshard_concat(ordered_tensors, dim=params.dim)
    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _unshard_concat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)
