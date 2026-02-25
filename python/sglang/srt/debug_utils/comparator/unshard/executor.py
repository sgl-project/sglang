import torch

from sglang.srt.debug_utils.comparator.unshard.types import (
    ConcatParams,
    UnshardParams,
    UnshardPlan,
)


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for group in plan.groups:
        group_tensors = [tensors[i] for i in group]
        result.append(_apply_unshard(plan.params, group_tensors))
    return result


def _apply_unshard(
    params: UnshardParams, ordered_tensors: list[torch.Tensor]
) -> torch.Tensor:
    if isinstance(params, ConcatParams):
        return _unshard_concat(ordered_tensors, dim=params.dim)
    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _unshard_concat(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)
