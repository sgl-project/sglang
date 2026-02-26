import torch

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    PickParams,
    UnsharderParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.output_types import ReplicatedMismatchWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink


def execute_unsharder_plan(
    plan: UnsharderPlan,
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []

    for group_idx, group in enumerate(plan.groups):
        group_tensors = [tensors[i] for i in group]
        tensor = _apply_unshard(
            plan.params,
            group_tensors,
            axis=plan.axis,
            group_index=group_idx,
        )
        result.append(tensor)

    return result


def _apply_unshard(
    params: UnsharderParams,
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> torch.Tensor:
    if isinstance(params, PickParams):
        _verify_replicated_group(
            ordered_tensors,
            axis=axis,
            group_index=group_index,
        )
        return ordered_tensors[0]

    if isinstance(params, ConcatParams):
        return torch.cat(ordered_tensors, dim=params.dim)

    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _verify_replicated_group(
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> None:
    baseline = ordered_tensors[0]

    for i in range(1, len(ordered_tensors)):
        other = ordered_tensors[i]
        if not torch.allclose(baseline, other, atol=1e-6):
            warning_sink.add(
                ReplicatedMismatchWarning(
                    axis=axis.value,
                    group_index=group_index,
                    differing_index=i,
                    baseline_index=0,
                    max_abs_diff=(baseline - other).abs().max().item(),
                )
            )
