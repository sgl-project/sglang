import torch

from sglang.srt.debug_utils.comparator.aligner.unshard.types import (
    ConcatParams,
    PickParams,
    UnshardParams,
    UnshardPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.output_types import (
    AlignWarning,
    ReplicatedMismatchWarning,
)


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    all_warnings: list[AlignWarning] = []
    result: list[torch.Tensor] = []

    for group_idx, group in enumerate(plan.groups):
        group_tensors = [tensors[i] for i in group]
        tensor, warnings = _apply_unshard(
            plan.params,
            group_tensors,
            axis=plan.axis,
            group_index=group_idx,
        )
        result.append(tensor)
        all_warnings.extend(warnings)

    return result, all_warnings


def _apply_unshard(
    params: UnshardParams,
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> tuple[torch.Tensor, list[AlignWarning]]:
    if isinstance(params, PickParams):
        warnings = _verify_replicated_group(
            ordered_tensors,
            axis=axis,
            group_index=group_index,
        )
        return ordered_tensors[0], warnings

    if isinstance(params, ConcatParams):
        return torch.cat(ordered_tensors, dim=params.dim), []

    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _verify_replicated_group(
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> list[ReplicatedMismatchWarning]:
    warnings: list[ReplicatedMismatchWarning] = []
    baseline = ordered_tensors[0]

    for i in range(1, len(ordered_tensors)):
        other = ordered_tensors[i]
        if not torch.allclose(baseline, other, atol=1e-6):
            warnings.append(
                ReplicatedMismatchWarning(
                    axis=axis.value,
                    group_index=group_index,
                    differing_index=i,
                    baseline_index=0,
                    max_abs_diff=(baseline - other).abs().max().item(),
                )
            )

    return warnings
