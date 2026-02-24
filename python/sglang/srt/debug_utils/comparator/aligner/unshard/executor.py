import torch

from sglang.srt.debug_utils.comparator.aligner.unshard.types import (
    ConcatParams,
    PickParams,
    UnshardParams,
    UnshardPlan,
)
from sglang.srt.debug_utils.comparator.output_types import (
    AlignWarning,
    ReplicatedMismatchWarning,
)


def execute_unshard_plan(
    plan: UnshardPlan,
    tensors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[AlignWarning]]:
    warnings = verify_replicated_groups(plan=plan, tensors=tensors)

    result: list[torch.Tensor] = []
    for group in plan.groups:
        group_tensors = [tensors[i] for i in group]
        result.append(_apply_unshard(plan.params, group_tensors, axis=plan.axis))

    return result, warnings


def verify_replicated_groups(
    *,
    plan: UnshardPlan,
    tensors: list[torch.Tensor],
) -> list[ReplicatedMismatchWarning]:
    if not isinstance(plan.params, PickParams):
        return []

    warnings: list[ReplicatedMismatchWarning] = []
    for group_idx, group in enumerate(plan.groups):
        baseline = tensors[group[0]]
        for i in range(1, len(group)):
            other = tensors[group[i]]
            if not torch.allclose(baseline, other, atol=1e-6):
                warnings.append(
                    ReplicatedMismatchWarning(
                        axis=plan.axis.value,
                        group_index=group_idx,
                        differing_index=group[i],
                        baseline_index=group[0],
                        max_abs_diff=(baseline - other).abs().max().item(),
                    )
                )

    return warnings


def _apply_unshard(
    params: UnshardParams,
    ordered_tensors: list[torch.Tensor],
    *,
    axis: object,
) -> torch.Tensor:
    if isinstance(params, PickParams):
        return ordered_tensors[0]
    if isinstance(params, ConcatParams):
        return torch.cat(ordered_tensors, dim=params.dim)
    # Phase 2: ReduceSumParams, CpZigzagParams
    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")
