from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    execute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPerStepSubPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    execute_reorderer_plan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.utils import Pair


@dataclass(frozen=True)
class AlignerResult:
    tensors: Optional[Pair[torch.Tensor]]
    failed_side_xy: Optional[str]  # "x" or "y"; None if success


def execute_aligner_plan(
    *,
    tensors_pair: Pair[list[torch.Tensor]],
    plan: AlignerPlan,
) -> AlignerResult:
    """Execute unified unshard/reorder + token-align."""

    # Per-side: unshard + reorder -> dict[step, tensor]
    step_tensors_x: dict[int, torch.Tensor] = _execute_step_plans(
        tensors=tensors_pair.x, step_plans=plan.per_step_plans.x
    )
    step_tensors_y: dict[int, torch.Tensor] = _execute_step_plans(
        tensors=tensors_pair.y, step_plans=plan.per_step_plans.y
    )

    if not step_tensors_x or not step_tensors_y:
        failed_side_xy: str = "x" if not step_tensors_x else "y"
        return AlignerResult(tensors=None, failed_side_xy=failed_side_xy)

    # Cross-side: token alignment (or direct extraction for single-step)
    if plan.token_aligner_plan is not None:
        combined: Pair[torch.Tensor] = execute_token_aligner(
            plan=plan.token_aligner_plan,
            tensor_of_step_pair=Pair(x=step_tensors_x, y=step_tensors_y),
        )
    else:
        assert len(step_tensors_x) == 1 and len(step_tensors_y) == 1
        combined = Pair(
            x=list(step_tensors_x.values())[0],
            y=list(step_tensors_y.values())[0],
        )

    # Cross-side: axis alignment (squeeze singletons + rearrange dim order)
    if (aligner_plan := plan.axis_aligner_plan) is not None:
        combined = Pair(
            x=execute_axis_aligner_plan(tensor=combined.x, plan=aligner_plan, side="x"),
            y=execute_axis_aligner_plan(tensor=combined.y, plan=aligner_plan, side="y"),
        )

    return AlignerResult(tensors=combined, failed_side_xy=None)


def _execute_step_plans(
    tensors: list[torch.Tensor],
    step_plans: list[AlignerPerStepPlan],
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}

    for step_plan in step_plans:
        step_tensors: list[torch.Tensor] = [
            tensors[i] for i in step_plan.input_object_indices
        ]
        tensor: Optional[torch.Tensor] = execute_sub_plans(
            tensors=step_tensors, plans=step_plan.sub_plans
        )
        if tensor is not None:
            result[step_plan.step] = tensor

    return result


def execute_sub_plans(
    tensors: list[torch.Tensor],
    plans: list[AlignerPerStepSubPlan],
) -> Optional[torch.Tensor]:
    if not tensors:
        return None

    if not plans:
        if len(tensors) != 1:
            return None
        return tensors[0]

    current = tensors
    for plan in plans:
        current = execute_sub_plan(tensors=current, plan=plan)

    assert len(current) == 1
    return current[0]


def execute_sub_plan(
    tensors: list[torch.Tensor],
    plan: AlignerPerStepSubPlan,
) -> list[torch.Tensor]:
    if isinstance(plan, UnsharderPlan):
        return execute_unsharder_plan(plan, tensors)
    elif isinstance(plan, ReordererPlan):
        return execute_reorderer_plan(plan, tensors)
    else:
        raise NotImplementedError(f"Unknown {plan=}")
