from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    execute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.traced_types import (
    TracedAlignerPlan,
    TracedSidePlan,
    TracedStepPlan,
    TracedSubPlan,
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
from sglang.srt.debug_utils.comparator.aligner.token_aligner.concat_steps import (
    execute_token_aligner_concat_steps,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.executor import (
    execute_token_aligner,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    UnsharderResult,
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.output_types import (
    ReplicatedCheckResult,
    ShapeSnapshot,
)
from sglang.srt.debug_utils.comparator.utils import Pair


class StepPlansResult(NamedTuple):
    tensors: dict[int, torch.Tensor]
    checks: list[ReplicatedCheckResult]
    traced_side: TracedSidePlan


class SubPlansResult(NamedTuple):
    tensor: Optional[torch.Tensor]
    checks: list[ReplicatedCheckResult]
    snapshots: list[ShapeSnapshot]


@dataclass(frozen=True)
class AlignerResult:
    tensors: Optional[Pair[torch.Tensor]]
    failed_side_xy: Optional[str]  # "x" or "y"; None if success
    replicated_checks: list[ReplicatedCheckResult] = field(default_factory=list)
    traced_plan: Optional[TracedAlignerPlan] = None


def execute_aligner_plan(
    *,
    tensors_pair: Pair[list[torch.Tensor]],
    plan: AlignerPlan,
) -> AlignerResult:
    """Execute unified unshard/reorder + token-align."""
    all_checks: list[ReplicatedCheckResult] = []

    # Per-side: unshard + reorder -> dict[step, tensor]
    result_x: StepPlansResult = _execute_step_plans(
        tensors=tensors_pair.x, step_plans=plan.per_step_plans.x
    )
    all_checks.extend(result_x.checks)

    result_y: StepPlansResult = _execute_step_plans(
        tensors=tensors_pair.y, step_plans=plan.per_step_plans.y
    )
    all_checks.extend(result_y.checks)

    traced_plan: TracedAlignerPlan = TracedAlignerPlan(
        plan=plan,
        per_side=Pair(x=result_x.traced_side, y=result_y.traced_side),
    )

    if not result_x.tensors or not result_y.tensors:
        failed_side_xy: str = "x" if not result_x.tensors else "y"
        return AlignerResult(
            tensors=None,
            failed_side_xy=failed_side_xy,
            replicated_checks=all_checks,
            traced_plan=traced_plan,
        )

    # Cross-side: token alignment (or direct extraction for single-step)
    step_pair: Pair[dict[int, torch.Tensor]] = Pair(
        x=result_x.tensors, y=result_y.tensors
    )
    combined: Pair[torch.Tensor]
    if plan.token_aligner_mode == "concat_steps":
        combined = execute_token_aligner_concat_steps(tensor_of_step_pair=step_pair)
    elif plan.token_aligner_mode == "smart":
        assert plan.token_aligner_plan is not None
        combined = execute_token_aligner(
            plan=plan.token_aligner_plan,
            tensor_of_step_pair=step_pair,
        )
    else:
        assert len(result_x.tensors) == 1 and len(result_y.tensors) == 1
        combined = Pair(
            x=list(result_x.tensors.values())[0],
            y=list(result_y.tensors.values())[0],
        )

    # Cross-side: axis alignment (squeeze singletons + rearrange dim order)
    if (aligner_plan := plan.axis_aligner_plan) is not None:
        combined = Pair(
            x=execute_axis_aligner_plan(tensor=combined.x, plan=aligner_plan, side="x"),
            y=execute_axis_aligner_plan(tensor=combined.y, plan=aligner_plan, side="y"),
        )

    return AlignerResult(
        tensors=combined,
        failed_side_xy=None,
        replicated_checks=all_checks,
        traced_plan=traced_plan,
    )


def _execute_step_plans(
    tensors: list[torch.Tensor],
    step_plans: list[AlignerPerStepPlan],
) -> StepPlansResult:
    result: dict[int, torch.Tensor] = {}
    all_checks: list[ReplicatedCheckResult] = []
    traced_steps: list[TracedStepPlan] = []

    for step_plan in step_plans:
        step_tensors: list[torch.Tensor] = [
            tensors[i] for i in step_plan.input_object_indices
        ]
        sub_result: SubPlansResult = execute_sub_plans(
            tensors=step_tensors, plans=step_plan.sub_plans
        )
        all_checks.extend(sub_result.checks)

        traced_subs: list[TracedSubPlan] = [
            TracedSubPlan(plan=sub_plan, snapshot=snapshot)
            for sub_plan, snapshot in zip(step_plan.sub_plans, sub_result.snapshots)
        ]
        traced_steps.append(
            TracedStepPlan(
                step=step_plan.step,
                input_object_indices=step_plan.input_object_indices,
                sub_plans=traced_subs,
            )
        )

        if sub_result.tensor is not None:
            result[step_plan.step] = sub_result.tensor

    return StepPlansResult(
        tensors=result,
        checks=all_checks,
        traced_side=TracedSidePlan(step_plans=traced_steps),
    )


def execute_sub_plans(
    tensors: list[torch.Tensor],
    plans: list[AlignerPerStepSubPlan],
) -> SubPlansResult:
    if not tensors:
        return SubPlansResult(tensor=None, checks=[], snapshots=[])

    if not plans:
        if len(tensors) != 1:
            return SubPlansResult(tensor=None, checks=[], snapshots=[])
        return SubPlansResult(tensor=tensors[0], checks=[], snapshots=[])

    current: list[torch.Tensor] = tensors
    all_checks: list[ReplicatedCheckResult] = []
    all_snapshots: list[ShapeSnapshot] = []
    for plan in plans:
        input_shapes: list[list[int]] = [list(t.shape) for t in current]
        current, checks = execute_sub_plan(tensors=current, plan=plan)
        output_shapes: list[list[int]] = [list(t.shape) for t in current]
        all_checks.extend(checks)
        all_snapshots.append(
            ShapeSnapshot(
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
        )

    assert len(current) == 1
    return SubPlansResult(tensor=current[0], checks=all_checks, snapshots=all_snapshots)


def execute_sub_plan(
    tensors: list[torch.Tensor],
    plan: AlignerPerStepSubPlan,
) -> tuple[list[torch.Tensor], list[ReplicatedCheckResult]]:
    if isinstance(plan, UnsharderPlan):
        unsharder_result: UnsharderResult = execute_unsharder_plan(plan, tensors)
        return unsharder_result.tensors, unsharder_result.replicated_checks
    elif isinstance(plan, ReordererPlan):
        return execute_reorderer_plan(plan, tensors), []
    else:
        raise NotImplementedError(f"Unknown {plan=}")
