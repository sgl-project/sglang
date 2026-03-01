from __future__ import annotations

from typing import Any, Optional

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    AxisAlignerPlan,
    compute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPerStepSubPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.planner import (
    compute_reorderer_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    _SingletonDimUtil,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_aligner_plan(
    *,
    metas_pair: Pair[list[dict[str, Any]]],
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
) -> AlignerPlan:
    dims_str_pair: Pair[Optional[str]] = metas_pair.map(
        lambda metas: metas[0].get("dims") if metas else None
    )
    axis_aligner_plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
        dims_str_pair=dims_str_pair
    )

    return AlignerPlan(
        per_step_plans=Pair(
            x=_compute_per_step_plans(
                metas=metas_pair.x,
                thd_seq_lens_by_step=thd_seq_lens_by_step_pair.x,
            ),
            y=_compute_per_step_plans(
                metas=metas_pair.y,
                thd_seq_lens_by_step=thd_seq_lens_by_step_pair.y,
            ),
        ),
        token_aligner_mode=token_aligner_mode,
        token_aligner_plan=token_aligner_plan,
        axis_aligner_plan=axis_aligner_plan,
    )


def _compute_per_step_plans(
    metas: list[dict[str, Any]],
    *,
    thd_seq_lens_by_step: Optional[dict[int, list[int]]] = None,
) -> list[AlignerPerStepPlan]:
    step_to_input_indices: dict[int, list[int]] = {}
    for i, meta in enumerate(metas):
        step: int = int(meta["step"])
        step_to_input_indices.setdefault(step, []).append(i)

    result: list[AlignerPerStepPlan] = []
    for step in sorted(step_to_input_indices):
        input_indices: list[int] = step_to_input_indices[step]
        step_metas: list[dict[str, Any]] = [metas[idx] for idx in input_indices]
        step_seq_lens: Optional[list[int]] = (
            thd_seq_lens_by_step.get(step) if thd_seq_lens_by_step is not None else None
        )
        plans: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=step_metas,
            thd_global_seq_lens=step_seq_lens,
        )
        result.append(
            AlignerPerStepPlan(
                step=step, input_object_indices=input_indices, sub_plans=plans
            )
        )

    return result


def compute_per_step_sub_plans(
    metas: list[dict[str, Any]],
    *,
    thd_global_seq_lens: Optional[list[int]] = None,
) -> list[AlignerPerStepSubPlan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs: list[DimSpec] = _SingletonDimUtil.filter_out(parse_dims(dims_str).dims)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]

    unsharder_plans = compute_unsharder_plan(
        dim_specs=dim_specs,
        parallel_infos=parallel_infos,
        thd_global_seq_lens=thd_global_seq_lens,
    )
    reorderer_plans = compute_reorderer_plans(
        dim_specs=dim_specs,
        parallel_infos=parallel_infos,
        thd_global_seq_lens=thd_global_seq_lens,
    )
    return [*unsharder_plans, *reorderer_plans]
