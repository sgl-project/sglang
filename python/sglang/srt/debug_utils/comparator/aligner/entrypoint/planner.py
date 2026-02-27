from __future__ import annotations

from typing import Any, Optional

from sglang.srt.debug_utils.comparator.aligner.axis_swapper import (
    AxisSwapperPlan,
    compute_axis_swapper_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPerStepSubPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.planner import (
    compute_reorderer_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.dims import (
    TOKEN_DIM_NAME,
    find_dim_index,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.utils import Pair


def compute_aligner_plan(
    *,
    metas_pair: Pair[list[dict[str, Any]]],
    token_aligner_plan: Optional[TokenAlignerPlan],
) -> AlignerPlan:
    dims_str_pair: Pair[Optional[str]] = metas_pair.map(
        lambda metas: metas[0].get("dims") if metas else None
    )
    axis_swapper_plan: Optional[AxisSwapperPlan] = compute_axis_swapper_plan(
        dims_str_pair=dims_str_pair
    )

    return AlignerPlan(
        per_step_plans=metas_pair.map(
            lambda metas: _compute_per_step_plans(metas=metas)
        ),
        token_aligner_plan=token_aligner_plan,
        axis_swapper_plan=axis_swapper_plan,
    )


def _compute_token_dim(metas: list[dict[str, Any]]) -> int:
    fallback_dim = 0

    if not metas:
        return fallback_dim

    dims_str: Optional[str] = metas[0].get("dims")
    if dims_str is None:
        return fallback_dim

    idx: Optional[int] = find_dim_index(parse_dims(dims_str), TOKEN_DIM_NAME)
    if idx is None:
        return fallback_dim

    return idx


def _compute_per_step_plans(metas: list[dict[str, Any]]) -> list[AlignerPerStepPlan]:
    step_to_input_indices: dict[int, list[int]] = {}
    for i, meta in enumerate(metas):
        step: int = int(meta["step"])
        step_to_input_indices.setdefault(step, []).append(i)

    result: list[AlignerPerStepPlan] = []
    for step in sorted(step_to_input_indices):
        input_indices: list[int] = step_to_input_indices[step]
        step_metas: list[dict[str, Any]] = [metas[idx] for idx in input_indices]
        plans: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=step_metas
        )
        result.append(
            AlignerPerStepPlan(
                step=step, input_object_indices=input_indices, sub_plans=plans
            )
        )

    return result


def compute_per_step_sub_plans(
    metas: list[dict[str, Any]],
) -> list[AlignerPerStepSubPlan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]

    unsharder_plans = compute_unsharder_plan(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    reorderer_plans = compute_reorderer_plans(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    return [*unsharder_plans, *reorderer_plans]
