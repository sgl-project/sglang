from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.unshard.executor import execute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.unshard.planner import compute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.types import Plan, UnshardPlan
from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def process_tensor_group(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
) -> ComparisonRecord | SkipRecord:
    b_tensors = _load_tensors(baseline_filenames, baseline_path)
    t_tensors = _load_tensors(target_filenames, target_path)

    b_plans, t_plans = _compute_plans(
        baseline_metas=[item.meta for item in b_tensors],
        target_metas=[item.meta for item in t_tensors],
    )

    b_extracted = _extract_tensors(b_tensors)
    t_extracted = _extract_tensors(t_tensors)
    del b_tensors, t_tensors

    b_tensor = _execute_plans(b_extracted, b_plans)
    t_tensor = _execute_plans(t_extracted, t_plans)

    if b_tensor is None or t_tensor is None:
        reason = "baseline_load_failed" if b_tensor is None else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    info = compare_tensors(
        x_baseline=b_tensor,
        x_target=t_tensor,
        name=name,
        diff_threshold=diff_threshold,
    )

    return ComparisonRecord(**info.model_dump())


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans(
    *,
    baseline_metas: list[dict[str, Any]],
    target_metas: list[dict[str, Any]],
) -> tuple[list[Plan], list[Plan]]:
    """This function deliberately takes metadata, since plan computation must never inspect actual tensor data."""
    return (
        _compute_plans_for_group(baseline_metas),
        _compute_plans_for_group(target_metas),
    )


def _compute_plans_for_group(metas: list[dict[str, Any]]) -> list[Plan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]
    plan = compute_unshard_plan(dim_specs=dim_specs, parallel_infos=parallel_infos)

    return [plan] if plan is not None else []


def _extract_tensors(
    loaded: list[ValueWithMeta],
) -> Optional[list[torch.Tensor]]:
    return [value for item in loaded if isinstance(value := item.value, torch.Tensor)]


def _execute_plans(
    tensors: list[torch.Tensor],
    plans: list[Plan],
) -> Optional[torch.Tensor]:
    if not tensors:
        return None

    if not plans:
        if len(tensors) != 1:
            return None
        return tensors[0]

    assert len(plans) <= 1, "multi-plan not supported yet"

    for plan in plans:
        if isinstance(plan, UnshardPlan):
            # TODO: incorrect `tensors_by_world_rank` if multi UnshardPlan
            tensors = execute_unshard_plan(
                plan, tensors_by_world_rank=dict(enumerate(tensors))
            )
        else:
            raise NotImplementedError(f"Unknown {plan=}")

    return tensors
