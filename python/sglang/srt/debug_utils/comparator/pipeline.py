from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    execute_reorderer_plan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.planner import (
    compute_reorderer_plans,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    AnyWarning,
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

Plan = Union[UnsharderPlan, ReordererPlan]


def process_tensor_group(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
) -> ComparisonRecord | SkipRecord:
    with warning_sink.context() as collected_warnings:
        return _process_tensor_group_raw(
            name=name,
            baseline_filenames=baseline_filenames,
            target_filenames=target_filenames,
            baseline_path=baseline_path,
            target_path=target_path,
            diff_threshold=diff_threshold,
            collected_warnings=collected_warnings,
        )


def _process_tensor_group_raw(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
    collected_warnings: list[AnyWarning],
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
        return SkipRecord(name=name, reason=reason, warnings=collected_warnings)

    info = compare_tensor_pair(
        x_baseline=b_tensor,
        x_target=t_tensor,
        name=name,
        diff_threshold=diff_threshold,
    )

    return ComparisonRecord(**info.model_dump(), warnings=collected_warnings)


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

    unsharder_plans = compute_unsharder_plan(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    reorderer_plans = compute_reorderer_plans(
        dim_specs=dim_specs, parallel_infos=parallel_infos
    )
    return [*unsharder_plans, *reorderer_plans]


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

    current = tensors
    for plan in plans:
        current = _execute_plan(current, plan)

    assert len(current) == 1
    return current[0]


def _execute_plan(
    tensors: list[torch.Tensor],
    plan: Plan,
) -> list[torch.Tensor]:
    if isinstance(plan, UnsharderPlan):
        return execute_unsharder_plan(plan, tensors)
    elif isinstance(plan, ReordererPlan):
        return execute_reorderer_plan(plan, tensors)
    else:
        raise NotImplementedError(f"Unknown {plan=}")
