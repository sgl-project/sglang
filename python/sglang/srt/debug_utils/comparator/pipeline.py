import argparse
from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.unshard.execute import execute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.unshard.plan import compute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.types import UnshardPlan
from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def process_tensor_group(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    baseline_loaded = _load_tensors(baseline_filenames, Path(args.baseline_path))
    target_loaded = _load_tensors(target_filenames, Path(args.target_path))

    baseline_plan, target_plan = _compute_plans(
        baseline_metas=[item.meta for item in baseline_loaded],
        target_metas=[item.meta for item in target_loaded],
    )

    baseline_tensor = _execute_plan(baseline_loaded, baseline_plan)
    target_tensor = _execute_plan(target_loaded, target_plan)

    if baseline_tensor is None or target_tensor is None:
        reason = "baseline_load_failed" if baseline_tensor is None else "target_load_failed"
        counts["skipped"] += 1
        print_record(SkipRecord(name=name, reason=reason), output_format=args.output_format)
        return

    info = compare_tensors(
        x_baseline=baseline_tensor,
        x_target=target_tensor,
        name=name,
        diff_threshold=args.diff_threshold,
    )

    counts["passed" if info.diff is not None and info.diff.passed else "failed"] += 1
    print_record(
        ComparisonRecord(**info.model_dump()),
        output_format=args.output_format,
    )


def _load_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [ValueWithMeta.load(base_path / f) for f in filenames]


def _compute_plans(
    *,
    baseline_metas: list[dict[str, Any]],
    target_metas: list[dict[str, Any]],
) -> tuple[Optional[UnshardPlan], Optional[UnshardPlan]]:
    """Compute plans for both sides from metadata only.

    This function deliberately takes metadata dicts, not tensors or
    ValueWithMeta objects — plan computation is a pure metadata operation
    and must never inspect actual tensor data.

    Currently only produces unshard plans; future pipeline components
    (e.g. reduction, reordering) will also contribute to the plan.
    """
    return (
        _compute_single_plan(baseline_metas),
        _compute_single_plan(target_metas),
    )


def _compute_single_plan(metas: list[dict[str, Any]]) -> Optional[UnshardPlan]:
    if not metas or len(metas) == 1:
        return None

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return None

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]
    return compute_unshard_plan(dim_specs=dim_specs, parallel_infos=parallel_infos)


def _execute_plan(
    loaded: list[ValueWithMeta],
    plan: Optional[UnshardPlan],
) -> Optional[torch.Tensor]:
    if not loaded:
        return None

    if plan is None:
        if len(loaded) != 1:
            return None
        value = loaded[0].value
        if not isinstance(value, torch.Tensor):
            return None
        return value

    tensors_by_index: dict[int, torch.Tensor] = {}
    for i, item in enumerate(loaded):
        if not isinstance(item.value, torch.Tensor):
            return None
        tensors_by_index[i] = item.value

    return execute_unshard_plan(plan, tensors_by_index)
