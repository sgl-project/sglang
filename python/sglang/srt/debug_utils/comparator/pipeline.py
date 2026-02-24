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
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    b_tensors = _load_tensors(baseline_filenames, Path(args.baseline_path))
    t_tensors = _load_tensors(target_filenames, Path(args.target_path))

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
        counts["skipped"] += 1
        print_record(
            SkipRecord(name=name, reason=reason), output_format=args.output_format
        )
        return

    info = compare_tensors(
        x_baseline=b_tensor,
        x_target=t_tensor,
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
) -> tuple[list[Plan], list[Plan]]:
    """This function deliberately takes metadata, since plan computation must never inspect actual tensor data."""
    return (
        _compute_single_plan(baseline_metas),
        _compute_single_plan(target_metas),
    )


def _compute_single_plan(metas: list[dict[str, Any]]) -> list[Plan]:
    if not metas or len(metas) == 1:
        return []

    dims_str = metas[0].get("dims")
    if dims_str is None:
        return []

    dim_specs = parse_dims(dims_str)
    parallel_infos = [normalize_parallel_info(meta) for meta in metas]
    plan = compute_unshard_plan(dim_specs=dim_specs, parallel_infos=parallel_infos)

    if plan is None:
        return []
    return [plan]


def _extract_tensors(
    loaded: list[ValueWithMeta],
) -> Optional[list[torch.Tensor]]:
    if not loaded:
        return None

    tensors: list[torch.Tensor] = []
    for item in loaded:
        if not isinstance(item.value, torch.Tensor):
            return None
        tensors.append(item.value)

    return tensors


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

    unshard_plans = [p for p in plans if isinstance(p, UnshardPlan)]
    assert (
        len(unshard_plans) <= 1
    ), f"Expected at most 1 unshard plan, got {len(unshard_plans)}"

    tensors_by_world_rank = dict(enumerate(tensors))
    return execute_unshard_plan(unshard_plans[0], tensors_by_world_rank)
