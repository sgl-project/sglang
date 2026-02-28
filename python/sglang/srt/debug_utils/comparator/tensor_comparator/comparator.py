from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DEFAULT_PERCENTILES,
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)
from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    argmax_coord,
    calc_rel_diff,
    compute_smaller_dtype,
    try_unify_shape,
)
from sglang.srt.debug_utils.dumper import get_truncated_value

QUANTILE_NUMEL_THRESHOLD = 10_000_000
SAMPLE_DIFF_THRESHOLD = 1e-3


def compare_tensor_pair(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    name: str = "",
    diff_threshold: float = 1e-3,
) -> TensorComparisonInfo:
    baseline_info = TensorInfo(
        shape=list(x_baseline.shape),
        dtype=str(x_baseline.dtype),
        stats=_compute_tensor_stats(x_baseline.float()),
    )
    target_info = TensorInfo(
        shape=list(x_target.shape),
        dtype=str(x_target.dtype),
        stats=_compute_tensor_stats(x_target.float()),
    )

    x_baseline = try_unify_shape(x_baseline, target_shape=x_target.shape)
    unified_shape = list(x_baseline.shape)

    baseline_original_dtype = x_baseline.dtype
    target_original_dtype = x_target.dtype

    x_baseline_f = x_baseline.float()
    x_target_f = x_target.float()

    shape_mismatch = x_baseline_f.shape != x_target_f.shape

    diff: Optional[DiffInfo] = None
    diff_downcast: Optional[DiffInfo] = None
    downcast_dtype: Optional[torch.dtype] = None

    if not shape_mismatch:
        diff = _compute_diff(
            x_baseline=x_baseline_f,
            x_target=x_target_f,
            diff_threshold=diff_threshold,
        )

        needs_sample = diff.max_abs_diff > SAMPLE_DIFF_THRESHOLD
        if needs_sample:
            baseline_info.sample = str(get_truncated_value(x_baseline_f))
            target_info.sample = str(get_truncated_value(x_target_f))

        if baseline_original_dtype != target_original_dtype:
            downcast_dtype = compute_smaller_dtype(
                Pair(x=baseline_original_dtype, y=target_original_dtype)
            )
            if downcast_dtype is not None:
                diff_downcast = _compute_diff(
                    x_baseline=x_baseline_f.to(downcast_dtype),
                    x_target=x_target_f.to(downcast_dtype),
                    diff_threshold=diff_threshold,
                )

    return TensorComparisonInfo(
        name=name,
        baseline=baseline_info,
        target=target_info,
        unified_shape=unified_shape,
        shape_mismatch=shape_mismatch,
        diff=diff,
        diff_downcast=diff_downcast,
        downcast_dtype=str(downcast_dtype) if downcast_dtype is not None else None,
    )


def _compute_tensor_stats(x: torch.Tensor) -> TensorStats:
    if x.numel() == 0:
        return TensorStats(
            mean=0.0,
            abs_mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            percentiles={},
        )

    include_quantiles: bool = x.numel() < QUANTILE_NUMEL_THRESHOLD
    return TensorStats(
        mean=torch.mean(x).item(),
        abs_mean=torch.mean(x.abs()).item(),
        std=torch.std(x).item(),
        min=torch.min(x).item(),
        max=torch.max(x).item(),
        percentiles=_compute_percentiles(x, include=include_quantiles),
    )


def _compute_percentiles(x: torch.Tensor, *, include: bool) -> dict[int, float]:
    if not include:
        return {}
    x_float: torch.Tensor = x.float()
    return {p: torch.quantile(x_float, p / 100.0).item() for p in DEFAULT_PERCENTILES}


def _compute_diff(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    diff_threshold: float = 1e-3,
) -> DiffInfo:
    if x_baseline.numel() == 0:
        return DiffInfo(
            rel_diff=0.0,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            abs_diff_percentiles={},
            max_diff_coord=[],
            baseline_at_max=0.0,
            target_at_max=0.0,
            diff_threshold=diff_threshold,
            passed=True,
        )

    raw_abs_diff = (x_target - x_baseline).abs()
    max_diff_coord = argmax_coord(raw_abs_diff)

    rel_diff = calc_rel_diff(x_target, x_baseline).item()
    max_abs_diff = raw_abs_diff.max().item()
    mean_abs_diff = raw_abs_diff.mean().item()

    include_quantiles: bool = raw_abs_diff.numel() < QUANTILE_NUMEL_THRESHOLD

    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        abs_diff_percentiles=_compute_percentiles(
            raw_abs_diff, include=include_quantiles
        ),
        max_diff_coord=list(max_diff_coord),
        baseline_at_max=x_baseline[max_diff_coord].item(),
        target_at_max=x_target[max_diff_coord].item(),
        diff_threshold=diff_threshold,
        passed=rel_diff <= diff_threshold,
    )
