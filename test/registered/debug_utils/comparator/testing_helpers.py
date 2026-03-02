"""Shared test helpers for comparator tests."""

from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorInfo,
    TensorStats,
)

DEFAULT_PERCENTILES: dict[int, float] = {
    1: -1.8,
    5: -1.5,
    50: 0.0,
    95: 1.5,
    99: 1.8,
}

DEFAULT_ABS_DIFF_PERCENTILES: dict[int, float] = {
    1: 0.0001,
    5: 0.0001,
    50: 0.0002,
    95: 0.0004,
    99: 0.0005,
}


def make_stats(
    mean: float = 0.0,
    abs_mean: float = 0.8,
    std: float = 1.0,
    min: float = -2.0,
    max: float = 2.0,
    percentiles: Optional[dict[int, float]] = None,
) -> TensorStats:
    return TensorStats(
        mean=mean,
        abs_mean=abs_mean,
        std=std,
        min=min,
        max=max,
        percentiles=percentiles if percentiles is not None else DEFAULT_PERCENTILES,
    )


def make_diff(
    rel_diff: float = 0.0001,
    max_abs_diff: float = 0.0005,
    mean_abs_diff: float = 0.0002,
    abs_diff_percentiles: Optional[dict[int, float]] = None,
    diff_threshold: float = 1e-3,
    passed: bool = True,
) -> DiffInfo:
    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        abs_diff_percentiles=(
            abs_diff_percentiles
            if abs_diff_percentiles is not None
            else DEFAULT_ABS_DIFF_PERCENTILES
        ),
        max_diff_coord=[2, 3],
        baseline_at_max=1.0,
        target_at_max=1.0005,
        diff_threshold=diff_threshold,
        passed=passed,
    )


def make_tensor_info(
    shape: Optional[list[int]] = None,
    dtype: str = "torch.float32",
    stats: Optional[TensorStats] = None,
    sample: Optional[str] = None,
) -> TensorInfo:
    return TensorInfo(
        shape=shape if shape is not None else [4, 8],
        dtype=dtype,
        stats=stats if stats is not None else make_stats(),
        sample=sample,
    )
