from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DEFAULT_PERCENTILES,
    DiffInfo,
    TensorComparisonInfo,
    TensorInfo,
    TensorStats,
)
from sglang.srt.debug_utils.comparator.threshold_dsl import (
    DiffThresholdRule,
    evaluate_predicate,
    parse_predicate,
    resolve_predicate,
)
from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    argmax_coord,
    calc_per_token_rel_diff,
    calc_rel_diff,
    compute_smaller_dtype,
    try_unify_shape,
)
from sglang.srt.debug_utils.dumper import get_truncated_value

QUANTILE_NUMEL_THRESHOLD = 10_000_000
SAMPLE_DIFF_THRESHOLD = 1e-3
DEFAULT_PREDICATE: str = "rel <= 0.001"


@dataclass
class FailureDisplayBudget:
    max_detail: int = 50
    num_emitted: int = 0

    def take(self) -> bool:
        if self.max_detail < 0:
            return True
        if self.num_emitted >= self.max_detail:
            return False
        self.num_emitted += 1
        return True


def compute_tensor_info(
    tensor: torch.Tensor,
    *,
    include_sample: bool = False,
    include_percentiles: bool = True,
) -> TensorInfo:
    """Compute TensorInfo (shape, dtype, stats, optional sample) for a single tensor."""
    tensor_f = tensor.float()
    return _compute_tensor_info_from_float(
        tensor_f,
        original_shape=list(tensor.shape),
        original_dtype=str(tensor.dtype),
        include_sample=include_sample,
        include_percentiles=include_percentiles,
    )


def _compute_tensor_info_from_float(
    tensor_f: torch.Tensor,
    *,
    original_shape: list[int],
    original_dtype: str,
    include_sample: bool = False,
    include_percentiles: bool = True,
) -> TensorInfo:
    stats: TensorStats = _compute_tensor_stats(
        tensor_f, include_percentiles=include_percentiles
    )
    sample: Optional[str] = (
        str(get_truncated_value(tensor_f)) if include_sample else None
    )
    return TensorInfo(
        shape=original_shape,
        dtype=original_dtype,
        stats=stats,
        sample=sample,
    )


def compare_tensor_pair(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    name: str = "",
    diff_threshold_rules: Optional[list[DiffThresholdRule]] = None,
    seq_dim: Optional[int] = None,
    failure_display_budget: Optional[FailureDisplayBudget] = None,
) -> TensorComparisonInfo:
    predicate = resolve_predicate(
        name, diff_threshold_rules, default_predicate=DEFAULT_PREDICATE
    )

    x_baseline_original = x_baseline
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
    raw_abs_diff: Optional[torch.Tensor] = None

    if not shape_mismatch:
        raw_abs_diff = (x_target_f - x_baseline_f).abs()
        diff = compute_diff(
            x_baseline=x_baseline_f,
            x_target=x_target_f,
            predicate=predicate,
            seq_dim=seq_dim,
            include_percentiles=False,
            _raw_abs_diff=raw_abs_diff,
        )

    is_failure = shape_mismatch or (diff is not None and not diff.passed)
    needs_detail = is_failure and (
        failure_display_budget is None or failure_display_budget.take()
    )

    baseline_info: TensorInfo = _compute_tensor_info_from_float(
        x_baseline_f,
        original_shape=list(x_baseline_original.shape),
        original_dtype=str(x_baseline_original.dtype),
        include_percentiles=needs_detail,
    )
    target_info: TensorInfo = _compute_tensor_info_from_float(
        x_target_f,
        original_shape=list(x_target.shape),
        original_dtype=str(x_target.dtype),
        include_percentiles=needs_detail,
    )

    if diff is not None and raw_abs_diff is not None and needs_detail:
        diff.abs_diff_percentiles = _compute_percentiles(
            raw_abs_diff,
            include=raw_abs_diff.numel() < QUANTILE_NUMEL_THRESHOLD,
        )

    if diff is not None:
        needs_sample = diff.max_abs_diff > SAMPLE_DIFF_THRESHOLD
        if needs_sample:
            baseline_info.sample = str(get_truncated_value(x_baseline_f))
            target_info.sample = str(get_truncated_value(x_target_f))

        if baseline_original_dtype != target_original_dtype:
            downcast_dtype = compute_smaller_dtype(
                Pair(x=baseline_original_dtype, y=target_original_dtype)
            )
            if downcast_dtype is not None:
                diff_downcast = compute_diff(
                    x_baseline=x_baseline_f.to(downcast_dtype),
                    x_target=x_target_f.to(downcast_dtype),
                    predicate=predicate,
                    seq_dim=seq_dim,
                    include_percentiles=needs_detail,
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


def _compute_tensor_stats(
    x: torch.Tensor, *, include_percentiles: bool = True
) -> TensorStats:
    if x.numel() == 0:
        return TensorStats(
            mean=0.0,
            abs_mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            percentiles={},
        )

    include_quantiles: bool = (
        include_percentiles and x.numel() < QUANTILE_NUMEL_THRESHOLD
    )
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
    import numpy as np

    arr = x.detach().float().cpu().numpy().ravel()
    values = np.percentile(arr, list(DEFAULT_PERCENTILES))
    return {p: float(v) for p, v in zip(DEFAULT_PERCENTILES, values)}


def compute_diff(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    predicate: str = DEFAULT_PREDICATE,
    seq_dim: Optional[int] = None,
    include_percentiles: bool = True,
    *,
    _raw_abs_diff: Optional[torch.Tensor] = None,
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
            predicate=predicate,
            passed=True,
        )

    raw_abs_diff = (
        _raw_abs_diff
        if _raw_abs_diff is not None
        else (x_target.float() - x_baseline.float()).abs()
    )
    max_diff_coord = argmax_coord(raw_abs_diff)

    max_abs_diff = raw_abs_diff.max().item()
    rel_diff = (
        0.0 if max_abs_diff == 0.0 else calc_rel_diff(x_target, x_baseline).item()
    )
    mean_abs_diff = raw_abs_diff.mean().item()

    include_quantiles: bool = (
        include_percentiles and raw_abs_diff.numel() < QUANTILE_NUMEL_THRESHOLD
    )

    per_token_rel_diff: Optional[list[float]] = None
    if seq_dim is not None and x_baseline.dim() > seq_dim:
        per_token_rel_diff = calc_per_token_rel_diff(
            x_baseline, x_target, seq_dim=seq_dim
        ).tolist()

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
        predicate=predicate,
        passed=evaluate_predicate(
            parse_predicate(predicate),
            rel=rel_diff,
            max_abs=max_abs_diff,
            mean_abs=mean_abs_diff,
        ),
        per_token_rel_diff=per_token_rel_diff,
    )
