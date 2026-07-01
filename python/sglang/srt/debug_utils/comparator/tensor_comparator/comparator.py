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
    DEFAULT_PREDICATE,
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

# Diagnostic detail (percentiles, samples) is computed only for the first
# ``MAX_FAILURE_DETAIL`` failing tensors. Percentiles dominate per-tensor cost
# (one numpy.percentile call is ~10-40x the rel_diff reduction) and are purely
# diagnostic — they never affect the pass/fail predicate. When most tensors fail
# (e.g. a systematic divergence), this keeps the run fast while still giving
# enough failing examples to debug from.
# Set to a negative value to disable the cap (always emit full detail).
MAX_FAILURE_DETAIL = 50
_failure_detail_emitted = 0


def reset_failure_detail_budget() -> None:
    """Reset the per-run failing-tensor detail counter (call before a fresh run)."""
    global _failure_detail_emitted
    _failure_detail_emitted = 0


def _take_failure_detail_budget() -> bool:
    """Return True (and consume one unit) if full detail should be emitted for
    the current failing tensor; False once the cap is reached."""
    global _failure_detail_emitted
    if MAX_FAILURE_DETAIL < 0:
        return True
    if _failure_detail_emitted >= MAX_FAILURE_DETAIL:
        return False
    _failure_detail_emitted += 1
    return True


def compute_tensor_info(
    tensor: torch.Tensor,
    *,
    include_sample: bool = False,
    include_percentiles: bool = True,
) -> TensorInfo:
    """Compute TensorInfo (shape, dtype, stats, optional sample) for a single tensor."""
    stats: TensorStats = _compute_tensor_stats(
        tensor.float(), include_percentiles=include_percentiles
    )
    sample: Optional[str] = (
        str(get_truncated_value(tensor.float())) if include_sample else None
    )
    return TensorInfo(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        stats=stats,
        sample=sample,
    )


def compare_tensor_pair(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    name: str = "",
    diff_threshold_rules: Optional[list[DiffThresholdRule]] = None,
    seq_dim: Optional[int] = None,
) -> TensorComparisonInfo:
    predicate = resolve_predicate(name, diff_threshold_rules)

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

    # Pass 1: compute the diff WITHOUT percentiles. Percentiles are diagnostic
    # detail only (consumed by the formatter, never by the pass/fail predicate),
    # and they dominate per-tensor cost (one numpy.percentile call is ~10-40x the
    # cost of the rel_diff reduction). So compute them lazily — only when the
    # comparison fails / shape-mismatches.
    if not shape_mismatch:
        diff = compute_diff(
            x_baseline=x_baseline_f,
            x_target=x_target_f,
            predicate=predicate,
            seq_dim=seq_dim,
            include_percentiles=False,
        )

    is_failure = shape_mismatch or (diff is not None and not diff.passed)
    # Emit expensive diagnostic detail only for the first MAX_FAILURE_DETAIL
    # failures (pass/fail itself is already decided above and unaffected).
    needs_detail = is_failure and _take_failure_detail_budget()

    baseline_info: TensorInfo = compute_tensor_info(
        x_baseline_original, include_percentiles=needs_detail
    )
    target_info: TensorInfo = compute_tensor_info(
        x_target, include_percentiles=needs_detail
    )

    if not shape_mismatch and needs_detail:
        # Recompute the diff with percentiles for the failing-tensor report.
        diff = compute_diff(
            x_baseline=x_baseline_f,
            x_target=x_target_f,
            predicate=predicate,
            seq_dim=seq_dim,
            include_percentiles=True,
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

    arr = x.detach().float().numpy().ravel()
    values = np.percentile(arr, list(DEFAULT_PERCENTILES))
    return {p: float(v) for p, v in zip(DEFAULT_PERCENTILES, values)}


def compute_diff(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    predicate: str = DEFAULT_PREDICATE,
    seq_dim: Optional[int] = None,
    include_percentiles: bool = True,
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

    raw_abs_diff = (x_target - x_baseline).abs()
    max_diff_coord = argmax_coord(raw_abs_diff)

    max_abs_diff = raw_abs_diff.max().item()
    # Bitwise-identical tensors (e.g. all-zero starved-MoE-expert grads) have an undefined
    # relative diff (0/0 -> NaN), which would spuriously fail a "rel <= 0" predicate. A zero
    # absolute diff means the tensors are identical, so their relative diff is zero.
    rel_diff = 0.0 if max_abs_diff == 0.0 else calc_rel_diff(x_target, x_baseline).item()
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
