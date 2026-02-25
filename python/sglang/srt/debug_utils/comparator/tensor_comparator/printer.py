from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorStats,
)


def print_comparison(info: TensorComparisonInfo, diff_threshold: float) -> None:
    baseline = info.baseline
    target = info.target

    dtype_marker = "" if baseline.dtype == target.dtype else "🟠"
    print(
        f"Raw "
        f"[shape] {baseline.shape} vs {target.shape}\t"
        f"[{dtype_marker}dtype] {baseline.dtype} vs {target.dtype}"
    )

    if info.unified_shape != baseline.shape:
        print(
            f"Unify shape: {baseline.shape} -> {info.unified_shape} "
            f"(to match {target.shape})"
        )

    print(
        f"After unify "
        f"[shape] {info.unified_shape} vs {target.shape}\t"
        f"[dtype] {baseline.dtype} vs {target.dtype}"
    )

    _print_stats_comparison(baseline=baseline.stats, target=target.stats)

    if info.shape_mismatch:
        print("⚠️ Shape mismatch")
        return

    if info.diff is not None:
        _print_diff(
            diff=info.diff,
            diff_threshold=diff_threshold,
        )

    if info.diff_downcast is not None and info.downcast_dtype is not None:
        _print_diff(
            diff=info.diff_downcast,
            diff_threshold=diff_threshold,
            prefix_text=f"When downcast to {info.downcast_dtype}: ",
        )

    if baseline.sample is not None:
        print(f"x_baseline(sample)={baseline.sample}")
    if target.sample is not None:
        print(f"x_target(sample)={target.sample}")


def _print_stats_comparison(baseline: TensorStats, target: TensorStats) -> None:
    stat_names = list(TensorStats.model_fields.keys())
    for stat_name in stat_names:
        value_baseline = getattr(baseline, stat_name)
        value_target = getattr(target, stat_name)
        if value_baseline is None or value_target is None:
            continue
        print(
            f"[{stat_name}] {value_baseline:.4f} vs {value_target:.4f} "
            f"(diff: {value_target - value_baseline:.4f})"
        )


def _print_diff(diff: DiffInfo, diff_threshold: float, prefix_text: str = "") -> None:
    print(
        prefix_text
        + "\t".join(
            f"{'❌' if value > diff_threshold else '✅'} {name}={value}"
            for name, value in [
                ("rel_diff", diff.rel_diff),
                ("max_abs_diff", diff.max_abs_diff),
                ("mean_abs_diff", diff.mean_abs_diff),
            ]
        )
    )

    print(
        f"max_abs_diff happens at coord={diff.max_diff_coord} with "
        f"baseline={diff.baseline_at_max} "
        f"target={diff.target_at_max}"
    )
