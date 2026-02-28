from sglang.srt.debug_utils.comparator.tensor_comparator.types import (
    DiffInfo,
    TensorComparisonInfo,
    TensorStats,
)


def format_comparison(info: TensorComparisonInfo) -> str:
    lines: list[str] = []
    baseline = info.baseline
    target = info.target

    dtype_marker = "" if baseline.dtype == target.dtype else "🟠"
    lines.append(
        f"Raw "
        f"[shape] {baseline.shape} vs {target.shape}\t"
        f"[{dtype_marker}dtype] {baseline.dtype} vs {target.dtype}"
    )

    if info.unified_shape != baseline.shape:
        lines.append(
            f"Unify shape: {baseline.shape} -> {info.unified_shape} "
            f"(to match {target.shape})"
        )

    lines.append(
        f"After unify "
        f"[shape] {info.unified_shape} vs {target.shape}\t"
        f"[dtype] {baseline.dtype} vs {target.dtype}"
    )

    lines.extend(_format_stats_comparison(baseline=baseline.stats, target=target.stats))

    if info.shape_mismatch:
        lines.append("⚠️ Shape mismatch")
        return "\n".join(lines)

    if info.diff is not None:
        lines.extend(_format_diff(diff=info.diff))

    if info.diff_downcast is not None and info.downcast_dtype is not None:
        lines.extend(
            _format_diff(
                diff=info.diff_downcast,
                prefix_text=f"When downcast to {info.downcast_dtype}: ",
            )
        )

    if baseline.sample is not None:
        lines.append(f"x_baseline(sample)={baseline.sample}")
    if target.sample is not None:
        lines.append(f"x_target(sample)={target.sample}")

    return "\n".join(lines)


def _format_stats_comparison(baseline: TensorStats, target: TensorStats) -> list[str]:
    lines: list[str] = []

    for stat_name in TensorStats.model_fields:
        if stat_name == "percentiles":
            continue
        value_baseline: float = getattr(baseline, stat_name)
        value_target: float = getattr(target, stat_name)
        lines.append(
            f"[{stat_name}] {value_baseline:.4f} vs {value_target:.4f} "
            f"(diff: {value_target - value_baseline:.4f})"
        )

    for p in sorted(set(baseline.percentiles) & set(target.percentiles)):
        value_baseline = baseline.percentiles[p]
        value_target = target.percentiles[p]
        lines.append(
            f"[p{p}] {value_baseline:.4f} vs {value_target:.4f} "
            f"(diff: {value_target - value_baseline:.4f})"
        )

    return lines


def _format_diff(diff: DiffInfo, prefix_text: str = "") -> list[str]:
    rel_diff_marker: str = "❌" if diff.rel_diff > diff.diff_threshold else "✅"
    lines: list[str] = [
        prefix_text
        + f"{rel_diff_marker} rel_diff={diff.rel_diff}\t"
        + f"max_abs_diff={diff.max_abs_diff}\t"
        + f"mean_abs_diff={diff.mean_abs_diff}",
        f"max_abs_diff happens at coord={diff.max_diff_coord} with "
        f"baseline={diff.baseline_at_max} "
        f"target={diff.target_at_max}",
    ]

    if diff.abs_diff_percentiles:
        quantile_parts: list[str] = [
            f"p{p}={value:.4f}"
            for p, value in sorted(diff.abs_diff_percentiles.items())
        ]
        lines.append("[abs_diff] " + " ".join(quantile_parts))

    return lines
