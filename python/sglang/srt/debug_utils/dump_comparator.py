import argparse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from sglang.srt.debug_utils.dumper import get_truncated_value

# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class TensorStats:
    mean: float
    std: float
    min: float
    max: float
    p1: Optional[float]
    p5: Optional[float]
    p95: Optional[float]
    p99: Optional[float]


@dataclass
class TensorInfo:
    shape: Tuple[int, ...]
    dtype: str
    stats: Optional[TensorStats]
    sample: Any


@dataclass
class DiffInfo:
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    max_diff_coord: Tuple[int, ...]
    baseline_at_max: float
    target_at_max: float


@dataclass
class TensorComparisonInfo:
    name: Optional[str]
    baseline: Optional[TensorInfo]
    target: Optional[TensorInfo]
    baseline_unified_shape: Optional[Tuple[int, ...]]
    shape_mismatch: bool
    diff: Optional[DiffInfo]
    diff_downcast: Optional[DiffInfo]
    downcast_dtype: Optional[str]


# ─── Public API ──────────────────────────────────────────────────────────────


def main(args):
    import polars as pl

    from sglang.srt.debug_utils.dump_loader import find_row, read_meta

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_id) & (pl.col("step") <= args.end_id)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    df_baseline = read_meta(args.baseline_path)
    print("df_target", df_target)
    print("df_baseline", df_baseline)

    for row in df_target.iter_rows(named=True):
        path_target = Path(args.target_path) / row["filename"]
        baseline_step = row["step"] - args.start_id + args.baseline_start_id

        row_baseline = find_row(
            df_baseline,
            conditions=dict(
                step=baseline_step,
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ["step", "dump_index", "filename"]
                },
            ),
        )

        if row_baseline is None:
            print(f"Skip: target={str(path_target)} since no baseline")
            x_target = _load_object(path_target)
            if x_target is not None:
                print(f"x_target(sample)={get_truncated_value(x_target)}")
            continue

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]
        x_baseline = _load_object(path_baseline)
        x_target = _load_object(path_target)

        print(
            f"Check:\n"
            f"target={str(path_target)} (duplicate_index={row['duplicate_index']})\n"
            f"baseline={str(path_baseline)} (duplicate_index={row_baseline['duplicate_index']})"
        )

        if x_baseline is None or x_target is None:
            print(
                f"Skip comparison because of None: "
                f"x_baseline={x_baseline}, x_target={x_target}"
            )
            continue

        info = compare_tensors(
            x_baseline=x_baseline,
            x_target=x_target,
            name=row["name"],
        )
        print_comparison(info, diff_threshold=args.diff_threshold)
        print()


def compare_tensors(
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
    name: str = "",
) -> TensorComparisonInfo:
    raw_baseline_shape = tuple(x_baseline.shape)
    raw_baseline_dtype = str(x_baseline.dtype)
    raw_target_dtype = str(x_target.dtype)

    x_baseline = _try_unify_shape(x_baseline, target_shape=x_target.shape)
    baseline_unified_shape = tuple(x_baseline.shape)

    baseline_original_dtype = x_baseline.dtype
    target_original_dtype = x_target.dtype
    x_baseline_float = x_baseline.float()
    x_target_float = x_target.float()

    baseline_info = TensorInfo(
        shape=raw_baseline_shape,
        dtype=raw_baseline_dtype,
        stats=_compute_tensor_stats(x_baseline_float),
        sample=get_truncated_value(x_baseline_float),
    )
    target_info = TensorInfo(
        shape=tuple(x_target.shape),
        dtype=raw_target_dtype,
        stats=_compute_tensor_stats(x_target_float),
        sample=get_truncated_value(x_target_float),
    )

    shape_mismatch = x_baseline_float.shape != x_target_float.shape

    diff = None
    diff_downcast = None
    downcast_dtype_str = None

    if not shape_mismatch:
        diff = _compute_diff(x_baseline_float, x_target_float)

        if baseline_original_dtype != target_original_dtype:
            downcast_dtype = _compute_smaller_dtype(
                baseline_original_dtype, target_original_dtype
            )
            if downcast_dtype is not None:
                downcast_dtype_str = str(downcast_dtype)
                diff_downcast = _compute_diff(
                    x_baseline_float.to(downcast_dtype),
                    x_target_float.to(downcast_dtype),
                )

    return TensorComparisonInfo(
        name=name,
        baseline=baseline_info,
        target=target_info,
        baseline_unified_shape=baseline_unified_shape,
        shape_mismatch=shape_mismatch,
        diff=diff,
        diff_downcast=diff_downcast,
        downcast_dtype=downcast_dtype_str,
    )


def print_comparison(info: TensorComparisonInfo, diff_threshold: float = 1e-3) -> None:
    if info.baseline is not None and info.target is not None:
        dtype_marker = "" if info.baseline.dtype == info.target.dtype else "🟠"
        print(
            f"Raw "
            f"[shape] {info.baseline.shape} vs {info.target.shape}\t"
            f"[{dtype_marker}dtype] {info.baseline.dtype} vs {info.target.dtype}"
        )

        if info.baseline_unified_shape != info.baseline.shape:
            print(
                f"Unify shape: {info.baseline.shape} -> "
                f"{info.baseline_unified_shape} "
                f"(to match {info.target.shape})"
            )

    if (
        info.baseline is not None
        and info.target is not None
        and info.baseline.stats is not None
        and info.target.stats is not None
    ):
        stats_b = info.baseline.stats
        stats_t = info.target.stats
        for field_name in ["mean", "std", "min", "max", "p1", "p5", "p95", "p99"]:
            value_baseline = getattr(stats_b, field_name)
            value_target = getattr(stats_t, field_name)
            if value_baseline is None or value_target is None:
                continue
            print(
                f"[{field_name}] {value_baseline:.4f} vs {value_target:.4f} "
                f"(diff: {value_target - value_baseline:.4f})"
            )

    if info.shape_mismatch:
        print("⚠️ Shape mismatch")
        return

    if info.diff is not None:
        _print_diff(info.diff, diff_threshold=diff_threshold)

    if info.diff_downcast is not None and info.downcast_dtype is not None:
        _print_diff(
            info.diff_downcast,
            diff_threshold=diff_threshold,
            prefix_text=f"When downcast to {info.downcast_dtype}: ",
        )

    needs_sample = info.diff is not None and info.diff.max_abs_diff > 1e-3
    if needs_sample:
        if info.baseline is not None:
            print(f"x_baseline(sample)={info.baseline.sample}")
        if info.target is not None:
            print(f"x_target(sample)={info.target.sample}")


# ─── Internal ────────────────────────────────────────────────────────────────


def _compute_tensor_stats(x: torch.Tensor) -> TensorStats:
    include_quantiles = x.numel() < 10_000_000
    return TensorStats(
        mean=torch.mean(x).item(),
        std=torch.std(x).item(),
        min=torch.min(x).item(),
        max=torch.max(x).item(),
        p1=torch.quantile(x, 0.01).item() if include_quantiles else None,
        p5=torch.quantile(x, 0.05).item() if include_quantiles else None,
        p95=torch.quantile(x, 0.95).item() if include_quantiles else None,
        p99=torch.quantile(x, 0.99).item() if include_quantiles else None,
    )


def _compute_diff(x_baseline: torch.Tensor, x_target: torch.Tensor) -> DiffInfo:
    raw_abs_diff = (x_target - x_baseline).abs()
    max_diff_coord = _argmax_coord(raw_abs_diff)
    return DiffInfo(
        rel_diff=_calc_rel_diff(x_target, x_baseline),
        max_abs_diff=raw_abs_diff.max().item(),
        mean_abs_diff=raw_abs_diff.mean().item(),
        max_diff_coord=max_diff_coord,
        baseline_at_max=x_baseline[max_diff_coord].item(),
        target_at_max=x_target[max_diff_coord].item(),
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


def _argmax_coord(x: torch.Tensor) -> tuple:
    flat_idx = x.argmax()
    return tuple(idx.item() for idx in torch.unravel_index(flat_idx, x.shape))


def _compute_smaller_dtype(dtype_a, dtype_b):
    info_dict = {
        (torch.float32, torch.bfloat16): torch.bfloat16,
    }
    return info_dict.get((dtype_a, dtype_b)) or info_dict.get((dtype_b, dtype_a))


def _try_unify_shape(x: torch.Tensor, target_shape):
    x_shape = x.shape
    num_dim_to_remove = len(x_shape) - len(target_shape)
    if (x_shape[num_dim_to_remove:] == target_shape) and all(
        val == 1 for val in x_shape[:num_dim_to_remove]
    ):
        return functools.reduce(lambda a, _: a.squeeze(0), range(num_dim_to_remove), x)

    return x


# Copied from DeepGEMM
def _calc_rel_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _load_object(path):
    try:
        x = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"Skip load {path} since error {e}")
        return None

    if isinstance(x, dict) and "value" in x:
        x = x["value"]

    if not isinstance(x, torch.Tensor):
        print(f"Skip load {path} since {type(x)=} is not a Tensor ({x=})")
        return None
    return x.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=1000000)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--filter", type=str, default=None, help="Regex to filter filenames"
    )
    args = parser.parse_args()
    main(args)
