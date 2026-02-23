import argparse
from pathlib import Path
from typing import Optional

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparison import compare_tensors
from sglang.srt.debug_utils.comparator.unshard import (
    compute_unshard_plan,
    execute_unshard_plan,
    normalize_parallel_info,
)
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, find_row, read_meta


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    df_baseline = read_meta(args.baseline_path)

    print_record(
        ConfigRecord(
            baseline_path=args.baseline_path,
            target_path=args.target_path,
            diff_threshold=args.diff_threshold,
            start_step=args.start_step,
            end_step=args.end_step,
        ),
        output_format=args.output_format,
    )

    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    logical_key_cols = [
        c
        for c in df_target.columns
        if c not in {"rank", "dump_index", "filename", "duplicate_index"}
    ]
    logical_groups = df_target.unique(subset=logical_key_cols)

    for row in logical_groups.iter_rows(named=True):
        _process_logical_tensor(
            row=row,
            df_target=df_target,
            df_baseline=df_baseline,
            args=args,
            counts=counts,
            logical_key_cols=logical_key_cols,
        )

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=args.output_format,
    )


def _process_logical_tensor(
    *,
    row: dict,
    df_target: pl.DataFrame,
    df_baseline: pl.DataFrame,
    args: argparse.Namespace,
    counts: dict[str, int],
    logical_key_cols: list[str],
) -> None:
    target_rows = _filter_by_logical_key(df_target, row, logical_key_cols)
    first_target_path = Path(args.target_path) / target_rows[0]["filename"]

    first_target = ValueWithMeta.load(first_target_path)
    dims_str = first_target.meta.get("dims")

    if dims_str is not None:
        _process_with_dims(
            row=row,
            dims_str=dims_str,
            target_rows=target_rows,
            df_baseline=df_baseline,
            args=args,
            counts=counts,
            logical_key_cols=logical_key_cols,
            first_target=first_target,
        )
    else:
        _process_without_dims(
            target_rows=target_rows,
            df_baseline=df_baseline,
            args=args,
            counts=counts,
        )


def _process_with_dims(
    *,
    row: dict,
    dims_str: str,
    target_rows: list[dict],
    df_baseline: pl.DataFrame,
    args: argparse.Namespace,
    counts: dict[str, int],
    logical_key_cols: list[str],
    first_target: ValueWithMeta,
) -> None:
    name = row["name"]

    target_tensor = _unshard_side(
        rows=target_rows,
        base_path=Path(args.target_path),
        dims_str=dims_str,
        side_name="target",
        tensor_name=name,
        preloaded_first=first_target,
    )
    if target_tensor is None:
        counts["skipped"] += 1
        print_record(
            SkipRecord(name=name, reason="target_unshard_failed"),
            output_format=args.output_format,
        )
        return

    baseline_conditions = {k: row[k] for k in logical_key_cols}
    baseline_rows = _find_baseline_rows(df_baseline, baseline_conditions)

    if not baseline_rows:
        counts["skipped"] += 1
        print_record(
            SkipRecord(name=name, reason="no_baseline"),
            output_format=args.output_format,
        )
        return

    first_baseline_path = Path(args.baseline_path) / baseline_rows[0]["filename"]
    first_baseline = ValueWithMeta.load(first_baseline_path)
    baseline_dims_str = first_baseline.meta.get("dims")

    if baseline_dims_str is not None:
        baseline_tensor = _unshard_side(
            rows=baseline_rows,
            base_path=Path(args.baseline_path),
            dims_str=baseline_dims_str,
            side_name="baseline",
            tensor_name=name,
            preloaded_first=first_baseline,
        )
    else:
        if len(baseline_rows) > 1:
            counts["skipped"] += 1
            print_record(
                SkipRecord(name=name, reason="ambiguous_baseline_no_dims"),
                output_format=args.output_format,
            )
            return
        baseline_tensor = _extract_tensor(first_baseline)

    if baseline_tensor is None:
        counts["skipped"] += 1
        print_record(
            SkipRecord(name=name, reason="baseline_load_failed"),
            output_format=args.output_format,
        )
        return

    _compare_and_record(
        x_baseline=baseline_tensor,
        x_target=target_tensor,
        name=name,
        args=args,
        counts=counts,
    )


def _process_without_dims(
    *,
    target_rows: list[dict],
    df_baseline: pl.DataFrame,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    for row in target_rows:
        path_target = Path(args.target_path) / row["filename"]
        baseline_step = row["step"]

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
            counts["skipped"] += 1
            print_record(
                SkipRecord(name=row["name"], reason="no_baseline"),
                output_format=args.output_format,
            )
            continue

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]

        x_baseline = _load_tensor(path_baseline)
        x_target = _load_tensor(path_target)

        if x_baseline is None or x_target is None:
            counts["skipped"] += 1
            print_record(
                SkipRecord(name=row["name"], reason="load_failed"),
                output_format=args.output_format,
            )
            continue

        _compare_and_record(
            x_baseline=x_baseline,
            x_target=x_target,
            name=row["name"],
            args=args,
            counts=counts,
        )


def _compare_and_record(
    *,
    x_baseline,
    x_target,
    name: str,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    info = compare_tensors(
        x_baseline=x_baseline,
        x_target=x_target,
        name=name,
        diff_threshold=args.diff_threshold,
    )

    if info.diff is not None and info.diff.passed:
        counts["passed"] += 1
    else:
        counts["failed"] += 1

    print_record(
        ComparisonRecord(**info.model_dump()),
        output_format=args.output_format,
    )


def _load_tensor(path: Path) -> Optional[torch.Tensor]:
    loaded = ValueWithMeta.load(path)
    if not isinstance(loaded.value, torch.Tensor):
        return None
    return loaded.value


def _unshard_side(
    *,
    rows: list[dict],
    base_path: Path,
    dims_str: str,
    side_name: str,
    tensor_name: str,
    preloaded_first: Optional[ValueWithMeta] = None,
) -> Optional[torch.Tensor]:
    dim_specs = parse_dims(dims_str)
    loaded: list[ValueWithMeta] = []

    for i, row in enumerate(rows):
        if i == 0 and preloaded_first is not None:
            loaded.append(preloaded_first)
        else:
            path = base_path / row["filename"]
            loaded.append(ValueWithMeta.load(path))

    parallel_infos = [normalize_parallel_info(item.meta) for item in loaded]
    world_ranks = [item.meta.get("rank", i) for i, item in enumerate(loaded)]

    try:
        world_ranks_int = [int(r) for r in world_ranks]
    except (TypeError, ValueError):
        world_ranks_int = list(range(len(loaded)))

    plan = compute_unshard_plan(
        dim_specs=dim_specs,
        parallel_infos=parallel_infos,
        tensor_name=tensor_name,
        dims_str=dims_str,
    )

    tensors_by_world_rank: dict[int, torch.Tensor] = {}
    for world_rank, item in zip(world_ranks_int, loaded):
        tensor = _extract_tensor(item)
        if tensor is None:
            return None
        tensors_by_world_rank[world_rank] = tensor

    return execute_unshard_plan(plan, tensors_by_world_rank)


def _extract_tensor(item: ValueWithMeta) -> Optional[torch.Tensor]:
    value = item.value
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    return value


def _filter_by_logical_key(
    df: pl.DataFrame, row: dict, logical_key_cols: list[str]
) -> list[dict]:
    import functools

    conditions = [
        (
            pl.col(col) == _cast_value(row[col], df.schema[col])
            if row[col] is not None
            else pl.col(col).is_null()
        )
        for col in logical_key_cols
        if col in df.columns
    ]
    filtered = df.filter(functools.reduce(lambda a, b: a & b, conditions))
    return filtered.to_dicts()


def _find_baseline_rows(df_baseline: pl.DataFrame, conditions: dict) -> list[dict]:
    import functools

    filter_exprs = [
        (
            pl.col(col) == _cast_value(val, df_baseline.schema[col])
            if val is not None
            else pl.col(col).is_null()
        )
        for col, val in conditions.items()
        if col in df_baseline.columns
    ]

    if not filter_exprs:
        return []

    filtered = df_baseline.filter(functools.reduce(lambda a, b: a & b, filter_exprs))
    return filtered.to_dicts()


def _cast_value(value, target_dtype):
    if target_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return int(value)
    elif target_dtype in (pl.Float64, pl.Float32):
        return float(value)
    elif target_dtype == pl.Boolean:
        return bool(value)
    elif target_dtype == pl.String:
        return str(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=1000000)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--filter", type=str, default=None, help="Regex to filter filenames"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (JSONL, one JSON object per line)",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="Which DP rank to compare",
    )
    return parser.parse_args()
