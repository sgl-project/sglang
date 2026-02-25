import argparse
from pathlib import Path
from typing import Any

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    execute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.output_types import (
    AnyWarning,
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    _OutputRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows, read_meta

_NON_KEY_COLS = {"dump_index", "filename"}


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    warning_sink.set_output_format(args.output_format)

    df_baseline = read_meta(args.baseline_path)

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    print_record(
        ConfigRecord.from_args(args),
        output_format=args.output_format,
    )

    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}
    grouping: str = args.grouping

    non_key_cols = _NON_KEY_COLS | ({"rank"} if grouping == "logical" else set())
    key_cols = [c for c in df_target.columns if c not in non_key_cols]
    tensor_group_keys = df_target.unique(subset=key_cols)

    for tensor_group_key in tensor_group_keys.iter_rows(named=True):
        conditions = {k: tensor_group_key[k] for k in key_cols}
        baseline_rows = filter_rows(df_baseline, conditions=conditions)
        target_rows = filter_rows(df_target, conditions=conditions)

        record = _process_tensor_group(
            name=tensor_group_key["name"],
            baseline_filenames=[r["filename"] for r in baseline_rows],
            target_filenames=[r["filename"] for r in target_rows],
            baseline_path=Path(args.baseline_path),
            target_path=Path(args.target_path),
            diff_threshold=args.diff_threshold,
        )
        counts[record.category] += 1
        print_record(record, output_format=args.output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=args.output_format,
    )


def _process_tensor_group(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
) -> _OutputRecord:
    with warning_sink.context() as collected_warnings:
        return _process_tensor_group_raw(
            name=name,
            baseline_filenames=baseline_filenames,
            target_filenames=target_filenames,
            baseline_path=baseline_path,
            target_path=target_path,
            diff_threshold=diff_threshold,
            collected_warnings=collected_warnings,
        )


def _process_tensor_group_raw(
    *,
    name: str,
    baseline_filenames: list[str],
    target_filenames: list[str],
    baseline_path: Path,
    target_path: Path,
    diff_threshold: float,
    collected_warnings: list[AnyWarning],
) -> ComparisonRecord | SkipRecord:
    loaded_pair: Pair[list[ValueWithMeta]] = Pair(
        x=[ValueWithMeta.load(baseline_path / f) for f in baseline_filenames],
        y=[ValueWithMeta.load(target_path / f) for f in target_filenames],
    )

    metas_pair: Pair[list[dict[str, Any]]] = loaded_pair.map(
        lambda items: [item.meta for item in items]
    )

    plan = compute_aligner_plan(metas_pair=metas_pair)

    tensors_pair: Pair[list[torch.Tensor]] = loaded_pair.map(
        lambda items: [
            item.value for item in items if isinstance(item.value, torch.Tensor)
        ]
    )

    result = execute_aligner_plan(tensors_pair=tensors_pair, plan=plan)

    if result.tensors is None:
        reason = (
            f"{'baseline' if result.failed_side_xy == 'x' else 'target'}_load_failed"
        )
        return SkipRecord(name=name, reason=reason, warnings=collected_warnings)

    info = compare_tensor_pair(
        x_baseline=result.tensors.x,
        x_target=result.tensors.y,
        name=name,
        diff_threshold=diff_threshold,
    )

    return ComparisonRecord(**info.model_dump(), warnings=collected_warnings)


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
        "--grouping",
        type=str,
        choices=["logical", "raw"],
        default="logical",
        help="Grouping mode: logical (cross-rank unshard) or raw (rank-by-rank)",
    )
    return parser.parse_args()
