import argparse
from pathlib import Path

import polars as pl

from sglang.srt.debug_utils.comparator.output_types import (
    ConfigRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.pipeline import process_tensor_group
from sglang.srt.debug_utils.dump_loader import filter_rows, read_meta

_NON_KEY_COLS = {"dump_index", "filename"}


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    df_baseline = read_meta(args.baseline_path)

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

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
    grouping: str = args.grouping

    non_key_cols = _NON_KEY_COLS | ({"rank"} if grouping == "logical" else set())
    key_cols = [c for c in df_target.columns if c not in non_key_cols]
    tensor_group_keys = df_target.unique(subset=key_cols)

    for tensor_group_key in tensor_group_keys.iter_rows(named=True):
        conditions = {k: tensor_group_key[k] for k in key_cols}
        baseline_rows = filter_rows(df_baseline, conditions=conditions)
        target_rows = filter_rows(df_target, conditions=conditions)

        record = process_tensor_group(
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
