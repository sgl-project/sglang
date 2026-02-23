import argparse

import polars as pl

from sglang.srt.debug_utils.comparator.output_types import (
    ConfigRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.pipeline import process_logical_tensor
from sglang.srt.debug_utils.dump_loader import read_meta


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

    for logical_group in logical_groups.iter_rows(named=True):
        process_logical_tensor(
            row=logical_group,
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
