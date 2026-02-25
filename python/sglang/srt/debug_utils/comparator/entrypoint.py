import argparse
from pathlib import Path

import polars as pl

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparison import compare_tensors
from sglang.srt.debug_utils.comparator.utils import load_object
from sglang.srt.debug_utils.dump_loader import find_row, read_meta


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

    for row in df_target.iter_rows(named=True):
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

        x_baseline = load_object(path_baseline)
        x_target = load_object(path_target)

        if x_baseline is None or x_target is None:
            counts["skipped"] += 1
            print_record(
                SkipRecord(name=row["name"], reason="load_failed"),
                output_format=args.output_format,
            )
            continue

        info = compare_tensors(
            x_baseline=x_baseline,
            x_target=x_target,
            name=row["name"],
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

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=args.output_format,
    )


def _parse_args() -> argparse.Namespace:
    # python -m sglang.srt.debug_utils.comparator --baseline-path ... --target-path ...
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
    return parser.parse_args()
