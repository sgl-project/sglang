import argparse
from pathlib import Path

import polars as pl

from sglang.srt.debug_utils.comparator.tensor_comparison import (
    compare_tensors,
    print_comparison,
    print_jsonl,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.types import (
    ComparisonLine,
    ConfigLine,
    SkipLine,
    SummaryLine,
)
from sglang.srt.debug_utils.comparator.utils import load_object
from sglang.srt.debug_utils.dump_loader import find_row, read_meta
from sglang.srt.debug_utils.dumper import get_truncated_value


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

    is_json = args.output_format == "json"

    if is_json:
        print_jsonl(ConfigLine(
            baseline_path=args.baseline_path,
            target_path=args.target_path,
            diff_threshold=args.diff_threshold,
            start_step=args.start_step,
            end_step=args.end_step,
        ))
    else:
        print("df_target", df_target)
        print("df_baseline", df_baseline)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

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
            skipped_count += 1
            if is_json:
                print_jsonl(SkipLine(name=row["name"], reason="no_baseline"))
            else:
                print(f"Skip: target={str(path_target)} since no baseline")
                x_target = load_object(path_target)
                if x_target is not None:
                    print(f"x_target(sample)={get_truncated_value(x_target)}")
            continue

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]

        if not is_json:
            print(
                f"Check:\n"
                f"target={str(path_target)} (duplicate_index={row['duplicate_index']})\n"
                f"baseline={str(path_baseline)} (duplicate_index={row_baseline['duplicate_index']})"
            )

        x_baseline = load_object(path_baseline)
        x_target = load_object(path_target)

        if x_baseline is None or x_target is None:
            skipped_count += 1
            if is_json:
                print_jsonl(SkipLine(name=row["name"], reason="load_failed"))
            else:
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

        is_passed = _check_passed(info=info, diff_threshold=args.diff_threshold)
        if is_passed:
            passed_count += 1
        else:
            failed_count += 1

        if is_json:
            print_jsonl(ComparisonLine(**info.model_dump()))
        else:
            print_comparison(info=info, diff_threshold=args.diff_threshold)
            print()

    if is_json:
        print_jsonl(SummaryLine(
            total=passed_count + failed_count + skipped_count,
            passed=passed_count,
            failed=failed_count,
            skipped=skipped_count,
        ))


def _check_passed(
    info: "TensorComparisonInfo",
    diff_threshold: float,
) -> bool:
    if info.diff is None:
        return False
    return (
        info.diff.rel_diff <= diff_threshold
        and info.diff.max_abs_diff <= diff_threshold
        and info.diff.mean_abs_diff <= diff_threshold
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
        "-o",
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (JSONL, one JSON object per line)",
    )
    return parser.parse_args()
