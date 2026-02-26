from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional, Union

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    AUX_NAMES,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.entrypoint import (
    compute_maybe_token_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.bundle_comparator import compare_bundle_pair
from sglang.srt.debug_utils.comparator.bundle_matcher import (
    TensorBundleInfo,
    match_bundles,
)
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    SkipRecord,
    SummaryRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import read_meta


def main() -> None:
    args = _parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    print_record(
        ConfigRecord.from_args(args),
        output_format=args.output_format,
    )

    warning_sink.set_output_format(args.output_format)

    dfs: Pair[pl.DataFrame] = _read_df(args)
    token_aligner_plan = compute_maybe_token_aligner_plan(args, dfs)

    dfs = dfs.map(lambda df: df.filter(~pl.col("name").is_in(AUX_NAMES)))

    bundle_info_pairs: list[Pair[TensorBundleInfo]] = match_bundles(
        dfs=dfs,
        skip_keys=_compute_skip_keys(
            args, has_token_aligner_plan=token_aligner_plan is not None
        ),
    )

    comparison_records = _compare_bundle_pairs(
        bundle_info_pairs=bundle_info_pairs,
        baseline_path=Path(args.baseline_path),
        target_path=Path(args.target_path),
        token_aligner_plan=token_aligner_plan,
        diff_threshold=args.diff_threshold,
    )
    _consume_comparison_records(
        comparison_records=comparison_records, output_format=args.output_format
    )


def _read_df(args: argparse.Namespace) -> Pair[pl.DataFrame]:
    df_baseline = read_meta(args.baseline_path)

    df_target = read_meta(args.target_path)
    df_target = df_target.filter(
        (pl.col("step") >= args.start_step) & (pl.col("step") <= args.end_step)
    )
    if args.filter:
        df_target = df_target.filter(pl.col("filename").str.contains(args.filter))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    return Pair(x=df_baseline, y=df_target)


def _compute_skip_keys(args, *, has_token_aligner_plan: bool):
    skip_keys: set[str] = {"dump_index", "filename"}
    if args.grouping == "logical":
        skip_keys |= {"rank"}
        if has_token_aligner_plan:
            skip_keys |= {"step"}
    return skip_keys


def _compare_bundle_pairs(
    *,
    bundle_info_pairs: list[Pair[TensorBundleInfo]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
) -> Iterator[Union[ComparisonRecord, SkipRecord]]:
    for bundle_info_pair in bundle_info_pairs:
        if not bundle_info_pair.y:
            continue

        name: str = bundle_info_pair.y[0].name
        filenames_pair: Pair[list[str]] = bundle_info_pair.map(
            lambda infos: [info.filename for info in infos]
        )
        yield compare_bundle_pair(
            name=name,
            filenames_pair=filenames_pair,
            baseline_path=baseline_path,
            target_path=target_path,
            token_aligner_plan=token_aligner_plan,
            diff_threshold=diff_threshold,
        )


def _consume_comparison_records(
    *,
    comparison_records: Iterator[Union[ComparisonRecord, SkipRecord]],
    output_format: str,
) -> None:
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    for record in comparison_records:
        counts[record.category] += 1
        print_record(record, output_format=output_format)

    print_record(
        SummaryRecord(total=sum(counts.values()), **counts),
        output_format=output_format,
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
