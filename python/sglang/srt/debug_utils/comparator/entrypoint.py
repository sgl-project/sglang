from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.entrypoint import (
    TokenAlignerResult,
    compute_maybe_token_aligner_result,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_loader import (
    AUX_NAMES,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.bundle_comparator import compare_bundle_pair
from sglang.srt.debug_utils.comparator.bundle_matcher import (
    TensorBundleInfo,
    match_bundles,
)
from sglang.srt.debug_utils.comparator.display import emit_display_records
from sglang.srt.debug_utils.comparator.meta_overrider import MetaOverrider
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    ConfigRecord,
    NonTensorRecord,
    SkipRecord,
    SummaryRecord,
    report_sink,
)
from sglang.srt.debug_utils.comparator.per_token_visualizer import (
    generate_per_token_heatmap,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import read_meta, read_tokenizer_path


def main() -> None:
    args = _parse_args()
    sys.exit(run(args))


def run(args: argparse.Namespace) -> int:
    report_path: Optional[Path] = _resolve_report_path(args)
    report_sink.configure(
        output_format=args.output_format,
        report_path=report_path,
    )

    try:
        report_sink.add(ConfigRecord.from_args(args))

        dfs: Pair[pl.DataFrame] = _read_df(args)

        tokenizer: Any = _maybe_load_tokenizer(args)
        for label, df, dump_dir in [
            ("baseline", dfs.x, Path(args.baseline_path)),
            ("target", dfs.y, Path(args.target_path)),
        ]:
            emit_display_records(
                df=df,
                dump_dir=dump_dir,
                label=label,
                tokenizer=tokenizer,
            )

        ta_result: TokenAlignerResult = compute_maybe_token_aligner_result(args, dfs)

        if ta_result.mode == "smart":
            dfs = dfs.map(lambda df: df.filter(~pl.col("name").is_in(AUX_NAMES)))

        bundle_info_pairs: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=dfs,
            skip_keys=_compute_skip_keys(
                args, has_token_aligner=ta_result.mode is not None
            ),
        )

        viz_output_dir: Optional[Path] = (
            Path(args.viz_output_dir) if args.viz_bundle_details else None
        )

        visualize_per_token: Optional[Path] = (
            Path(args.visualize_per_token) if args.visualize_per_token else None
        )

        meta_overrider: MetaOverrider = MetaOverrider.from_args_and_config(
            override_dims=args.override_dims,
            override_baseline_dims=args.override_baseline_dims,
            override_target_dims=args.override_target_dims,
            override_config=(
                Path(args.override_config) if args.override_config else None
            ),
        )

        comparison_records = _compare_bundle_pairs(
            bundle_info_pairs=bundle_info_pairs,
            baseline_path=Path(args.baseline_path),
            target_path=Path(args.target_path),
            token_aligner_mode=ta_result.mode,
            token_aligner_plan=ta_result.plan,
            diff_threshold=args.diff_threshold,
            thd_seq_lens_by_step_pair=ta_result.thd_seq_lens_by_step_pair,
            viz_output_dir=viz_output_dir,
            compute_per_token=visualize_per_token is not None,
            meta_overrider=meta_overrider,
        )
        summary, skipped_names = _consume_comparison_records(
            comparison_records=comparison_records,
            visualize_per_token=visualize_per_token,
        )
        return _compute_exit_code(
            summary,
            allow_skip_pattern=args.allow_skip_pattern,
            skipped_names=skipped_names,
        )
    finally:
        report_sink.close()
        if report_path is not None:
            print(f"Report: {report_path}", file=sys.stderr)


def _compute_exit_code(
    summary: SummaryRecord,
    *,
    allow_skip_pattern: str,
    skipped_names: list[str],
) -> int:
    if summary.failed > 0:
        return 1

    pattern: re.Pattern[str] = re.compile(allow_skip_pattern)
    forbidden: list[str] = [n for n in skipped_names if not pattern.fullmatch(n)]
    if forbidden:
        return 1

    return 0


def _resolve_report_path(args: argparse.Namespace) -> Optional[Path]:
    if args.report_path is not None:
        return Path(args.report_path) if args.report_path else None
    return Path(args.target_path) / "comparator_report.jsonl"


def _maybe_load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer_path: Optional[str] = getattr(args, "tokenizer", None)

    if tokenizer_path is None:
        for directory in [Path(args.baseline_path), Path(args.target_path)]:
            tokenizer_path = read_tokenizer_path(directory)
            if tokenizer_path is not None:
                break

    if tokenizer_path is None:
        return None

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        return None


def _maybe_load_tokenizer(args: argparse.Namespace) -> Any:
    tokenizer_path: Optional[str] = getattr(args, "tokenizer", None)

    if tokenizer_path is None:
        for directory in [Path(args.baseline_path), Path(args.target_path)]:
            tokenizer_path = read_tokenizer_path(directory)
            if tokenizer_path is not None:
                break

    if tokenizer_path is None:
        return None

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        return None


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


def _compute_skip_keys(args, *, has_token_aligner: bool) -> set[str]:
    skip_keys: set[str] = {"dump_index", "filename"}
    if args.grouping == "logical":
        skip_keys |= {"rank", "recompute_status"}
        if has_token_aligner:
            skip_keys |= {"step"}
    return skip_keys


def _compare_bundle_pairs(
    *,
    bundle_info_pairs: list[Pair[TensorBundleInfo]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]],
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
    meta_overrider: Optional[MetaOverrider] = None,
) -> Iterator[Union[ComparisonRecord, SkipRecord, NonTensorRecord]]:
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
            token_aligner_mode=token_aligner_mode,
            token_aligner_plan=token_aligner_plan,
            diff_threshold=diff_threshold,
            thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
            viz_output_dir=viz_output_dir,
            compute_per_token=compute_per_token,
            meta_overrider=meta_overrider,
        )


def _consume_comparison_records(
    *,
    comparison_records: Iterator[Union[ComparisonRecord, SkipRecord, NonTensorRecord]],
    visualize_per_token: Optional[Path] = None,
) -> tuple[SummaryRecord, list[str]]:
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}
    collected_comparisons: list[ComparisonRecord] = []
    skipped_names: list[str] = []

    for record in comparison_records:
        counts[record.category] += 1
        report_sink.add(record)
        if isinstance(record, SkipRecord) and record.category == "skipped":
            skipped_names.append(record.name)
        if visualize_per_token is not None and isinstance(record, ComparisonRecord):
            collected_comparisons.append(record)

    summary: SummaryRecord = SummaryRecord(total=sum(counts.values()), **counts)
    report_sink.add(summary)

    if visualize_per_token is not None and collected_comparisons:
        generate_per_token_heatmap(
            records=collected_comparisons,
            output_path=visualize_per_token,
        )

    return summary, skipped_names


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
    parser.add_argument(
        "--token-aligner",
        type=str,
        choices=["smart", "concat_steps"],
        default="concat_steps",
        help="Token aligner mode: concat_steps (BS=1, no aux needed) or smart (BS>1, sequence matching)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path for decoding input_ids (auto-discovered from dump metadata if not set)",
    )
    parser.add_argument(
        "--viz-bundle-details",
        action="store_true",
        default=False,
        help="Generate comparison heatmap/histogram PNG for each compared tensor",
    )
    parser.add_argument(
        "--viz-output-dir",
        type=str,
        default="/tmp/comparator_viz/",
        help="Output directory for visualization PNGs (default: /tmp/comparator_viz/)",
    )
    parser.add_argument(
        "--visualize-per-token",
        type=str,
        default=None,
        help="Output path for per-token relative difference heatmap PNG",
    )

    # Dims override
    parser.add_argument(
        "--override-dims",
        action="append",
        default=[],
        help="Override dims for both sides: 'name:dims_string' (repeatable)",
    )
    parser.add_argument(
        "--override-baseline-dims",
        action="append",
        default=[],
        help="Override dims for baseline only: 'name:dims_string' (repeatable)",
    )
    parser.add_argument(
        "--override-target-dims",
        action="append",
        default=[],
        help="Override dims for target only: 'name:dims_string' (repeatable)",
    )
    parser.add_argument(
        "--override-config",
        type=str,
        default=None,
        help="Path to YAML override config file (dims overrides, etc.)",
    )
    parser.add_argument(
        "--allow-skip-pattern",
        type=str,
        default=".*",
        help="Regex pattern for tensor names allowed to be skipped. "
        "Default '.*' allows all skips. Use '^$' to forbid all skips.",
    )

    # Report output
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path for JSONL report (default: <target-path>/comparator_report.jsonl). "
        "Pass empty string '' to disable.",
    )

    return parser.parse_args()
