from __future__ import annotations

import argparse
import sys
import traceback as _traceback_module
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
    ComparisonErrorRecord,
    ComparisonNonTensorRecord,
    ComparisonSkipRecord,
    ComparisonTensorRecord,
    ConfigRecord,
    RecordLocation,
    SummaryRecord,
)
from sglang.srt.debug_utils.comparator.per_token_visualizer import (
    generate_per_token_heatmap,
)
from sglang.srt.debug_utils.comparator.preset import PRESETS, expand_preset
from sglang.srt.debug_utils.comparator.report_sink import report_sink
from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    auto_descend_dir,
    compute_exit_code,
)
from sglang.srt.debug_utils.dump_loader import read_meta, read_tokenizer_path

_DEFAULT_SKIP_KEYS: set[str] = {"dump_index", "filename"}


def main() -> None:
    args = parse_args(sys.argv[1:])
    sys.exit(run(args))


def run(args: argparse.Namespace) -> int:
    report_sink.configure(
        output_format=args.output_format,
        report_path=None,
        verbosity=args.verbosity,
    )

    dir_pair: Pair[Path] = Pair(
        x=auto_descend_dir(Path(args.baseline_path), label="baseline_path"),
        y=auto_descend_dir(Path(args.target_path), label="target_path"),
    )
    viz_output_dir: Optional[Path] = (
        Path(args.viz_output_dir) if args.viz_bundle_details else None
    )
    visualize_per_token: Optional[Path] = (
        Path(args.visualize_per_token) if args.visualize_per_token else None
    )
    override_config: Optional[Path] = (
        Path(args.override_config) if args.override_config else None
    )

    report_path: Optional[Path] = _resolve_report_path(
        target_path=dir_pair.y,
        report_path_arg=args.report_path,
    )
    report_sink.configure(
        output_format=args.output_format,
        report_path=report_path,
        verbosity=args.verbosity,
    )

    try:
        report_sink.add(ConfigRecord(config=vars(args)))

        dfs: Pair[pl.DataFrame] = _read_df(
            dir_pair=dir_pair,
            start_step=args.start_step,
            end_step=args.end_step,
            filter_pattern=args.filter,
        )

        tokenizer: Any = _maybe_load_tokenizer(
            tokenizer_arg=args.tokenizer, dir_pair=dir_pair
        )
        for label, df, dump_dir in [
            ("baseline", dfs.x, dir_pair.x),
            ("target", dfs.y, dir_pair.y),
        ]:
            emit_display_records(
                df=df, dump_dir=dump_dir, label=label, tokenizer=tokenizer
            )

        ta_result: TokenAlignerResult = compute_maybe_token_aligner_result(
            dir_pair=dir_pair,
            dfs=dfs,
            token_aligner_mode=args.token_aligner,
        )

        if ta_result.mode == "smart":
            dfs = dfs.map(lambda df: df.filter(~pl.col("name").is_in(AUX_NAMES)))

        skip_keys: set[str] = _DEFAULT_SKIP_KEYS | set(args.grouping_skip_keys or [])
        bundle_info_pairs: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=dfs, skip_keys=skip_keys
        )

        meta_overrider: MetaOverrider = MetaOverrider.from_args_and_config(
            override_dims=args.override_dims,
            override_baseline_dims=args.override_baseline_dims,
            override_target_dims=args.override_target_dims,
            override_config=override_config,
        )

        comparison_records = _compare_bundle_pairs(
            bundle_info_pairs=bundle_info_pairs,
            dir_pair=dir_pair,
            token_aligner_mode=ta_result.mode,
            token_aligner_plan=ta_result.plan,
            diff_threshold=args.diff_threshold,
            thd_seq_lens_by_step_pair=ta_result.thd_seq_lens_by_step_pair,
            viz_output_dir=viz_output_dir,
            compute_per_token=visualize_per_token is not None,
            meta_overrider=meta_overrider,
        )
        summary, skipped_names, failed_names, errored_names = (
            _consume_comparison_records(
                comparison_records=comparison_records,
                visualize_per_token=visualize_per_token,
            )
        )
        return compute_exit_code(
            summary,
            allow_skipped_pattern=args.allow_skipped_pattern,
            skipped_names=skipped_names,
            allow_failed_pattern=args.allow_failed_pattern,
            failed_names=failed_names,
            errored_names=errored_names,
        )
    finally:
        report_sink.close()
        if report_path is not None:
            print(f"Report: {report_path}", file=sys.stderr)


def _resolve_report_path(
    *, target_path: Path, report_path_arg: Optional[str]
) -> Optional[Path]:
    if report_path_arg is not None:
        return Path(report_path_arg) if report_path_arg else None
    return target_path / "comparator_report.jsonl"


def _maybe_load_tokenizer(*, tokenizer_arg: Optional[str], dir_pair: Pair[Path]) -> Any:
    tokenizer_path: Optional[str] = tokenizer_arg

    if tokenizer_path is None:
        for directory in [dir_pair.x, dir_pair.y]:
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


def _read_df(
    *,
    dir_pair: Pair[Path],
    start_step: int,
    end_step: int,
    filter_pattern: Optional[str],
) -> Pair[pl.DataFrame]:
    df_baseline = read_meta(dir_pair.x)

    df_target = read_meta(dir_pair.y)
    df_target = df_target.filter(
        (pl.col("step") >= start_step) & (pl.col("step") <= end_step)
    )
    if filter_pattern:
        df_target = df_target.filter(pl.col("filename").str.contains(filter_pattern))
    assert all(c in df_target.columns for c in ["rank", "step", "dump_index", "name"])

    return Pair(x=df_baseline, y=df_target)


def _compare_bundle_pairs(
    *,
    bundle_info_pairs: list[Pair[TensorBundleInfo]],
    dir_pair: Pair[Path],
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]],
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
    meta_overrider: Optional[MetaOverrider] = None,
) -> Iterator[
    Union[
        ComparisonTensorRecord,
        ComparisonSkipRecord,
        ComparisonNonTensorRecord,
        ComparisonErrorRecord,
    ]
]:
    for bundle_info_pair in bundle_info_pairs:
        if not bundle_info_pair.y:
            continue

        name: str = bundle_info_pair.y[0].name
        filenames_pair: Pair[list[str]] = bundle_info_pair.map(
            lambda infos: [info.filename for info in infos]
        )

        record: Union[
            ComparisonTensorRecord,
            ComparisonSkipRecord,
            ComparisonNonTensorRecord,
            ComparisonErrorRecord,
        ]
        try:
            record = compare_bundle_pair(
                name=name,
                filenames_pair=filenames_pair,
                dir_pair=dir_pair,
                token_aligner_mode=token_aligner_mode,
                token_aligner_plan=token_aligner_plan,
                diff_threshold=diff_threshold,
                thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
                viz_output_dir=viz_output_dir,
                compute_per_token=compute_per_token,
                meta_overrider=meta_overrider,
            )
        except Exception as exc:
            record = ComparisonErrorRecord(
                name=name,
                exception_type=type(exc).__name__,
                traceback_str=_traceback_module.format_exc(),
            )

        target_steps: set[int] = {info.step for info in bundle_info_pair.y}
        step: Optional[int] = target_steps.pop() if len(target_steps) == 1 else None
        if step is not None:
            record = record.model_copy(update={"location": RecordLocation(step=step)})

        yield record


def _consume_comparison_records(
    *,
    comparison_records: Iterator[
        Union[
            ComparisonTensorRecord,
            ComparisonSkipRecord,
            ComparisonNonTensorRecord,
            ComparisonErrorRecord,
        ]
    ],
    visualize_per_token: Optional[Path] = None,
) -> tuple[SummaryRecord, list[str], list[str], list[str]]:
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errored": 0}
    collected_comparisons: list[ComparisonTensorRecord] = []
    skipped_names: list[str] = []
    failed_names: list[str] = []
    errored_names: list[str] = []

    for record in comparison_records:
        counts[record.category] += 1
        report_sink.add(record)
        if isinstance(record, ComparisonSkipRecord) and record.category == "skipped":
            skipped_names.append(record.name)
        if record.category == "failed":
            failed_names.append(record.name)
        if isinstance(record, ComparisonErrorRecord):
            errored_names.append(record.name)
        if visualize_per_token is not None and isinstance(
            record, ComparisonTensorRecord
        ):
            collected_comparisons.append(record)

    summary: SummaryRecord = SummaryRecord(total=sum(counts.values()), **counts)
    report_sink.add(summary)

    if visualize_per_token is not None and collected_comparisons:
        generate_per_token_heatmap(
            records=collected_comparisons,
            output_path=visualize_per_token,
        )

    return summary, skipped_names, failed_names, errored_names


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments from an argv list. Applies preset expansion."""
    argv = expand_preset(argv, presets=PRESETS)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=1000000)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--filter", type=str, default=None, help="Regex to filter filenames (include)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json (JSONL, one JSON object per line)",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["minimal", "normal", "verbose"],
        default="normal",
        help="Output verbosity: minimal (1 line per tensor), normal (compact lifecycle), "
        "verbose (full detail). Default: normal",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="Preset configuration (expanded before parsing). "
        f"Available: {list(PRESETS.keys())}",
    )
    parser.add_argument(
        "--grouping-skip-keys",
        nargs="*",
        default=None,
        help="Metadata keys to skip when grouping bundles (additive on top of "
        "always-skipped dump_index and filename). "
        "E.g. '--grouping-skip-keys rank step' skips rank and step.",
    )
    parser.add_argument(
        "--token-aligner",
        type=str,
        choices=["smart", "concat_steps"],
        default=None,
        help="Token aligner mode: concat_steps (BS=1, no aux needed) or smart (BS>1, sequence matching). "
        "Default None (per-step comparison).",
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
        "--allow-skipped-pattern",
        type=str,
        default=".*",
        help="Regex pattern for tensor names allowed to be skipped. "
        "Default '.*' allows all skips. Use '^$' to forbid all skips.",
    )
    parser.add_argument(
        "--allow-failed-pattern",
        type=str,
        default=None,
        help="Regex pattern for tensor names allowed to fail without affecting exit code. "
        "Default None (all failures affect exit code).",
    )

    # Report output
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path for JSONL report (default: <target-path>/comparator_report.jsonl). "
        "Pass empty string '' to disable.",
    )

    return parser.parse_args(argv)
