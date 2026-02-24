import argparse
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.unshard.load import load_and_unshard
from sglang.srt.debug_utils.comparator.utils import _single
from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def process_smart(
    *,
    target_rows: list[dict],
    baseline_rows: list[dict],
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    name = target_rows[0]["name"]
    fmt = args.output_format

    target_tensor = load_and_unshard(
        rows=target_rows,
        base_path=Path(args.target_path),
    )
    baseline_tensor = _load_baseline(
        rows=baseline_rows,
        base_path=Path(args.baseline_path),
    )

    if target_tensor is None or baseline_tensor is None:
        reason = "target_load_failed" if target_tensor is None else "baseline_load_failed"
        _skip(name, reason, counts, fmt)
        return

    _compare_and_record(
        x_baseline=baseline_tensor,
        x_target=target_tensor,
        name=name,
        args=args,
        counts=counts,
    )


def _load_baseline(
    *,
    rows: list[dict],
    base_path: Path,
) -> Optional[torch.Tensor]:
    if not rows:
        return None

    result = load_and_unshard(rows=rows, base_path=base_path)
    if result is not None:
        return result

    if len(rows) == 1:
        return _load_tensor(base_path / rows[0]["filename"])

    return None


def process_per_rank(
    *,
    target_rows: list[dict],
    baseline_rows: list[dict],
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    row = _single(target_rows)
    name = row["name"]

    if not baseline_rows:
        _skip(name, "no_baseline", counts, args.output_format)
        return

    if len(baseline_rows) > 1:
        _skip(name, "ambiguous_baseline", counts, args.output_format)
        return

    path_target = Path(args.target_path) / row["filename"]
    path_baseline = Path(args.baseline_path) / baseline_rows[0]["filename"]

    x_baseline = _load_tensor(path_baseline)
    x_target = _load_tensor(path_target)

    if x_baseline is None or x_target is None:
        _skip(name, "load_failed", counts, args.output_format)
        return

    _compare_and_record(
        x_baseline=x_baseline,
        x_target=x_target,
        name=name,
        args=args,
        counts=counts,
    )


def _compare_and_record(
    *,
    x_baseline: torch.Tensor,
    x_target: torch.Tensor,
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
    return _as_tensor(loaded.value)


def _as_tensor(value: object) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    return value


def _skip(
    name: str,
    reason: str,
    counts: dict[str, int],
    output_format: str,
) -> None:
    counts["skipped"] += 1
    print_record(SkipRecord(name=name, reason=reason), output_format=output_format)
