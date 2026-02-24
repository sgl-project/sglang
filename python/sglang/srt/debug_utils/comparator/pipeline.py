import argparse
from pathlib import Path
from typing import Callable, Optional

import torch

from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
    print_record,
)
from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.unshard.load import load_and_unshard
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

LoadFn = Callable[[list[dict], Path], Optional[torch.Tensor]]

_LOAD_FNS: dict[str, LoadFn] = {
    "smart": load_and_unshard,
    "per-rank": lambda rows, base_path: _load_single_tensor(rows, base_path),
}


def process(
    *,
    baseline_rows: list[dict],
    target_rows: list[dict],
    args: argparse.Namespace,
    counts: dict[str, int],
    load_fn: LoadFn,
) -> None:
    name = (baseline_rows or target_rows)[0]["name"]
    fmt = args.output_format

    baseline_tensor = load_fn(baseline_rows, Path(args.baseline_path))
    target_tensor = load_fn(target_rows, Path(args.target_path))

    if baseline_tensor is None or target_tensor is None:
        reason = "baseline_load_failed" if baseline_tensor is None else "target_load_failed"
        _skip(name, reason, counts, fmt)
        return

    _compare_and_record(
        x_baseline=baseline_tensor,
        x_target=target_tensor,
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


def _load_single_tensor(rows: list[dict], base_path: Path) -> Optional[torch.Tensor]:
    if len(rows) != 1:
        return None

    loaded = ValueWithMeta.load(base_path / rows[0]["filename"])
    if not isinstance(loaded.value, torch.Tensor):
        return None
    return loaded.value


def _skip(
    name: str,
    reason: str,
    counts: dict[str, int],
    output_format: str,
) -> None:
    counts["skipped"] += 1
    print_record(SkipRecord(name=name, reason=reason), output_format=output_format)
