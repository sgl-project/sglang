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
from sglang.srt.debug_utils.dump_loader import ValueWithMeta


def process_tensor_group(
    *,
    baseline_rows: list[dict],
    target_rows: list[dict],
    args: argparse.Namespace,
    counts: dict[str, int],
    grouping: str,
) -> None:
    name = (baseline_rows or target_rows)[0]["name"]

    baseline_tensor = load_tensor(baseline_rows, Path(args.baseline_path), grouping=grouping)
    target_tensor = load_tensor(target_rows, Path(args.target_path), grouping=grouping)

    if baseline_tensor is None or target_tensor is None:
        reason = "baseline_load_failed" if baseline_tensor is None else "target_load_failed"
        counts["skipped"] += 1
        print_record(SkipRecord(name=name, reason=reason), output_format=args.output_format)
        return

    info = compare_tensors(
        x_baseline=baseline_tensor,
        x_target=target_tensor,
        name=name,
        diff_threshold=args.diff_threshold,
    )
    k = "passed" if info.diff is not None and info.diff.passed else "failed"
    counts[k] += 1

    print_record(
        ComparisonRecord(**info.model_dump()),
        output_format=args.output_format,
    )


def load_tensor(
    rows: list[dict],
    base_path: Path,
    *,
    grouping: str,
) -> Optional[torch.Tensor]:
    if not rows:
        return None

    if grouping == "raw":
        if len(rows) != 1:
            return None
        return _as_tensor(ValueWithMeta.load(base_path / rows[0]["filename"]).value)

    return load_and_unshard(rows, base_path)


def _as_tensor(value: object) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    return value
