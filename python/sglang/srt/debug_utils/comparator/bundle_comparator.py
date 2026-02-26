"""Compare two tensor bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    AlignerResult,
    execute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_aligner_plan,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import AlignerPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_FAILED_SIDE_MAP: dict[str, str] = {"x": "baseline", "y": "target"}


def compare_bundle_pair(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    with warning_sink.context() as collected_warnings:
        result = _compare_bundle_pair_raw(
            name=name,
            filenames_pair=filenames_pair,
            baseline_path=baseline_path,
            target_path=target_path,
            token_aligner_plan=token_aligner_plan,
            diff_threshold=diff_threshold,
        )

    return result.model_copy(update={"warnings": collected_warnings})


def _compare_bundle_pair_raw(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
) -> Union[ComparisonRecord, SkipRecord]:
    # 1. Load (tensor + meta, ungrouped)
    valid_pair: Pair[list[ValueWithMeta]] = Pair(
        x=_load_valid_tensors(filenames=filenames_pair.x, base_path=baseline_path),
        y=_load_valid_tensors(filenames=filenames_pair.y, base_path=target_path),
    )

    if not valid_pair.x or not valid_pair.y:
        reason = "baseline_load_failed" if not valid_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # 2. Plan (meta only, no tensor)
    metas_pair: Pair[list[dict[str, Any]]] = valid_pair.map(
        lambda items: [it.meta for it in items]
    )
    plan: AlignerPlan = compute_aligner_plan(
        metas_pair=metas_pair, token_aligner_plan=token_aligner_plan
    )

    # 3. Execute (tensor + plan only, no meta)
    tensors_pair: Pair[list[torch.Tensor]] = valid_pair.map(
        lambda items: [it.value for it in items]
    )
    aligner_result: AlignerResult = execute_aligner_plan(
        tensors_pair=tensors_pair, plan=plan
    )

    if aligner_result.tensors is None:
        assert aligner_result.failed_side_xy is not None
        side_name: str = _FAILED_SIDE_MAP[aligner_result.failed_side_xy]
        reason = f"{side_name}_load_failed"
        return SkipRecord(name=name, reason=reason)

    # 4. Compare
    info = compare_tensor_pair(
        x_baseline=aligner_result.tensors.x,
        x_target=aligner_result.tensors.y,
        name=name,
        diff_threshold=diff_threshold,
    )
    return ComparisonRecord(**info.model_dump())


def _load_valid_tensors(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    return [
        x
        for f in filenames
        if isinstance((x := ValueWithMeta.load(base_path / f)).value, torch.Tensor)
    ]
