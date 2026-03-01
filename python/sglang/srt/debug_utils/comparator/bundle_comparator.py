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
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    apply_dim_names,
    parse_dims,
    resolve_dim_names,
)
from sglang.srt.debug_utils.comparator.dp_utils import filter_to_non_empty_dp_rank
from sglang.srt.debug_utils.comparator.meta_overrider import MetaOverrider
from sglang.srt.debug_utils.comparator.output_types import (
    ComparisonRecord,
    GeneralWarning,
    NonTensorRecord,
    SkipRecord,
)
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import (
    compare_tensor_pair,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import LOAD_FAILED, ValueWithMeta

_FAILED_SIDE_MAP: dict[str, str] = {"x": "baseline", "y": "target"}


def compare_bundle_pair(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
    meta_overrider: Optional[MetaOverrider] = None,
) -> Union[ComparisonRecord, SkipRecord, NonTensorRecord]:
    with warning_sink.context() as collected_warnings:
        result = _compare_bundle_pair_inner(
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

    return result.model_copy(update={"warnings": collected_warnings})


def _compare_bundle_pair_inner(
    *,
    name: str,
    filenames_pair: Pair[list[str]],
    baseline_path: Path,
    target_path: Path,
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
    meta_overrider: Optional[MetaOverrider] = None,
) -> Union[ComparisonRecord, SkipRecord, NonTensorRecord]:
    # 1. Load all successfully loaded values
    all_pair: Pair[list[ValueWithMeta]] = Pair(
        x=_load_all_values(filenames=filenames_pair.x, base_path=baseline_path),
        y=_load_all_values(filenames=filenames_pair.y, base_path=target_path),
    )

    if not all_pair.x or not all_pair.y:
        reason = "baseline_load_failed" if not all_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # 1b. Dims override: patch meta["dims"] before DP filter reads it
    # (--override-dims may add ``# dp:=moe_dp``, so it must run first)
    if meta_overrider is not None and not meta_overrider.is_empty:
        _apply = meta_overrider.apply_to_meta
        all_pair = Pair(
            x=[
                ValueWithMeta(
                    value=v.value, meta=_apply(name=name, meta=v.meta, side="baseline")
                )
                for v in all_pair.x
            ],
            y=[
                ValueWithMeta(
                    value=v.value, meta=_apply(name=name, meta=v.meta, side="target")
                )
                for v in all_pair.y
            ],
        )

    # 1c. DP filter: keep only the non-empty dp_rank
    all_pair = all_pair.map(
        lambda items: filter_to_non_empty_dp_rank(
            items, dp_group_alias=_extract_dp_alias_from_items(items)
        )
    )

    # 2. Check if any side has non-tensor values → non-tensor display path
    has_non_tensor: bool = any(
        not isinstance(it.value, torch.Tensor) for it in [*all_pair.x, *all_pair.y]
    )
    if has_non_tensor:
        return _compare_bundle_pair_non_tensor_type(name=name, value_pair=all_pair)

    # 3. All values are tensors → tensor comparison path
    return _compare_bundle_pair_tensor_type(
        name=name,
        valid_pair=all_pair,
        token_aligner_mode=token_aligner_mode,
        token_aligner_plan=token_aligner_plan,
        diff_threshold=diff_threshold,
        thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
        viz_output_dir=viz_output_dir,
        compute_per_token=compute_per_token,
    )


def _extract_dp_alias_from_items(items: list[ValueWithMeta]) -> Optional[str]:
    """Extract dp group alias from the first item's ``meta["dims"]``."""
    if not items:
        return None
    dims_str: Optional[str] = items[0].meta.get("dims")
    if dims_str is None:
        return None
    return parse_dims(dims_str).dp_group_alias


def _compare_bundle_pair_tensor_type(
    *,
    name: str,
    valid_pair: Pair[list[ValueWithMeta]],
    token_aligner_mode: Optional[str],
    token_aligner_plan: Optional[TokenAlignerPlan],
    diff_threshold: float,
    thd_seq_lens_by_step_pair: Pair[Optional[dict[int, list[int]]]] = Pair(
        x=None, y=None
    ),
    viz_output_dir: Optional[Path] = None,
    compute_per_token: bool = False,
) -> Union[ComparisonRecord, SkipRecord]:
    if not valid_pair.x or not valid_pair.y:
        reason = "baseline_load_failed" if not valid_pair.x else "target_load_failed"
        return SkipRecord(name=name, reason=reason)

    # Plan (meta only, no tensor)
    metas_pair: Pair[list[dict[str, Any]]] = valid_pair.map(
        lambda items: [it.meta for it in items]
    )
    plan: AlignerPlan = compute_aligner_plan(
        metas_pair=metas_pair,
        token_aligner_mode=token_aligner_mode,
        token_aligner_plan=token_aligner_plan,
        thd_seq_lens_by_step_pair=thd_seq_lens_by_step_pair,
    )

    # Apply dim names to tensors, then execute
    tensors_pair: Pair[list[torch.Tensor]] = Pair(
        x=_apply_dim_names_from_meta(
            tensors=[it.value for it in valid_pair.x],
            metas=metas_pair.x,
        ),
        y=_apply_dim_names_from_meta(
            tensors=[it.value for it in valid_pair.y],
            metas=metas_pair.y,
        ),
    )
    aligner_result: AlignerResult = execute_aligner_plan(
        tensors_pair=tensors_pair, plan=plan
    )
    replicated_checks = aligner_result.replicated_checks

    if aligner_result.tensors is None:
        assert aligner_result.failed_side_xy is not None
        side_name: str = _FAILED_SIDE_MAP[aligner_result.failed_side_xy]
        reason: str = f"{side_name}_load_failed"
        return SkipRecord(name=name, reason=reason)

    # Resolve seq_dim for per-token computation
    seq_dim: Optional[int] = (
        _resolve_seq_dim(aligner_result.tensors.y) if compute_per_token else None
    )

    # Compare
    aligned_baseline: torch.Tensor = aligner_result.tensors.x.rename(None)
    aligned_target: torch.Tensor = aligner_result.tensors.y.rename(None)

    info = compare_tensor_pair(
        x_baseline=aligned_baseline,
        x_target=aligned_target,
        name=name,
        diff_threshold=diff_threshold,
        seq_dim=seq_dim,
    )
    record = ComparisonRecord(
        **info.model_dump(),
        aligner_plan=plan,
        replicated_checks=replicated_checks,
    )

    if viz_output_dir is not None:
        _try_generate_viz(
            baseline=aligned_baseline,
            target=aligned_target,
            name=name,
            viz_output_dir=viz_output_dir,
        )

    return record


def _try_generate_viz(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    name: str,
    viz_output_dir: Path,
) -> None:
    from sglang.srt.debug_utils.comparator.visualizer import (
        generate_comparison_figure,
    )
    from sglang.srt.debug_utils.comparator.visualizer.preprocessing import (
        _sanitize_filename,
    )

    filename: str = _sanitize_filename(name) + ".png"
    output_path: Path = viz_output_dir / filename

    try:
        generate_comparison_figure(
            baseline=baseline,
            target=target,
            name=name,
            output_path=output_path,
        )
    except Exception as exc:
        warning_sink.add(
            GeneralWarning(
                category="visualizer",
                message=f"Visualization failed for {name}: {exc}",
            )
        )


def _resolve_seq_dim(tensor: torch.Tensor) -> Optional[int]:
    """Find the token/seq dimension index from the tensor's named dims."""
    if tensor.names[0] is None:
        return None

    names: tuple[Optional[str], ...] = tensor.names
    for target_name in (TOKEN_DIM_NAME, SEQ_DIM_NAME):
        if target_name in names:
            return list(names).index(target_name)

    return None


def _compare_bundle_pair_non_tensor_type(
    *,
    name: str,
    value_pair: Pair[list[ValueWithMeta]],
) -> NonTensorRecord:
    baseline_value: Any = value_pair.x[0].value
    target_value: Any = value_pair.y[0].value

    try:
        values_equal: bool = bool(baseline_value == target_value)
    except Exception:
        values_equal = False

    return NonTensorRecord(
        name=name,
        baseline_value=repr(baseline_value),
        target_value=repr(target_value),
        baseline_type=type(baseline_value).__name__,
        target_type=type(target_value).__name__,
        values_equal=values_equal,
    )


def _apply_dim_names_from_meta(
    *,
    tensors: list[torch.Tensor],
    metas: list[dict[str, Any]],
) -> list[torch.Tensor]:
    if not metas:
        return tensors

    dims_str: Optional[str] = metas[0].get("dims")
    if dims_str is None:
        return tensors

    dim_names: list[str] = resolve_dim_names(dims_str)
    return [apply_dim_names(t, dim_names) for t in tensors]


def _load_all_values(filenames: list[str], base_path: Path) -> list[ValueWithMeta]:
    result: list[ValueWithMeta] = []
    for f in filenames:
        item: ValueWithMeta = ValueWithMeta.load(base_path / f)
        if item.value is LOAD_FAILED:
            warning_sink.add(
                GeneralWarning(
                    category="load_failed",
                    message=f"Failed to load tensor file: {f}",
                )
            )
            continue
        result.append(item)
    return result
