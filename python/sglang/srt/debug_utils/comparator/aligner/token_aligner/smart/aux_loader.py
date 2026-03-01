from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    execute_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_per_step_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_plugins import (
    AUX_NAMES,
    _AuxFrameworkPlugin,
    _plugins,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerGlobalAux,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.dims import (
    ParallelAxis,
    TokenLayout,
    apply_dim_names,
    resolve_dim_names,
)
from sglang.srt.debug_utils.comparator.dp_utils import filter_to_non_empty_dp_rank
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

# re-export for existing callers
__all__ = [
    "AUX_NAMES",
    "has_aux_tensors",
    "load_and_normalize_aux",
]


def load_and_normalize_aux(
    dump_path: Path, df: pl.DataFrame
) -> Optional[TokenAlignerGlobalAux]:
    """Bootstrap: load, unshard, and normalize auxiliary tensors for one side."""
    plugin: Optional[_AuxFrameworkPlugin] = _detect_plugin(df, dump_path=dump_path)
    if plugin is None:
        return None

    available_names: set[str] = set(df["name"].unique().to_list()) & plugin.all_names
    steps: list[int] = sorted(df["step"].unique().to_list())
    tensor_names: set[str] = available_names & plugin.tensor_names
    non_tensor_names: set[str] = available_names & plugin.non_tensor_names

    steps_data: dict[int, dict[str, object]] = {}
    thd_seq_lens_by_step: dict[int, list[int]] = {}
    for step in steps:
        step_data, thd_seq_lens = _load_step_data(
            step=step,
            tensor_names=tensor_names,
            non_tensor_names=non_tensor_names,
            df=df,
            dump_path=dump_path,
            plugin=plugin,
        )
        if step_data:
            steps_data[step] = step_data
        if thd_seq_lens is not None:
            thd_seq_lens_by_step[step] = thd_seq_lens

    layout: TokenLayout = plugin.detect_layout(steps_data)

    step_auxs: dict[int, TokenAlignerStepAux] = {
        step: plugin.compute_step_aux(step_data, layout=layout, step=step)
        for step, step_data in steps_data.items()
    }

    return TokenAlignerGlobalAux(
        step_auxs=step_auxs,
        framework=plugin.name,
        layout=layout,
        thd_seq_lens_by_step=thd_seq_lens_by_step or None,
    )


def has_aux_tensors(df: pl.DataFrame) -> bool:
    """Check if the DataFrame contains the minimum auxiliary tensors for alignment."""
    names: set[str] = set(df["name"].unique().to_list())
    return any(plugin.has_required_names(names) for plugin in _plugins)


def _detect_plugin(df: pl.DataFrame, dump_path: Path) -> Optional[_AuxFrameworkPlugin]:
    names: set[str] = set(df["name"].unique().to_list())

    for plugin in _plugins:
        if names & plugin.discriminating_names:
            return plugin

    first_row: dict = df.row(0, named=True)
    value: ValueWithMeta = ValueWithMeta.load(dump_path / first_row["filename"])

    for plugin in _plugins:
        if f"{plugin.name}_parallel_info" in value.meta:
            return plugin

    return None


def _load_step_data(
    *,
    step: int,
    tensor_names: set[str],
    non_tensor_names: set[str],
    df: pl.DataFrame,
    dump_path: Path,
    plugin: _AuxFrameworkPlugin,
) -> tuple[dict[str, object], Optional[list[int]]]:
    """Load all tensor and non-tensor aux values for a single step.

    Two-pass loading: non-CP-sharded tensors first (to obtain cu_seqlens_q
    for seq_lens), then CP-sharded tensors with seq_lens for THD unshard/reorder.

    Returns (step_data, thd_global_seq_lens).
    """
    result: dict[str, object] = {}

    # Pass 0: non-tensor values
    for name in non_tensor_names:
        value = _load_non_tensor_aux(name=name, step=step, df=df, dump_path=dump_path)
        if value is not None:
            result[name] = value

    # Pass 1: non-CP-sharded tensors (e.g. cu_seqlens_q, seq_lens)
    non_cp_tensor_names: set[str] = tensor_names - plugin.cp_sharded_names
    cp_tensor_names: set[str] = tensor_names & plugin.cp_sharded_names

    for name in non_cp_tensor_names:
        tensor = _load_and_align_aux_tensor(
            name=name, step=step, df=df, dump_path=dump_path, plugin=plugin
        )
        if tensor is not None:
            result[name] = tensor

    # Derive global seq_lens for THD unshard (framework-specific extraction)
    thd_global_seq_lens: Optional[list[int]] = plugin.extract_global_seq_lens(result)

    # Pass 2: CP-sharded tensors (input_ids, position_ids, etc.)
    for name in cp_tensor_names:
        tensor = _load_and_align_aux_tensor(
            name=name,
            step=step,
            df=df,
            dump_path=dump_path,
            plugin=plugin,
            thd_global_seq_lens=thd_global_seq_lens,
        )
        if tensor is not None:
            result[name] = tensor

    return result, thd_global_seq_lens


def _load_non_tensor_aux(
    *, name: str, step: int, df: pl.DataFrame, dump_path: Path
) -> Optional[object]:
    """Load a non-tensor auxiliary value for a step, validating consistency across ranks."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]
    loaded = filter_to_non_empty_dp_rank(loaded)

    if len(loaded) > 1:
        first_value = loaded[0].value
        for i, item in enumerate(loaded[1:], start=1):
            if item.value != first_value:
                warning_sink.add(
                    GeneralWarning(
                        category=f"{name}_mismatch",
                        message=(
                            f"{name} mismatch across ranks: rank 0 has {first_value}, "
                            f"rank {i} has {item.value}"
                        ),
                    )
                )
                break

    return loaded[0].value


def _load_and_align_aux_tensor(
    *,
    name: str,
    step: int,
    df: pl.DataFrame,
    dump_path: Path,
    plugin: _AuxFrameworkPlugin,
    thd_global_seq_lens: Optional[list[int]] = None,
) -> Optional[torch.Tensor]:
    """Load an auxiliary tensor for (name, step), align if needed."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]
    loaded = filter_to_non_empty_dp_rank(loaded)

    tensors: list[torch.Tensor] = [
        item.value for item in loaded if isinstance(item.value, torch.Tensor)
    ]
    if not tensors:
        return None

    if len(tensors) == 1:
        return tensors[0]

    metas: list[dict[str, Any]] = [item.meta for item in loaded]
    metas = _ensure_dims_in_metas(
        name=name, plugin=plugin, metas=metas, ndim=tensors[0].ndim
    )

    sub_plans = compute_per_step_sub_plans(
        metas=metas,
        thd_global_seq_lens=(
            thd_global_seq_lens if name in plugin.cp_sharded_names else None
        ),
    )
    if sub_plans:
        dims_str: Optional[str] = metas[0].get("dims")
        if dims_str is not None:
            dim_names: list[str] = resolve_dim_names(dims_str)
            tensors = [apply_dim_names(t, dim_names) for t in tensors]

        result, _replicated_checks = execute_sub_plans(tensors=tensors, plans=sub_plans)
        assert result is not None
        return result.rename(None)  # strip named dims before returning to plugin

    warning_sink.add(
        GeneralWarning(
            category="aux_no_dims",
            message=(
                f"aux tensor '{name}' has {len(tensors)} ranks "
                f"but no dims metadata, using rank 0 only"
            ),
        )
    )
    return tensors[0]


def _ensure_dims_in_metas(
    *,
    name: str,
    plugin: _AuxFrameworkPlugin,
    metas: list[dict[str, Any]],
    ndim: int,
) -> list[dict[str, Any]]:
    """Inject inferred dims into metas if not already present.

    Returns metas unchanged if dims is already set, or a new list with dims
    injected if inference succeeds for CP-sharded tensors.
    """
    if metas[0].get("dims") is not None:
        return metas

    parallel_infos = [normalize_parallel_info(m) for m in metas]
    has_cp: bool = any(ParallelAxis.CP in info for info in parallel_infos)
    if not has_cp:
        return metas

    if name in plugin.cp_sharded_names:
        inferred_dims: str = plugin.infer_cp_sharded_dims(name=name, ndim=ndim)
        return [{**m, "dims": inferred_dims} for m in metas]

    return metas
