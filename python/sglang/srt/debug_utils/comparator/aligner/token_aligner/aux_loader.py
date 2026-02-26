from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import polars as pl
import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    execute_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    compute_per_step_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    AUX_NAMES,
    _AuxFrameworkPlugin,
    _plugins,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerGlobalAux,
    TokenAlignerStepAux,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.parallel_info import (
    normalize_parallel_info,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.srt.debug_utils.dump_loader import ValueWithMeta, filter_rows

# re-export for existing callers
__all__ = ["AUX_NAMES", "has_aux_tensors", "load_and_normalize_aux"]


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
    for step in steps:
        step_data = dict(
            _load_step_data(
                step=step,
                tensor_names=tensor_names,
                non_tensor_names=non_tensor_names,
                df=df,
                dump_path=dump_path,
                plugin=plugin,
            )
        )
        if step_data:
            steps_data[step] = step_data

    layout: str = plugin.detect_layout(steps_data)

    step_auxs: dict[int, TokenAlignerStepAux] = {
        step: plugin.compute_step_aux(step_data, layout=layout, step=step)
        for step, step_data in steps_data.items()
    }

    return TokenAlignerGlobalAux(
        step_auxs=step_auxs, framework=plugin.name, layout=layout
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
) -> Iterable[Tuple[str, object]]:
    """Load all tensor and non-tensor aux values for a single step."""
    for name in non_tensor_names:
        value = _load_non_tensor_aux(name=name, step=step, df=df, dump_path=dump_path)
        if value is not None:
            yield name, value

    for name in tensor_names:
        tensor = _load_and_align_aux_tensor(
            name=name, step=step, df=df, dump_path=dump_path, plugin=plugin
        )
        if tensor is not None:
            yield name, tensor


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
) -> Optional[torch.Tensor]:
    """Load an auxiliary tensor for (name, step), align if needed."""
    rows = filter_rows(df, conditions={"name": name, "step": step})
    if not rows:
        return None

    loaded: list[ValueWithMeta] = [
        ValueWithMeta.load(dump_path / r["filename"]) for r in rows
    ]

    tensors: list[torch.Tensor] = [
        item.value for item in loaded if isinstance(item.value, torch.Tensor)
    ]
    if not tensors:
        return None

    if len(tensors) == 1:
        return tensors[0]

    metas: list[dict] = [item.meta for item in loaded]
    metas = _ensure_dims_in_metas(name=name, plugin=plugin, metas=metas)

    sub_plans = compute_per_step_sub_plans(metas=metas)
    if sub_plans:
        result = execute_sub_plans(tensors=tensors, plans=sub_plans)
        assert result is not None
        return result

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
    *, name: str, plugin: _AuxFrameworkPlugin, metas: list[dict]
) -> list[dict]:
    """Inject inferred dims into metas if not already present.

    Returns metas unchanged if dims is already set, or a new list with dims
    injected if inference succeeds. Raises if the tensor is CP-sharded
    (not yet supported).
    """
    if metas[0].get("dims") is not None:
        return metas

    parallel_infos = [normalize_parallel_info(m) for m in metas]
    has_cp: bool = any(ParallelAxis.CP in info for info in parallel_infos)
    if not has_cp:
        return metas

    if name in plugin.cp_sharded_names:
        raise NotImplementedError(
            f"Aux tensor '{name}' is CP-sharded but reorderer does not yet support "
            f"zigzag reordering on the 't' dimension. "
            f"Pass explicit dims= at dump time or wait for t-dim zigzag support."
        )

    return metas
