from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_loader import (
    _detect_plugin,
    _load_and_align_aux_tensor,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_plugins import (
    _AuxFrameworkPlugin,
)


def load_thd_seq_lens_only(
    dump_path: Path, df: pl.DataFrame
) -> Optional[dict[int, list[int]]]:
    plugin: Optional[_AuxFrameworkPlugin] = _detect_plugin(df, dump_path=dump_path)
    if plugin is None or not plugin.cp_sharded_names:
        return None

    non_cp_tensor_names: set[str] = (
        set(df["name"].unique().to_list()) & plugin.tensor_names
    ) - plugin.cp_sharded_names
    steps: list[int] = sorted(df["step"].unique().to_list())

    result: dict[int, list[int]] = {}
    for step in steps:
        step_data: dict[str, object] = {}
        for name in non_cp_tensor_names:
            tensor = _load_and_align_aux_tensor(
                name=name, step=step, df=df, dump_path=dump_path, plugin=plugin
            )
            if tensor is not None:
                step_data[name] = tensor

        seq_lens: Optional[list[int]] = plugin.extract_global_seq_lens(step_data)
        if seq_lens is not None:
            result[step] = seq_lens

    return result or None
