from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import polars as pl

from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.dump_loader import filter_rows


@dataclass(frozen=True)
class TensorFileInfo:
    filename: str
    name: str
    step: int


TensorBundleInfo = list[TensorFileInfo]


def match_bundles(
    *,
    dfs: Pair[pl.DataFrame],
    skip_keys: set[str],
) -> list[Pair[TensorBundleInfo]]:
    match_key_cols: list[str] = [c for c in dfs.y.columns if c not in skip_keys]
    unique_keys: pl.DataFrame = dfs.y.select(match_key_cols).unique(maintain_order=True)

    results: list[Pair[TensorBundleInfo]] = []
    for key_values in unique_keys.iter_rows(named=True):
        result = dfs.map(
            lambda df: _rows_to_tensor_infos(filter_rows(df, conditions=key_values))
        )
        results.append(result)

    return results


def _rows_to_tensor_infos(rows: list[dict[str, Any]]) -> list[TensorFileInfo]:
    tensor_info_fields: set[str] = {f.name for f in dataclasses.fields(TensorFileInfo)}
    return [
        TensorFileInfo(**{k: v for k, v in row.items() if k in tensor_info_fields})
        for row in rows
    ]
