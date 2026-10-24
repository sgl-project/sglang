from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import polars as pl

from sglang.srt.debug_utils.comparator.utils import Pair


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
    """Pair baseline/target tensor files that share the same identity key.

    The identity key is every column except ``skip_keys`` (e.g. drop ``rank``
    so the same logical tensor on different DP ranks lands in one bundle).

    Linear time: each side is hashed into ``key -> bundle`` in a single pass,
    then target keys drive the pairing -- a target key absent from baseline
    yields an empty baseline bundle, and baseline-only keys are dropped.
    """
    match_key_cols: list[str] = [c for c in dfs.y.columns if c not in skip_keys]
    # Baseline may lack some of target's key columns (e.g. a framework-specific
    # field in a cross-framework compare); match on the shared subset, exactly
    # as the previous per-key filter did by skipping absent-column conditions.
    shared_positions: list[int] = [
        i for i, c in enumerate(match_key_cols) if c in dfs.x.columns
    ]
    baseline_key_cols: list[str] = [match_key_cols[i] for i in shared_positions]

    baseline_by_key: dict[tuple, TensorBundleInfo] = _group_by_key(
        dfs.x, key_cols=baseline_key_cols
    )
    target_by_key: dict[tuple, TensorBundleInfo] = _group_by_key(
        dfs.y, key_cols=match_key_cols
    )

    return [
        Pair(
            x=baseline_by_key.get(tuple(target_key[i] for i in shared_positions), []),
            y=target_infos,
        )
        for target_key, target_infos in target_by_key.items()
    ]


def _group_by_key(
    df: pl.DataFrame, *, key_cols: list[str]
) -> dict[tuple, TensorBundleInfo]:
    # dict preserves first-appearance order, matching the previous
    # unique(maintain_order=True) iteration order (same underlying df order).
    if df.is_empty():
        return {}

    col_index: dict[str, int] = {name: i for i, name in enumerate(df.columns)}
    key_index: list[int] = [col_index[c] for c in key_cols]
    info_index: list[tuple[str, int]] = [
        (f.name, col_index[f.name])
        for f in dataclasses.fields(TensorFileInfo)
        if f.name in col_index
    ]

    groups: dict[tuple, TensorBundleInfo] = {}
    for row in df.iter_rows():
        key: tuple = tuple(row[i] for i in key_index)
        info: TensorFileInfo = TensorFileInfo(
            **{name: row[i] for name, i in info_index}
        )
        groups.setdefault(key, []).append(info)
    return groups


def _rows_to_tensor_infos(rows: list[dict[str, Any]]) -> list[TensorFileInfo]:
    tensor_info_fields: set[str] = {f.name for f in dataclasses.fields(TensorFileInfo)}
    return [
        TensorFileInfo(**{k: v for k, v in row.items() if k in tensor_info_fields})
        for row in rows
    ]
