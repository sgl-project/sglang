from __future__ import annotations

from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import polars as pl

from sglang.srt.debug_utils.comparator.output_types import (
    InputIdsRecord,
    RankInfoRecord,
    report_sink,
)
from sglang.srt.debug_utils.dump_loader import LOAD_FAILED, ValueWithMeta

_PARALLEL_INFO_KEYS: list[str] = ["sglang_parallel_info", "megatron_parallel_info"]


def emit_display_records(
    *,
    df: pl.DataFrame,
    dump_dir: Path,
    label: str,
    tokenizer: Any,
) -> None:
    rank_rows: Optional[list[dict[str, Any]]] = _collect_rank_info(
        df, dump_dir=dump_dir
    )
    if rank_rows is not None:
        report_sink.add(RankInfoRecord(label=label, rows=rank_rows))

    input_ids_rows: Optional[list[dict[str, Any]]] = _collect_input_ids_and_positions(
        df, dump_dir=dump_dir, tokenizer=tokenizer
    )
    if input_ids_rows is not None:
        report_sink.add(InputIdsRecord(label=label, rows=input_ids_rows))


def _render_polars_as_text(df: pl.DataFrame, *, title: Optional[str] = None) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for col in df.columns:
        table.add_column(col)
    for row in df.iter_rows():
        table.add_row(*[str(v) for v in row])

    buf = StringIO()
    Console(file=buf, force_terminal=False, width=200).print(table)
    return buf.getvalue().rstrip("\n")


def _collect_rank_info(
    df: pl.DataFrame, dump_dir: Path
) -> Optional[list[dict[str, Any]]]:
    unique_rows: pl.DataFrame = (
        df.filter(pl.col("name") == "input_ids")
        .sort("rank")
        .unique(subset=["rank"], keep="first")
    )
    if unique_rows.is_empty():
        return None

    table_rows: list[dict[str, Any]] = []
    for row in unique_rows.to_dicts():
        meta: dict[str, Any] = ValueWithMeta.load(dump_dir / row["filename"]).meta

        row_data: dict[str, Any] = {"rank": row["rank"]}
        for key in _PARALLEL_INFO_KEYS:
            _extract_parallel_info(row_data=row_data, info=meta.get(key, {}))
        table_rows.append(row_data)

    return table_rows or None


def _collect_input_ids_and_positions(
    df: pl.DataFrame,
    dump_dir: Path,
    *,
    tokenizer: Any = None,
) -> Optional[list[dict[str, Any]]]:
    filtered: pl.DataFrame = df.filter(pl.col("name").is_in(["input_ids", "positions"]))
    if filtered.is_empty():
        return None

    data_by_step_rank: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    for row in filtered.to_dicts():
        key: tuple[int, int] = (row["step"], row["rank"])
        item: ValueWithMeta = ValueWithMeta.load(dump_dir / row["filename"])
        if item.value is not LOAD_FAILED:
            data_by_step_rank[key][row["name"]] = item.value

    table_rows: list[dict[str, Any]] = []
    for (step, rank), data in sorted(data_by_step_rank.items()):
        ids = data.get("input_ids")
        pos = data.get("positions")

        ids_list: Optional[list[int]] = (
            ids.flatten().tolist() if ids is not None else None
        )

        row_data: dict[str, Any] = {
            "step": step,
            "rank": rank,
            "num_tokens": len(ids_list) if ids_list is not None else None,
            "input_ids": str(ids_list) if ids_list is not None else "N/A",
            "positions": str(pos.flatten().tolist()) if pos is not None else "N/A",
        }

        if tokenizer is not None and ids_list is not None:
            row_data["decoded_text"] = repr(
                tokenizer.decode(ids_list, skip_special_tokens=False)
            )

        table_rows.append(row_data)

    return table_rows or None


def _extract_parallel_info(row_data: dict[str, Any], info: dict[str, Any]) -> None:
    if not info or info.get("error"):
        return

    for key in sorted(info.keys()):
        if key.endswith("_rank"):
            base: str = key[:-5]
            size_key: str = f"{base}_size"
            if size_key in info:
                row_data[base] = f"{info[key]}/{info[size_key]}"
