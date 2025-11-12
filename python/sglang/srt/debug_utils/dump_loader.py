import functools
import os
from pathlib import Path
from typing import Any, Dict

import polars as pl
import torch


class DumpLoader:
    def __init__(self):
        directory = os.environ.get("SGLANG_DUMP_LOADER_DIR")

        self._enable = directory is not None
        if self._enable:
            self._directory = Path(directory)
            self._df = read_meta(directory)

    @property
    def enable(self):
        return self._enable

    def load(self, name, **kwargs):
        assert self._enable, "Please call DumpLoader.load only when it is enabled"

        from sglang.srt.debug_utils.dumper import dumper

        forward_pass_id = dumper._forward_pass_id
        conditions = dict(name=name, forward_pass_id=forward_pass_id, **kwargs)
        row = find_row(self._df, conditions=conditions)
        assert (
            row is not None
        ), f"DumpLoader cannot find row given query {name=} {kwargs=} {self._directory=}"

        path = self._directory / row["filename"]
        output = torch.load(path, weights_only=False)

        print(
            f"[DumpLoader] load from {path=} (query: {name=} {kwargs=}, output: {type(output)})"
        )
        return output


def read_meta(directory):
    directory = Path(directory)
    assert directory.is_dir(), f"{directory=} should be a directory"

    rows = []
    for p in directory.glob("*.pt"):
        full_kwargs = {}
        for kv in p.stem.split("___"):
            k, v = kv.split("=")
            full_kwargs[k] = v
        rows.append(
            {
                "filename": str(p.name),
                **full_kwargs,
            }
        )

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("forward_pass_id").cast(int),
        pl.col("rank").cast(int),
        pl.col("dump_index").cast(int),
    )
    return df


def find_row(df, conditions: Dict[str, Any]):
    df_sub = df.filter(
        functools.reduce(
            lambda a, b: a & b,
            [
                pl.col(col) == _cast_to_polars_dtype(conditions[col], df.schema[col])
                for col in conditions.keys()
            ],
        )
    )
    assert len(df_sub) <= 1
    return df_sub.to_dicts()[0] if len(df_sub) > 0 else None


def _cast_to_polars_dtype(value, target_dtype):
    if target_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return int(value)
    elif target_dtype in (pl.Float64, pl.Float32):
        return float(value)
    elif target_dtype == pl.Boolean:
        return bool(value)
    elif target_dtype == pl.String:
        return str(value)
    else:
        return value


dump_loader = DumpLoader()
