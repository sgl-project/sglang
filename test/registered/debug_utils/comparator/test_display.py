import sys
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pytest
import torch
from rich.console import Console

from sglang.srt.debug_utils.comparator.display import (
    _collect_input_ids_and_positions,
    _collect_rank_info,
    _extract_parallel_info,
    _render_polars_as_rich_table,
    _render_polars_as_text,
)
from sglang.srt.debug_utils.comparator.output_types import (
    InputIdsRecord,
    RankInfoRecord,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu-only", nightly=True)


def _render_rich(renderable: object) -> str:
    buf: StringIO = StringIO()
    Console(file=buf, force_terminal=False, width=120).print(renderable)
    return buf.getvalue().rstrip("\n")


def _save_dump_file(
    directory: Path,
    *,
    name: str,
    step: int,
    rank: int,
    dump_index: int,
    value: torch.Tensor,
    meta: dict,
) -> str:
    filename = f"name={name}___step={step}___rank={rank}___dump_index={dump_index}.pt"
    torch.save({"value": value, "meta": meta}, directory / filename)
    return filename


def _make_df(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("step").cast(int),
        pl.col("rank").cast(int),
        pl.col("dump_index").cast(int),
    )
    return df


class TestRenderPolarsAsText:
    def test_renders_table(self) -> None:
        df = pl.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        text: str = _render_polars_as_text(df, title="test table")

        assert "test table" in text
        assert "col_a" in text
        assert "col_b" in text

    def test_renders_empty_dataframe(self) -> None:
        df = pl.DataFrame({"a": [], "b": []})
        text: str = _render_polars_as_text(df, title="empty")
        assert "empty" in text


class TestCollectRankInfo:
    def test_collects_rank_info(self, tmp_path: Path) -> None:
        sglang_info = {
            "tp_rank": 0,
            "tp_size": 2,
            "pp_rank": 0,
            "pp_size": 1,
        }
        filename: str = _save_dump_file(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            dump_index=0,
            value=torch.tensor([1, 2, 3]),
            meta={"sglang_parallel_info": sglang_info},
        )
        df = _make_df(
            [
                {
                    "filename": filename,
                    "name": "input_ids",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                }
            ]
        )

        rows: Optional[list[dict[str, Any]]] = _collect_rank_info(df, dump_dir=tmp_path)

        assert rows is not None
        assert len(rows) == 1
        assert rows[0]["rank"] == 0
        assert rows[0]["tp"] == "0/2"
        assert rows[0]["pp"] == "0/1"

    def test_returns_none_when_no_input_ids(self, tmp_path: Path) -> None:
        df = _make_df(
            [
                {
                    "filename": "f.pt",
                    "name": "some_other",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                }
            ]
        )
        result = _collect_rank_info(df, dump_dir=tmp_path)
        assert result is None

    def test_deduplicates_ranks(self, tmp_path: Path) -> None:
        meta = {"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}}
        f1: str = _save_dump_file(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            dump_index=0,
            value=torch.tensor([1]),
            meta=meta,
        )
        f2: str = _save_dump_file(
            tmp_path,
            name="input_ids",
            step=1,
            rank=0,
            dump_index=1,
            value=torch.tensor([2]),
            meta=meta,
        )
        df = _make_df(
            [
                {
                    "filename": f1,
                    "name": "input_ids",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                },
                {
                    "filename": f2,
                    "name": "input_ids",
                    "step": 1,
                    "rank": 0,
                    "dump_index": 1,
                },
            ]
        )

        rows = _collect_rank_info(df, dump_dir=tmp_path)

        assert rows is not None
        assert len(rows) == 1


class TestCollectInputIdsAndPositions:
    def test_collects_ids_and_positions(self, tmp_path: Path) -> None:
        f_ids: str = _save_dump_file(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            dump_index=0,
            value=torch.tensor([10, 20, 30]),
            meta={},
        )
        f_pos: str = _save_dump_file(
            tmp_path,
            name="positions",
            step=0,
            rank=0,
            dump_index=1,
            value=torch.tensor([0, 1, 2]),
            meta={},
        )
        df = _make_df(
            [
                {
                    "filename": f_ids,
                    "name": "input_ids",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                },
                {
                    "filename": f_pos,
                    "name": "positions",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 1,
                },
            ]
        )

        rows = _collect_input_ids_and_positions(df, dump_dir=tmp_path)

        assert rows is not None
        assert len(rows) == 1
        assert rows[0]["step"] == 0
        assert rows[0]["rank"] == 0
        assert rows[0]["num_tokens"] == 3
        assert "10" in rows[0]["input_ids"]
        assert "0" in rows[0]["positions"]

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        df = _make_df(
            [
                {
                    "filename": "f.pt",
                    "name": "weight",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                }
            ]
        )
        result = _collect_input_ids_and_positions(df, dump_dir=tmp_path)
        assert result is None

    def test_with_mock_tokenizer(self, tmp_path: Path) -> None:
        f_ids: str = _save_dump_file(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            dump_index=0,
            value=torch.tensor([1, 2]),
            meta={},
        )
        df = _make_df(
            [
                {
                    "filename": f_ids,
                    "name": "input_ids",
                    "step": 0,
                    "rank": 0,
                    "dump_index": 0,
                }
            ]
        )

        class _MockTokenizer:
            def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
                return f"decoded:{ids}"

        rows = _collect_input_ids_and_positions(
            df, dump_dir=tmp_path, tokenizer=_MockTokenizer()
        )

        assert rows is not None
        assert "decoded_text" in rows[0]
        assert "decoded:" in rows[0]["decoded_text"]


class TestRankInfoRecordSnapshot:
    def test_to_text_snapshot(self) -> None:
        record = RankInfoRecord(
            label="baseline",
            rows=[
                {"rank": 0, "tp": "0/2", "pp": "0/1"},
                {"rank": 1, "tp": "1/2", "pp": "0/1"},
            ],
        )
        text: str = record.to_text()

        assert "baseline ranks" in text
        assert "rank" in text
        assert "tp" in text
        assert "pp" in text
        assert "0/2" in text
        assert "1/2" in text
        assert "0/1" in text

    def test_to_rich_snapshot(self) -> None:
        from rich.table import Table

        record = RankInfoRecord(
            label="baseline",
            rows=[
                {"rank": 0, "tp": "0/2", "pp": "0/1"},
                {"rank": 1, "tp": "1/2", "pp": "0/1"},
            ],
        )
        body = record._format_rich_body()

        assert isinstance(body, Table)
        rendered: str = _render_rich(body)
        assert "baseline ranks" in rendered
        assert "0/2" in rendered
        assert "1/2" in rendered

    def test_json_roundtrip(self) -> None:
        record = RankInfoRecord(
            label="target",
            rows=[{"rank": 0, "tp": "0/4"}],
        )
        json_str: str = record.model_dump_json()

        assert '"type":"rank_info"' in json_str
        assert '"label":"target"' in json_str
        assert '"tp":"0/4"' in json_str


class TestInputIdsRecordSnapshot:
    def test_to_text_snapshot(self) -> None:
        record = InputIdsRecord(
            label="target",
            rows=[
                {
                    "step": 0,
                    "rank": 0,
                    "num_tokens": 3,
                    "input_ids": "[10, 20, 30]",
                    "positions": "[0, 1, 2]",
                },
            ],
        )
        text: str = record.to_text()

        assert "target input_ids & positions" in text
        assert "step" in text
        assert "num_tokens" in text
        assert "10, 20, 30" in text
        assert "0, 1, 2" in text

    def test_to_rich_snapshot(self) -> None:
        from rich.table import Table

        record = InputIdsRecord(
            label="target",
            rows=[
                {
                    "step": 0,
                    "rank": 0,
                    "num_tokens": 3,
                    "input_ids": "[10, 20, 30]",
                    "positions": "[0, 1, 2]",
                },
            ],
        )
        body = record._format_rich_body()

        assert isinstance(body, Table)
        rendered: str = _render_rich(body)
        assert "target input_ids & positions" in rendered
        assert "10, 20, 30" in rendered
        assert "0, 1, 2" in rendered

    def test_json_roundtrip(self) -> None:
        record = InputIdsRecord(
            label="baseline",
            rows=[
                {
                    "step": 0,
                    "rank": 0,
                    "num_tokens": 2,
                    "input_ids": "[1, 2]",
                    "positions": "[0, 1]",
                    "decoded_text": "'hello'",
                },
            ],
        )
        json_str: str = record.model_dump_json()

        assert '"type":"input_ids"' in json_str
        assert '"label":"baseline"' in json_str
        assert '"decoded_text"' in json_str

    def test_to_text_with_decoded(self) -> None:
        record = InputIdsRecord(
            label="test",
            rows=[
                {
                    "step": 0,
                    "rank": 0,
                    "num_tokens": 2,
                    "input_ids": "[1, 2]",
                    "positions": "[0, 1]",
                    "decoded_text": "'hello world'",
                },
            ],
        )
        text: str = record.to_text()

        assert "decoded_text" in text
        assert "hello world" in text


class TestExtractParallelInfo:
    def test_extracts_rank_size_pairs(self) -> None:
        info: dict = {
            "tp_rank": 1,
            "tp_size": 4,
            "pp_rank": 0,
            "pp_size": 2,
        }
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info=info)

        assert row_data["tp"] == "1/4"
        assert row_data["pp"] == "0/2"

    def test_skips_error_info(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(
            row_data=row_data, info={"error": True, "tp_rank": 0, "tp_size": 1}
        )
        assert row_data == {}

    def test_skips_empty_info(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info={})
        assert row_data == {}

    def test_ignores_rank_without_size(self) -> None:
        row_data: dict = {}
        _extract_parallel_info(row_data=row_data, info={"tp_rank": 0})
        assert "tp" not in row_data


class TestRenderPolarsAsRichTable:
    def test_basic_dataframe_renders_table(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        table = _render_polars_as_rich_table(df)
        assert len(table.columns) == 2
        assert table.row_count == 2

    def test_empty_dataframe_returns_table_with_no_rows(self) -> None:
        df = pl.DataFrame(
            {"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)}
        )
        table = _render_polars_as_rich_table(df)
        assert len(table.columns) == 2
        assert table.row_count == 0

    def test_title_passed_to_table(self) -> None:
        df = pl.DataFrame({"a": [1]})
        table = _render_polars_as_rich_table(df, title="My Title")
        assert table.title == "My Title"

    def test_no_title_defaults_to_none(self) -> None:
        df = pl.DataFrame({"x": [1]})
        table = _render_polars_as_rich_table(df)
        assert table.title is None

    def test_column_names_match_dataframe(self) -> None:
        df = pl.DataFrame({"alpha": [1], "beta": [2], "gamma": [3]})
        table = _render_polars_as_rich_table(df)
        column_headers: list[str] = [col.header for col in table.columns]
        assert column_headers == ["alpha", "beta", "gamma"]

    def test_values_converted_to_strings(self) -> None:
        """Numeric and None values should be stringified in the rendered output."""
        df = pl.DataFrame({"num": [42], "text": ["hello"]})
        table = _render_polars_as_rich_table(df)
        rendered: str = _render_rich(table)
        assert "42" in rendered
        assert "hello" in rendered

    def test_single_column_dataframe(self) -> None:
        df = pl.DataFrame({"only_col": [10, 20, 30]})
        table = _render_polars_as_rich_table(df)
        assert len(table.columns) == 1
        assert table.row_count == 3

    def test_many_rows_all_present(self) -> None:
        """All rows from the dataframe appear in the rich table."""
        df = pl.DataFrame({"val": list(range(50))})
        table = _render_polars_as_rich_table(df)
        assert table.row_count == 50

    def test_null_values_rendered_as_string(self) -> None:
        """Null values should be converted to their string representation."""
        df = pl.DataFrame({"a": [1, None, 3]})
        table = _render_polars_as_rich_table(df)
        assert table.row_count == 3
        rendered: str = _render_rich(table)
        assert (
            "null" in rendered.lower()
            or "none" in rendered.lower()
            or "None" in rendered
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
