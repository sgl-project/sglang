import sys
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.display import (
    _collect_input_ids_and_positions,
    _collect_rank_info,
    _extract_parallel_info,
    _render_polars_as_text,
)
from sglang.srt.debug_utils.comparator.output_types import (
    InputIdsRecord,
    RankInfoRecord,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
