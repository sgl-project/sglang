import sys

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.dump_loader import (
    LOAD_FAILED,
    ValueWithMeta,
    _add_duplicate_index,
    _cast_to_polars_dtype,
    find_row,
    parse_meta_from_filename,
    read_meta,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestReadMeta:
    def test_basic(self, tmp_path):
        for fn in [
            "step=1___rank=0___dump_index=1___name=a.pt",
            "step=2___rank=0___dump_index=2___name=b.pt",
        ]:
            torch.save(torch.randn(5), tmp_path / fn)

        df = read_meta(str(tmp_path))
        assert len(df) == 2
        assert all(c in df.columns for c in ["step", "rank", "name"])


class TestFindRow:
    def test_single_match(self):
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "file": ["f1", "f2"]})
        assert find_row(df, {"id": 2})["file"] == "f2"

    def test_no_match(self):
        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "file": ["f1", "f2"]})
        assert find_row(df, {"id": 999}) is None

    def test_ambiguous(self):
        df = pl.DataFrame({"id": [1, 1], "file": ["f1", "f2"]})
        assert find_row(df, {"id": 1}) is None


class TestCastToPolars:
    def test_int(self):
        assert _cast_to_polars_dtype("42", pl.Int64) == 42

    def test_float(self):
        assert _cast_to_polars_dtype("3.14", pl.Float64) == pytest.approx(3.14)


class TestAddDuplicateIndex:
    def test_basic(self):
        df = pl.DataFrame(
            {
                "name": ["a", "a", "b"],
                "dump_index": [1, 2, 3],
                "filename": ["f1", "f2", "f3"],
            }
        )
        result = _add_duplicate_index(df)
        assert result.filter(pl.col("name") == "a").sort("dump_index")[
            "duplicate_index"
        ].to_list() == [0, 1]


class TestValueWithMeta:
    def test_load_dict_format(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=hidden.pt"
        tensor = torch.randn(4, 8)
        torch.save({"value": tensor, "meta": {"custom": "field"}}, path)

        loaded = ValueWithMeta.load(path)
        assert torch.allclose(loaded.value, tensor)
        assert loaded.meta["custom"] == "field"
        assert loaded.meta["name"] == "hidden"
        assert loaded.meta["rank"] == 0

    def test_load_bare_tensor(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=bare.pt"
        tensor = torch.randn(3, 3)
        torch.save(tensor, path)

        loaded = ValueWithMeta.load(path)
        assert torch.allclose(loaded.value, tensor)
        assert loaded.meta["name"] == "bare"

    def test_load_corrupted_file(self, tmp_path) -> None:
        path = tmp_path / "step=0___rank=0___dump_index=1___name=bad.pt"
        path.write_text("not a valid pt file")

        loaded = ValueWithMeta.load(path)
        assert loaded.value is LOAD_FAILED
        assert loaded.meta["name"] == "bad"


class TestRecomputeStatusParsing:
    def test_parse_recompute_status_from_filename(self) -> None:
        from pathlib import Path

        meta_disabled = parse_meta_from_filename(
            Path(
                "step=0___rank=0___dump_index=1___name=x___recompute_status=disabled.pt"
            )
        )
        assert meta_disabled["recompute_status"] == "disabled"

        meta_recompute = parse_meta_from_filename(
            Path(
                "step=0___rank=0___dump_index=1___name=x___recompute_status=recompute.pt"
            )
        )
        assert meta_recompute["recompute_status"] == "recompute"

        meta_original = parse_meta_from_filename(
            Path(
                "step=0___rank=0___dump_index=1___name=x___recompute_status=original.pt"
            )
        )
        assert meta_original["recompute_status"] == "original"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
