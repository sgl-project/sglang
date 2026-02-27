import sys
from pathlib import Path

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader import (
    _detect_plugin,
    _ensure_dims_in_metas,
    _load_and_align_aux_tensor,
    _load_non_tensor_aux,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_plugins import (
    _MegatronPlugin,
    _SGLangPlugin,
)
from sglang.srt.debug_utils.comparator.output_types import GeneralWarning
from sglang.srt.debug_utils.comparator.warning_sink import WarningSink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)

_sglang_plugin = _SGLangPlugin()
_megatron_plugin = _MegatronPlugin()


def _save_pt(
    dump_path: Path,
    *,
    name: str,
    step: int,
    rank: int,
    value: object,
    meta: dict | None = None,
) -> str:
    filename: str = f"name={name}___step={step}___rank={rank}.pt"
    payload: dict = {"value": value, "meta": meta or {}}
    torch.save(payload, dump_path / filename)
    return filename


def _make_df_from_filenames(filenames: list[str]) -> pl.DataFrame:
    rows: list[dict] = []
    for fn in filenames:
        parts: dict = {}
        stem: str = fn.removesuffix(".pt")
        for kv in stem.split("___"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k] = v
        rows.append(
            {
                "filename": fn,
                "name": parts["name"],
                "step": int(parts["step"]),
                "rank": int(parts["rank"]),
            }
        )
    return pl.DataFrame(rows)


class TestEnsureDimsInMetas:
    """Tests for _ensure_dims_in_metas."""

    def _make_meta(self, *, cp_size: int = 1, cp_rank: int = 0) -> dict:
        return {
            "sglang_parallel_info": {
                "tp_rank": 0,
                "tp_size": 1,
                "cp_rank": cp_rank,
                "cp_size": cp_size,
            }
        }

    def test_no_cp_returns_metas_unchanged(self):
        """Without CP parallelism, metas are returned as-is."""
        metas: list[dict] = [self._make_meta(cp_size=1)]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_sglang_plugin, metas=metas, ndim=1
        )
        assert result is metas

    def test_dims_already_present_returns_metas_unchanged(self):
        """If dims is already in meta, metas are returned as-is."""
        metas: list[dict] = [{**self._make_meta(cp_size=2, cp_rank=0), "dims": "t"}]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_sglang_plugin, metas=metas, ndim=1
        )
        assert result is metas

    def test_cp_sharded_sglang_input_ids_infers_dims(self):
        """CP + input_ids in sglang infers dims 't(cp,zigzag)'."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_sglang_plugin, metas=metas, ndim=1
        )
        assert result is not metas
        assert result[0]["dims"] == "t(cp,zigzag)"
        assert result[1]["dims"] == "t(cp,zigzag)"

    def test_cp_sharded_sglang_positions_infers_dims(self):
        """CP + positions in sglang infers dims 't(cp,zigzag)'."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(
            name="positions", plugin=_sglang_plugin, metas=metas, ndim=1
        )
        assert result[0]["dims"] == "t(cp,zigzag)"

    def test_cp_sharded_megatron_input_ids_infers_dims_1d(self):
        """CP + input_ids in megatron (1D) infers dims 't(cp,zigzag)'."""
        metas: list[dict] = [
            {"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
            {"megatron_parallel_info": {"cp_rank": 1, "cp_size": 2}},
        ]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_megatron_plugin, metas=metas, ndim=1
        )
        assert result[0]["dims"] == "t(cp,zigzag)"

    def test_cp_sharded_megatron_input_ids_infers_dims_2d(self):
        """CP + input_ids in megatron (2D) infers dims 'b s(cp,zigzag)'."""
        metas: list[dict] = [
            {"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
            {"megatron_parallel_info": {"cp_rank": 1, "cp_size": 2}},
        ]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_megatron_plugin, metas=metas, ndim=2
        )
        assert result[0]["dims"] == "b s(cp,zigzag)"

    def test_cp_non_sharded_name_returns_metas_unchanged(self):
        """CP + non-sharded tensor name (seq_lens) returns metas as-is."""
        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(
            name="seq_lens", plugin=_sglang_plugin, metas=metas, ndim=1
        )
        assert result is metas

    def test_unknown_plugin_returns_metas_unchanged(self):
        """CP + plugin with empty cp_sharded_names returns metas as-is."""

        class _DummyPlugin(_SGLangPlugin):
            @property
            def cp_sharded_names(self) -> frozenset[str]:
                return frozenset()

        metas: list[dict] = [
            self._make_meta(cp_size=2, cp_rank=0),
            self._make_meta(cp_size=2, cp_rank=1),
        ]
        result = _ensure_dims_in_metas(
            name="input_ids", plugin=_DummyPlugin(), metas=metas, ndim=1
        )
        assert result is metas


class TestDetectPlugin:
    def test_discriminating_names_sglang(self, tmp_path: Path) -> None:
        fn: str = _save_pt(
            tmp_path, name="seq_lens", step=0, rank=0, value=torch.tensor([3])
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = _detect_plugin(df, dump_path=tmp_path)

        assert result is not None
        assert result.name == "sglang"

    def test_fallback_to_meta_based_detection(self, tmp_path: Path) -> None:
        fn: str = _save_pt(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            value=torch.tensor([1, 2, 3]),
            meta={"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = _detect_plugin(df, dump_path=tmp_path)

        assert result is not None
        assert result.name == "sglang"

    def test_returns_none_no_match(self, tmp_path: Path) -> None:
        fn: str = _save_pt(
            tmp_path, name="unrelated_tensor", step=0, rank=0, value=torch.tensor([1])
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = _detect_plugin(df, dump_path=tmp_path)

        assert result is None


class TestLoadNonTensorAux:
    def test_multi_rank_mismatch_warning(self, tmp_path: Path) -> None:
        fn0: str = _save_pt(tmp_path, name="rids", step=0, rank=0, value=["req_A"])
        fn1: str = _save_pt(tmp_path, name="rids", step=0, rank=1, value=["req_B"])
        df: pl.DataFrame = _make_df_from_filenames([fn0, fn1])

        sink = WarningSink()
        with sink.context() as warnings:
            from unittest.mock import patch

            with patch(
                "sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader.warning_sink",
                sink,
            ):
                result = _load_non_tensor_aux(
                    name="rids", step=0, df=df, dump_path=tmp_path
                )

        assert result == ["req_A"]
        assert len(warnings) == 1
        assert isinstance(warnings[0], GeneralWarning)
        assert "rids_mismatch" in warnings[0].category

    def test_no_rows_returns_none(self, tmp_path: Path) -> None:
        df: pl.DataFrame = _make_df_from_filenames([])

        result = _load_non_tensor_aux(name="rids", step=0, df=df, dump_path=tmp_path)

        assert result is None


class TestLoadAndAlignAuxTensor:
    def test_multi_rank_no_dims_emits_warning(self, tmp_path: Path) -> None:
        fn0: str = _save_pt(
            tmp_path,
            name="input_ids",
            step=0,
            rank=0,
            value=torch.tensor([1, 2, 3]),
            meta={
                "sglang_parallel_info": {
                    "tp_rank": 0,
                    "tp_size": 2,
                    "cp_rank": 0,
                    "cp_size": 1,
                }
            },
        )
        fn1: str = _save_pt(
            tmp_path,
            name="input_ids",
            step=0,
            rank=1,
            value=torch.tensor([4, 5, 6]),
            meta={
                "sglang_parallel_info": {
                    "tp_rank": 1,
                    "tp_size": 2,
                    "cp_rank": 0,
                    "cp_size": 1,
                }
            },
        )
        df: pl.DataFrame = _make_df_from_filenames([fn0, fn1])

        sink = WarningSink()
        with sink.context() as warnings:
            from unittest.mock import patch

            with patch(
                "sglang.srt.debug_utils.comparator.aligner.token_aligner.aux_loader.warning_sink",
                sink,
            ):
                result = _load_and_align_aux_tensor(
                    name="input_ids",
                    step=0,
                    df=df,
                    dump_path=tmp_path,
                    plugin=_sglang_plugin,
                )

        assert result is not None
        assert torch.equal(result, torch.tensor([1, 2, 3]))
        assert len(warnings) == 1
        assert isinstance(warnings[0], GeneralWarning)
        assert "aux_no_dims" in warnings[0].category


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
