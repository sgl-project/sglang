import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.token_aligner.concat_steps.thd_seq_lens_loader import (
    load_thd_seq_lens_only,
)
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.aux_plugins import (
    _SGLangPlugin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


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


class TestLoadThdSeqLensOnly:
    """Tests for load_thd_seq_lens_only."""

    def test_returns_none_when_no_plugin(self, tmp_path: Path) -> None:
        """No recognized plugin → returns None."""
        fn: str = _save_pt(
            tmp_path, name="unrelated_tensor", step=0, rank=0, value=torch.tensor([1])
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is None

    def test_returns_none_when_no_cp_sharded_names(self, tmp_path: Path) -> None:
        """Plugin detected but cp_sharded_names is empty → returns None."""

        class _NoCpPlugin(_SGLangPlugin):
            @property
            def cp_sharded_names(self) -> frozenset[str]:
                return frozenset()

        fn: str = _save_pt(
            tmp_path,
            name="seq_lens",
            step=0,
            rank=0,
            value=torch.tensor([3, 5]),
            meta={"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        with patch(
            "sglang.srt.debug_utils.comparator.aligner.token_aligner.concat_steps.thd_seq_lens_loader._detect_plugin",
            return_value=_NoCpPlugin(),
        ):
            result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is None

    def test_sglang_extracts_seq_lens(self, tmp_path: Path) -> None:
        """SGLang format: seq_lens tensor present → extracts per-seq lengths."""
        fn: str = _save_pt(
            tmp_path,
            name="seq_lens",
            step=0,
            rank=0,
            value=torch.tensor([3, 5]),
            meta={"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is not None
        assert result == {0: [3, 5]}

    def test_megatron_extracts_from_cu_seqlens(self, tmp_path: Path) -> None:
        """Megatron format: cu_seqlens_q tensor → derives seq_lens via diff."""
        fn: str = _save_pt(
            tmp_path,
            name="cu_seqlens_q",
            step=0,
            rank=0,
            value=torch.tensor([0, 3, 8], dtype=torch.int64),
            meta={"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is not None
        assert result == {0: [3, 5]}

    def test_multi_step(self, tmp_path: Path) -> None:
        """Two steps with different seq_lens → returns both in result dict."""
        fn0: str = _save_pt(
            tmp_path,
            name="seq_lens",
            step=0,
            rank=0,
            value=torch.tensor([3, 5]),
            meta={"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}},
        )
        fn1: str = _save_pt(
            tmp_path,
            name="seq_lens",
            step=1,
            rank=0,
            value=torch.tensor([10, 20, 30]),
            meta={"sglang_parallel_info": {"tp_rank": 0, "tp_size": 1}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn0, fn1])

        result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is not None
        assert result == {0: [3, 5], 1: [10, 20, 30]}

    def test_returns_none_when_seq_lens_missing(self, tmp_path: Path) -> None:
        """Plugin with cp_sharded_names but no seq_lens/cu_seqlens_q tensor → None."""
        fn: str = _save_pt(
            tmp_path,
            name="cu_seqlens_kv",
            step=0,
            rank=0,
            value=torch.tensor([0, 4], dtype=torch.int64),
            meta={"megatron_parallel_info": {"cp_rank": 0, "cp_size": 2}},
        )
        df: pl.DataFrame = _make_df_from_filenames([fn])

        result = load_thd_seq_lens_only(dump_path=tmp_path, df=df)

        assert result is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
