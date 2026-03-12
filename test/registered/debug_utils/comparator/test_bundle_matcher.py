import sys
from typing import Any

import polars as pl
import pytest

from sglang.srt.debug_utils.comparator.bundle_matcher import (
    TensorBundleInfo,
    TensorFileInfo,
    _rows_to_tensor_infos,
    match_bundles,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _make_row(
    *,
    name: str,
    step: int = 0,
    rank: int = 0,
    layer_id: int | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    if filename is None:
        layer_part: str = f"___layer_id={layer_id}" if layer_id is not None else ""
        filename = f"name={name}___step={step}___rank={rank}{layer_part}.pt"
    row: dict[str, Any] = {
        "name": name,
        "step": step,
        "rank": rank,
        "filename": filename,
    }
    if layer_id is not None:
        row["layer_id"] = layer_id
    return row


def _make_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(rows)

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    normalized: list[dict[str, Any]] = [
        {k: row.get(k, None) for k in all_keys} for row in rows
    ]
    return pl.DataFrame(normalized)


class TestMatchBundles:
    def test_single_tensor_single_step(self) -> None:
        target_df: pl.DataFrame = _make_df([_make_row(name="t_a")])
        baseline_df: pl.DataFrame = _make_df([_make_row(name="t_a")])

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 1
        assert len(results[0].x) == 1
        assert len(results[0].y) == 1
        assert results[0].y[0].name == "t_a"

    def test_multiple_names_separate_bundles(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_b"),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_b"),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 2
        result_names: list[str] = [r.y[0].name for r in results]
        assert "t_a" in result_names
        assert "t_b" in result_names

    def test_skip_rank_groups_across_ranks(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", rank=0),
                _make_row(name="t_a", rank=1),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", rank=0),
                _make_row(name="t_a", rank=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename", "rank"},
        )

        assert len(results) == 1
        assert len(results[0].y) == 2

    def test_baseline_missing_tensor(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
                _make_row(name="t_extra"),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a"),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert len(results) == 2
        extra_pair: Pair[TensorBundleInfo] = [
            r for r in results if r.y[0].name == "t_extra"
        ][0]
        assert extra_pair.x == []

    def test_empty_target_returns_empty(self) -> None:
        target_df: pl.DataFrame = _make_df([])
        baseline_df: pl.DataFrame = _make_df([_make_row(name="t_a")])

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename"},
        )

        assert results == []

    def test_skip_step_groups_across_steps(self) -> None:
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", step=0),
                _make_row(name="t_a", step=1),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="t_a", step=0),
                _make_row(name="t_a", step=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys={"filename", "step"},
        )

        assert len(results) == 1
        assert len(results[0].y) == 2


class TestMatchBundlesPipelineParallel:
    """Tests verifying that PP works correctly with the existing matching logic."""

    LOGICAL_SKIP_KEYS: set[str] = {"filename", "rank", "dump_index", "recompute_status"}

    def test_same_layer_id_different_ranks_match(self) -> None:
        """SGLang PP=2 rank 0 (layers 0-31) vs Megatron PP=4 rank 2 (layers 16-31):
        layer_id=20 should match regardless of world rank."""
        target_df: pl.DataFrame = _make_df(
            [_make_row(name="hidden", rank=0, layer_id=20)]
        )
        baseline_df: pl.DataFrame = _make_df(
            [_make_row(name="hidden", rank=2, layer_id=20)]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 1
        assert len(results[0].x) == 1
        assert len(results[0].y) == 1

    def test_layer_id_none_non_layer_tensors_match(self) -> None:
        """Non-layer tensors (embedding, lm_head) have no layer_id.
        They should match across different PP ranks."""
        target_df: pl.DataFrame = _make_df([_make_row(name="embed_tokens", rank=0)])
        baseline_df: pl.DataFrame = _make_df([_make_row(name="embed_tokens", rank=0)])

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 1
        assert len(results[0].x) == 1
        assert len(results[0].y) == 1

    def test_different_pp_sizes_layer_and_non_layer_bundles(self) -> None:
        """SGLang PP=2 TP=2 (4 ranks) vs Megatron PP=4 TP=2 (8 ranks).
        Layer tensors match by (name, layer_id); non-layer tensors match by name.
        All ranks are grouped into the same bundle when rank is skipped."""
        target_df: pl.DataFrame = _make_df(
            [
                # SGLang: pp_stage=0 has ranks 0,1 (TP=2)
                _make_row(name="hidden", rank=0, layer_id=20),
                _make_row(name="hidden", rank=1, layer_id=20),
                # SGLang: embedding on pp_stage=0
                _make_row(name="embed_tokens", rank=0),
                _make_row(name="embed_tokens", rank=1),
                # SGLang: lm_head on pp_stage=1, ranks 2,3
                _make_row(name="lm_head", rank=2),
                _make_row(name="lm_head", rank=3),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                # Megatron: pp_stage=1 has ranks 2,3 for layer 20
                _make_row(name="hidden", rank=2, layer_id=20),
                _make_row(name="hidden", rank=3, layer_id=20),
                # Megatron: embedding on pp_stage=0
                _make_row(name="embed_tokens", rank=0),
                _make_row(name="embed_tokens", rank=1),
                # Megatron: lm_head on pp_stage=3, ranks 6,7
                _make_row(name="lm_head", rank=6),
                _make_row(name="lm_head", rank=7),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 3
        names_to_pairs: dict[str, Pair[TensorBundleInfo]] = {}
        for pair in results:
            key: str = pair.y[0].name
            layer_suffix: str = ""
            if "layer_id" in target_df.columns:
                row_match = [
                    r
                    for r in target_df.to_dicts()
                    if r["filename"] == pair.y[0].filename
                ]
                if row_match and row_match[0].get("layer_id") is not None:
                    layer_suffix = f"_{row_match[0]['layer_id']}"
            names_to_pairs[key + layer_suffix] = pair

        assert len(names_to_pairs["hidden_20"].x) == 2
        assert len(names_to_pairs["hidden_20"].y) == 2
        assert len(names_to_pairs["embed_tokens"].x) == 2
        assert len(names_to_pairs["embed_tokens"].y) == 2
        assert len(names_to_pairs["lm_head"].x) == 2
        assert len(names_to_pairs["lm_head"].y) == 2

    def test_unmatched_layer_id_creates_empty_baseline(self) -> None:
        """If target has a layer_id that baseline doesn't, the baseline side
        should be empty (not incorrectly matched to a different layer)."""
        target_df: pl.DataFrame = _make_df(
            [
                _make_row(name="hidden", rank=0, layer_id=10),
                _make_row(name="hidden", rank=0, layer_id=20),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                _make_row(name="hidden", rank=0, layer_id=10),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 2
        matched: list[Pair[TensorBundleInfo]] = [r for r in results if r.x]
        unmatched: list[Pair[TensorBundleInfo]] = [r for r in results if not r.x]
        assert len(matched) == 1
        assert len(unmatched) == 1

    def test_pp1_vs_pp_gt1_matches_by_layer_id(self) -> None:
        """PP=1 (all layers on 1 rank) vs PP>1 (layers split across ranks).
        Should match correctly by layer_id regardless of rank."""
        target_df: pl.DataFrame = _make_df(
            [
                # PP=1: all on rank 0
                _make_row(name="hidden", rank=0, layer_id=0),
                _make_row(name="hidden", rank=0, layer_id=1),
            ]
        )
        baseline_df: pl.DataFrame = _make_df(
            [
                # PP=2: layer 0 on rank 0, layer 1 on rank 1
                _make_row(name="hidden", rank=0, layer_id=0),
                _make_row(name="hidden", rank=1, layer_id=1),
            ]
        )

        results: list[Pair[TensorBundleInfo]] = match_bundles(
            dfs=Pair(x=baseline_df, y=target_df),
            skip_keys=self.LOGICAL_SKIP_KEYS,
        )

        assert len(results) == 2
        for pair in results:
            assert len(pair.x) == 1
            assert len(pair.y) == 1


class TestRowsToTensorInfos:
    def test_filters_extra_columns(self) -> None:
        rows: list[dict[str, Any]] = [
            {"filename": "a.pt", "name": "t_a", "step": 0, "rank": 7}
        ]
        infos: list[TensorFileInfo] = _rows_to_tensor_infos(rows)

        assert len(infos) == 1
        assert infos[0] == TensorFileInfo(filename="a.pt", name="t_a", step=0)

    def test_empty_rows(self) -> None:
        infos: list[TensorFileInfo] = _rows_to_tensor_infos([])
        assert infos == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
