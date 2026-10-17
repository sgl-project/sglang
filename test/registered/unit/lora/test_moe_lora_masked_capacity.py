"""Masked capacity and packed-schedule tile selection contracts."""

from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import MAX_TOKEN_CLUSTERS
from sglang.srt.lora.moe.base_gemm_provider import gemm_config_store
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_common import CuteDslTileMixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import masked_m_max
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("alignment", [8, 16, 64, 128])
@pytest.mark.parametrize(
    "num_tokens",
    [0, 1, 7, 8, 9, 63, 64, 65, 84, 127, 128, 129, 255, 256, 257, 320, 512],
)
def test_masked_capacity_is_minimal_positive_alignment(num_tokens, alignment):
    capacity = masked_m_max(num_tokens, alignment)
    assert capacity >= num_tokens
    assert capacity > 0
    assert capacity % alignment == 0
    assert capacity - alignment < max(1, num_tokens)
    if num_tokens and num_tokens % alignment == 0:
        assert capacity == num_tokens


def test_bf16_default_alignment_and_full_input_row_bound():
    assert masked_m_max(0) == 8
    assert masked_m_max(8) == 8
    # Every input row can select the same expert, regardless of mean occupancy.
    assert masked_m_max(257) == 264


def _tiles(*, wide_threshold=16, widths=(8, 64, 128), xwide_by_heuristic=True):
    """A bare mixin with the attributes `_init_tiles` gives a provider."""
    provider = CuteDslTileMixin()
    provider.WIDE_EXPECTED_M_THRESHOLD = wide_threshold
    provider._compiled = {width: {} for width in widths}
    provider._config_table = None
    provider._max_token_clusters = MAX_TOKEN_CLUSTERS
    # `_init_tiles` turns this off for FP8 on SM90.
    provider._xwide_by_heuristic = xwide_by_heuristic
    return provider


@pytest.mark.parametrize("xwide_by_heuristic", [True, False])
@pytest.mark.parametrize("wide_threshold", [16, 32])
def test_normal_capacity_preserves_heuristic_boundaries(
    wide_threshold, xwide_by_heuristic
):
    provider = _tiles(
        wide_threshold=wide_threshold, xwide_by_heuristic=xwide_by_heuristic
    )
    # Each transition once, with tiny and non-tiny capacities. Packing limits
    # are exercised separately below; the token count does not choose a tile.
    for expected_m in (1, wide_threshold - 1, wide_threshold, 95, 96):
        expected = (
            128
            if expected_m == 96 and xwide_by_heuristic
            else 64
            if expected_m >= wide_threshold
            else 8
        )
        for num_tokens in (0, 4096):
            assert (
                provider._token_width_for(masked_m_max(num_tokens), expected_m)
                == expected
            )
            assert provider._token_width_for(num_tokens, expected_m) == expected


@pytest.mark.parametrize("xwide_by_heuristic", [True, False])
@pytest.mark.parametrize(
    "path",
    sorted(
        Path(gemm_config_store._PACKAGE_CONFIG_DIR)
        .joinpath("base_gemm")
        .glob("provider=cutedsl_*_masked,*.json")
    ),
    ids=lambda path: path.name,
)
def test_normal_masked_bounds_preserve_shipped_table_choice(path, xwide_by_heuristic):
    table = msgspec.json.decode(
        path.read_bytes(), type=gemm_config_store.GemmConfigTable
    )
    provider = _tiles(
        widths={8, 64, 128} | {tile.token_width for tile in table.tiles},
        xwide_by_heuristic=xwide_by_heuristic,
    )
    provider._config_table = table
    boundaries = {
        1,
        *(max(1, bucket + delta) for bucket in table.buckets for delta in (-1, 0, 1)),
    }
    for expected_m in sorted(boundaries):
        width = table.pick(expected_m)["token_width"]
        for num_tokens in (0, 512, 131072):
            assert provider._token_width_for(num_tokens, expected_m) == width
            assert (
                provider._token_width_for(masked_m_max(num_tokens), expected_m) == width
            )


@pytest.mark.parametrize("xwide_by_heuristic", [True, False])
def test_table_choice_wins_over_the_heuristic_flag(xwide_by_heuristic):
    provider = _tiles(xwide_by_heuristic=xwide_by_heuristic)
    provider._config_table = SimpleNamespace(
        pick=lambda expected_m: {"token_width": 128}
    )
    assert provider._token_width_for(8, 1) == 128
    assert provider._token_width_for(8, 96) == 128
    provider._config_table = SimpleNamespace(
        pick=lambda expected_m: {"token_width": 64}
    )
    assert provider._token_width_for(8, 96) == 64


@pytest.mark.parametrize("xwide_by_heuristic", [True, False])
def test_capacity_widens_past_the_heuristic_choice(xwide_by_heuristic):
    provider = _tiles(xwide_by_heuristic=xwide_by_heuristic)
    limit = 64 * MAX_TOKEN_CLUSTERS
    assert provider._token_width_for(limit, 16) == 64
    # One row past the 64-wide packing widens to 128 whatever the flag says.
    assert provider._token_width_for(limit + 1, 16) == 128
    assert provider._token_width_for(limit + 1, 96) == 128


def test_masked_tile_selection_preserves_table_and_compiled_width_choices():
    provider = _tiles(widths=(64, 128))
    assert provider._token_width_for(8, 1) == 64
    provider._config_table = SimpleNamespace(
        pick=lambda expected_m: {"token_width": 128 if expected_m == 1 else 64}
    )
    assert provider._token_width_for(8, 1) == 128
    assert provider._token_width_for(8, 96) == 64
    assert provider._token_width_for(64 * MAX_TOKEN_CLUSTERS + 1, 96) == 128


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("widths", [(8, 64, 128), (8, 16, 64, 128), (64, 128)])
def test_real_masked_packing_boundaries(dtype, widths):
    provider = _tiles(widths=widths)
    for index, width in enumerate(widths):
        limit = width * MAX_TOKEN_CLUSTERS
        for num_tokens in (limit - 1, limit):
            rows = masked_m_max(num_tokens) if dtype == "bf16" else num_tokens
            selected = provider._token_width_for(rows, 1)
            assert selected == width
            capacity = masked_m_max(num_tokens, 8 if dtype == "bf16" else selected)
            assert capacity <= selected * MAX_TOKEN_CLUSTERS

        rows = masked_m_max(limit + 1) if dtype == "bf16" else limit + 1
        if index + 1 < len(widths):
            selected = provider._token_width_for(rows, 1)
            assert selected == widths[index + 1]
            capacity = masked_m_max(limit + 1, 8 if dtype == "bf16" else selected)
            assert capacity <= selected * MAX_TOKEN_CLUSTERS
        else:
            with pytest.raises(ValueError, match=f"max_expert_rows={rows} exceeds"):
                provider._token_width_for(rows, 1)


def test_table_buckets_must_select_declared_widths():
    defaults = ((8, 128), (64, 128), (128, 128))
    provider = _tiles()
    provider.contract = SimpleNamespace(key="cutedsl_fp8_masked")
    tiles = (gemm_config_store.GemmTile(token_width=16, persistent_clusters=132),)
    provider._config_table = gemm_config_store.GemmConfigTable(
        buckets={4: {"token_width": 16}, 96: {"token_width": 128}}, tiles=tiles
    )
    assert provider._merge_table_tiles(defaults) == (
        (8, 128),
        (16, 132),
        (64, 128),
        (128, 128),
    )
    # A bucket naming a width no tile declares falls back to the heuristics.
    provider._config_table = gemm_config_store.GemmConfigTable(
        buckets={4: {"token_width": 256}}
    )
    assert provider._merge_table_tiles(defaults) == defaults
    assert provider._config_table is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
