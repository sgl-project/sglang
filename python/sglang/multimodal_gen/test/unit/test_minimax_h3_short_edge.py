"""A short edge other than 768 resolves and generates; it only warns."""

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    constants,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_BASE_SHORT_EDGE,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_MAX_PIXELS,
    minimax_h3_resolve_spatial_shape,
)


@pytest.fixture(autouse=True)
def _reset_warn_cache():
    constants.warn_unverified_short_edge.cache_clear()
    yield
    constants.warn_unverified_short_edge.cache_clear()


class TestResolveSpatialShape:
    @pytest.mark.parametrize("short_edge", [352, 384, 416, 512, 768])
    def test_a_smaller_short_edge_resolves_to_an_aligned_canvas(self, short_edge):
        shape = minimax_h3_resolve_spatial_shape(
            width=16, height=9, base_short_edge=short_edge
        )
        assert shape["width"] % MINIMAX_H3_CANVAS_MULTIPLE == 0
        assert shape["height"] % MINIMAX_H3_CANVAS_MULTIPLE == 0
        assert shape["width"] * shape["height"] <= MINIMAX_H3_MAX_PIXELS
        # the short edge lands on the requested one, up to the 32px grid
        assert (
            abs(shape["effective_short_edge"] - short_edge) < MINIMAX_H3_CANVAS_MULTIPLE
        )

    def test_halving_the_short_edge_quarters_the_canvas(self):
        full = minimax_h3_resolve_spatial_shape(width=16, height=9)
        half = minimax_h3_resolve_spatial_shape(
            width=16, height=9, base_short_edge=MINIMAX_H3_BASE_SHORT_EDGE // 2
        )
        full_pixels = full["width"] * full["height"]
        half_pixels = half["width"] * half["height"]
        assert 3.8 < full_pixels / half_pixels < 4.2

    def test_a_larger_short_edge_is_capped_by_the_pixel_budget(self):
        shape = minimax_h3_resolve_spatial_shape(
            width=16, height=9, base_short_edge=1536
        )
        assert shape["size_mode"] == "area"
        assert shape["width"] * shape["height"] <= MINIMAX_H3_MAX_PIXELS

    @pytest.mark.parametrize("bad", [0, -768, 1.5])
    def test_a_non_positive_or_fractional_short_edge_is_rejected(self, bad):
        with pytest.raises(ValueError, match="short_edge"):
            minimax_h3_resolve_spatial_shape(width=16, height=9, base_short_edge=bad)


class TestWarning:
    def test_the_recommended_short_edge_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            minimax_h3_resolve_spatial_shape(width=16, height=9)
        assert "short_edge" not in caplog.text

    def test_an_unverified_short_edge_warns_once_per_value(self, caplog):
        with caplog.at_level("WARNING"):
            for _ in range(3):
                minimax_h3_resolve_spatial_shape(
                    width=16, height=9, base_short_edge=384
                )
        assert caplog.text.count("outside the verified configuration") == 1
        assert "768" in caplog.text
