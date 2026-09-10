# SPDX-License-Identifier: Apache-2.0
"""target.width/height names the canvas exactly, instead of a policy short edge.

A caller with a fixed delivery geometry can state it; the request boundary
rewrites it into the short_edge + aspect_ratio that reproduce it, so nothing
downstream changes.
"""

import msgspec
import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    _normalize_exact_canvas,
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_MAX_PIXELS,
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    minimax_h3_task_profile,
)


def _validate(target, *, task="t2va", conditions=()):
    return minimax_h3_validate_canonical_request(
        task=task,
        prompt="exact canvas",
        conditions=list(conditions),
        target=target,
        seed=0,
    )


class TestAcceptedCanvas:
    def test_an_exact_canvas_becomes_the_edge_and_ratio_it_denotes(self):
        canonical = _validate({"width": 864, "height": 480, "duration_seconds": 5.0})
        assert canonical["target"] == {
            "short_edge": 480,
            "aspect_ratio": "864:480",
            "duration_seconds": 5.0,
        }

    def test_the_resolved_plan_renders_the_requested_canvas(self):
        plan = minimax_h3_resolve_plan(
            _validate({"width": 864, "height": 480, "duration_seconds": 5.0})
        )
        assert (plan.shape["width"], plan.shape["height"]) == (864, 480)

    def test_a_canvas_is_indistinguishable_from_the_ratio_that_resolves_to_it(self):
        # 480 + 16:9 already resolves to 864x480, and the resolved shape does not
        # carry the ratio string. Nothing downstream can tell the two spellings
        # apart, which is why no stage needs to learn about width/height.
        by_canvas = minimax_h3_resolve_plan(
            _validate({"width": 864, "height": 480, "duration_seconds": 5.0})
        )
        by_ratio = minimax_h3_resolve_plan(
            _validate(
                {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5.0}
            )
        )
        assert by_canvas.shape == by_ratio.shape

    def test_the_flagship_canvas_sits_exactly_on_the_pixel_budget(self):
        assert 1344 * 768 == MINIMAX_H3_MAX_PIXELS
        plan = minimax_h3_resolve_plan(
            _validate({"width": 1344, "height": 768, "duration_seconds": 5.0})
        )
        assert (plan.shape["width"], plan.shape["height"]) == (1344, 768)

    def test_a_canvas_is_exempt_from_the_finite_ratio_allowlist(self):
        # 864:480 is 9:5, which the ratio allowlist does not carry. Naming both
        # axes is a choice, not a guess, so it is admitted -- while the same
        # geometry spelled as a ratio still is not.
        assert _validate({"width": 864, "height": 480, "duration_seconds": 5.0})
        with pytest.raises(ValueError, match="aspect_ratio"):
            _validate(
                {"short_edge": 480, "aspect_ratio": "9:5", "duration_seconds": 5.0}
            )

    def test_fl2va_may_name_the_canvas_its_keyframe_was_prepared_for(self):
        canonical = _validate(
            {"width": 864, "height": 480, "duration_seconds": 5.0},
            task="fl2va",
            conditions=[
                {
                    "type": "image",
                    "uri": "first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
        )
        assert canonical["target"]["aspect_ratio"] == "864:480"


class TestRejectedCanvas:
    def test_naming_both_spellings_is_refused(self):
        with pytest.raises(ValueError, match="not both"):
            _validate(
                {
                    "width": 864,
                    "height": 480,
                    "short_edge": 480,
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5.0,
                }
            )

    def test_half_a_pair_names_no_canvas(self):
        with pytest.raises(ValueError, match=r"target\.height must be an integer"):
            _validate({"width": 864, "duration_seconds": 5.0})

    @pytest.mark.parametrize("width", [864.0, "864", True])
    def test_a_non_integer_axis_is_refused(self, width):
        with pytest.raises(ValueError, match=r"target\.width must be an integer"):
            _validate({"width": width, "height": 480, "duration_seconds": 5.0})

    def test_an_off_grid_canvas_is_refused_rather_than_rounded(self):
        with pytest.raises(ValueError, match=r"target\.width must be a positive"):
            _validate({"width": 850, "height": 480, "duration_seconds": 5.0})

    def test_an_over_budget_canvas_is_refused_rather_than_scaled(self):
        with pytest.raises(ValueError, match=r"target\.width\*height"):
            _validate({"width": 1376, "height": 768, "duration_seconds": 5.0})

    def test_a_canvas_outside_the_supported_ratio_range_is_refused(self):
        with pytest.raises(ValueError, match=r"target\.width/height: .*1:4 to 4:1"):
            _validate({"width": 1024, "height": 128, "duration_seconds": 5.0})


class TestForcedAutoGeometry:
    def test_a_canvas_cannot_override_a_task_that_forces_auto(self):
        # No shipped task forces "auto" today, but the rewrite ends in a concrete
        # aspect_ratio: without this guard it would be a way around the
        # forced-"auto" check rather than another spelling of an allowed ratio.
        profile = msgspec.structs.replace(
            minimax_h3_task_profile("t2va"), aspect_ratio_forced_auto=True
        )
        with pytest.raises(ValueError, match="not caller-chosen"):
            _normalize_exact_canvas(
                {"width": 864, "height": 480}, path="target", profile=profile
            )


class TestOfflineProjection:
    def test_the_sampling_params_projection_keeps_both_axes(self):
        # The offline projection drops unknown target keys, so width/height have
        # to be declared there too or validation never sees them.
        from sglang.multimodal_gen.configs.sample.minimax_h3 import (
            MiniMaxH3SamplingParams,
        )

        params = MiniMaxH3SamplingParams(
            prompt="exact canvas",
            task="t2va",
            conditions=[],
            target={
                "width": 864,
                "height": 480,
                "duration_seconds": 5.0,
                "ignored_transport_key": "x",
            },
        )
        assert params.target == {"width": 864, "height": 480, "duration_seconds": 5.0}
        canonical = params.build_request_extra()["minimax_h3_canonical_request"]
        assert canonical["target"]["aspect_ratio"] == "864:480"
