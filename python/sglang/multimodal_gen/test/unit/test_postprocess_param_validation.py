# SPDX-License-Identifier: Apache-2.0
"""Post-processing controls are client-supplied and size the output."""

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_interpolation_scale", 0.0),
        ("frame_interpolation_scale", -1.0),
        ("frame_interpolation_scale", float("nan")),
        ("frame_interpolation_scale", float("inf")),
        ("frame_interpolation_scale", 100.0),
        ("upscaling_scale", 0),
        ("upscaling_scale", -3),
        ("upscaling_scale", 1000),
        ("upscaling_scale", True),
    ],
)
def test_out_of_range_postprocess_controls_are_rejected(field, value):
    """upscaling_scale sizes the output and scale feeds RIFE; values outside the
    supported range must be rejected, not reach the kernels."""
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})._validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_interpolation_scale": 0.5},
        {"frame_interpolation_scale": 4.0},
        {"upscaling_scale": 1},
        {"upscaling_scale": 8},
    ],
)
def test_supported_postprocess_controls_are_accepted(kwargs):
    SamplingParams(**kwargs)._validate()
