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


@pytest.mark.parametrize(
    "field", ["upscaling_model_path", "frame_interpolation_model_path"]
)
@pytest.mark.parametrize(
    "value", ["http://169.254.169.254/latest/x.pth", "https://internal.svc/w.pth"]
)
def test_url_model_paths_are_rejected_by_validate(field, value):
    """Every path into SamplingParams, not just the HTTP API, must reject URLs."""
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})._validate()


def test_local_model_paths_are_accepted_by_validate():
    SamplingParams(
        upscaling_model_path="/models/realesrgan.pth",
        frame_interpolation_model_path="org/rife-repo",
    )._validate()
