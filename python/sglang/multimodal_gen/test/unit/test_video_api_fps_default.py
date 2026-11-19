# SPDX-License-Identifier: Apache-2.0
"""/v1/videos falls back to the model's own frame rate, not a fixed 24 fps."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)


def _build(monkeypatch, sampling_params_cls, **request_fields):
    monkeypatch.setattr(video_api, "get_global_server_args", lambda: SimpleNamespace())
    monkeypatch.setattr(
        video_api, "resolve_sampling_params_cls", lambda _: sampling_params_cls
    )
    monkeypatch.setattr(
        video_api, "build_sampling_params", lambda request_id, **kwargs: kwargs
    )
    request = VideoGenerationsRequest(prompt="a cat", **request_fields)
    return video_api._build_video_sampling_params("fps-default", request)


@pytest.mark.parametrize(
    "sampling_params_cls,request_fields,fps,num_frames",
    [
        # Wan is a 16 fps model: 4 s -> 64 frames at 16 fps.
        (WanT2V_1_3B_SamplingParams, {}, 16, 64),
        (WanT2V_1_3B_SamplingParams, {"seconds": 5}, 16, 80),
        # An explicit request still wins.
        (WanT2V_1_3B_SamplingParams, {"fps": 24}, 24, 96),
        (WanT2V_1_3B_SamplingParams, {"num_frames": 81}, 16, 81),
        # Models without their own rate keep the generic 24 fps.
        (SamplingParams, {}, 24, 96),
    ],
)
def test_video_fps_defaults_to_model_rate(
    monkeypatch, sampling_params_cls, request_fields, fps, num_frames
):
    kwargs = _build(monkeypatch, sampling_params_cls, **request_fields)
    assert kwargs["fps"] == fps
    assert kwargs["num_frames"] == num_frames
