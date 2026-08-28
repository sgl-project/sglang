from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
)


def test_video_api_forwards_profiling_options():
    request = VideoGenerationsRequest(
        prompt="profile this request",
        task="t2va",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        profile=True,
        num_profiled_timesteps=3,
        profile_all_stages=False,
        quality="high",
    )
    server_args = SimpleNamespace(
        backend="auto",
        model_id=None,
        model_path="MiniMaxAI/MiniMax-H3",
        pipeline_class_name="MiniMaxH3Pipeline",
        pipeline_config=object(),
    )

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api."
            "build_sampling_params",
            side_effect=lambda request_id, **kwargs: kwargs,
        ),
    ):
        kwargs = _build_video_sampling_params("profile-request", request)

    assert kwargs["profile"] is True
    assert kwargs["num_profiled_timesteps"] == 3
    assert kwargs["profile_all_stages"] is False
    assert kwargs["quality"] == "high"


def test_realtime_video_api_forwards_sampling_quality():
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="profile this realtime request",
        first_frame="cat.png",
        quality="high",
    )
    chunk_inputs = RealtimeChunkInputs(prompt=request.prompt)

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime."
        "realtime_adapter.build_sampling_params",
        side_effect=lambda request_id, **kwargs: kwargs,
    ):
        kwargs = build_realtime_sampling_params(
            "realtime-profile-request",
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=9,
            num_inference_steps=4,
            chunk_size=9,
        )

    assert kwargs["quality"] == "high"
