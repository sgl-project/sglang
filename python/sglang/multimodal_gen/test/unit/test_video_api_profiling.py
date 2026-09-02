from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
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
    _video_request_model_kwargs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


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


def test_ltx25_video_extensions_remain_model_specific():
    field_values = {
        "use_diffusion_decoder": True,
        "auto_duration": True,
        "auto_duration_min_seconds": 2.0,
        "auto_duration_max_seconds": 8.0,
    }
    request = VideoGenerationsRequest(
        prompt="a fox in snow",
        extra_body=field_values,
    )
    base_fields = {field.name for field in fields(SamplingParams)}
    ltx25_fields = {field.name for field in fields(LTX25SamplingParams)}

    for field_name in field_values:
        assert field_name not in VideoGenerationsRequest.model_fields
        assert field_name not in base_fields
        assert field_name in ltx25_fields

    assert _video_request_model_kwargs(request, LTX25SamplingParams) == field_values
    assert _video_request_model_kwargs(request, SamplingParams) == {}


def test_ltx23_request_defaults_to_vae_decoder():
    request = Req(sampling_params=LTX23SamplingParams())

    assert request.use_diffusion_decoder is False


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
