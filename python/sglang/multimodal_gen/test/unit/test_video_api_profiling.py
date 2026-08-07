import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
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


def test_video_background_job_uses_indefinite_scheduler_timeout():
    captured = {}

    async def fake_process_generation_batch(_client, _batch, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing scheduler arguments")

    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(cleanup_video_request=lambda _batch: None)
    )
    update_fields = AsyncMock()

    with (
        patch.object(
            video_api,
            "process_generation_batch",
            new=fake_process_generation_batch,
        ),
        patch.object(video_api.VIDEO_STORE, "update_fields", new=update_fields),
        patch.object(video_api.logger, "exception"),
    ):
        asyncio.run(video_api._dispatch_job_async("job-id", batch))

    assert captured["scheduler_timeout_ms"] == -1
    update_fields.assert_awaited_once()
