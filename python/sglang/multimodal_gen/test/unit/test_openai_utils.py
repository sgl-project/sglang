# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import torch
from fastapi import FastAPI
from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import forward_to_scheduler
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    build_sampling_params,
    process_generation_batch,
    request_field_value,
    request_model_kwargs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    MemorySnapshot,
    RequestMetrics,
)
from sglang.srt.utils.json_response import orjson_response


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"output": [torch.zeros(3, 8, 8)]},
        {"output_file_paths": ["partial.png"], "output_image_bytes": [b"partial"]},
        {"raw_frame_batches": [b"partial"]},
    ],
    ids=["empty", "tensor", "encoded", "raw-frames"],
)
def test_generation_rejects_errors_before_publishing_partial_output(
    monkeypatch, tmp_path, payload
):
    monkeypatch.setattr(
        openai_utils,
        "get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )
    batch = Req(
        sampling_params=SamplingParams(
            prompt="test", output_path=str(tmp_path), output_file_name="retry.png"
        )
    )
    recovered = OutputBatch(output_file_paths=["recovered.png"])
    client = SimpleNamespace(
        forward=AsyncMock(
            side_effect=[OutputBatch(error="output save failed", **payload), recovered]
        )
    )

    async def run():
        with pytest.raises(RuntimeError, match="output save failed"):
            await process_generation_batch(client, batch)
        assert list(tmp_path.iterdir()) == []
        paths, result = await process_generation_batch(client, batch)
        assert paths == ["recovered.png"] and result is recovered

    asyncio.run(run())


@pytest.mark.parametrize("error", [None, "output save failed"])
@pytest.mark.parametrize("file_state", ["valid", "missing", "empty"])
def test_vertex_response_does_not_publish_failed_output(
    monkeypatch, tmp_path, error, file_state
):
    path = tmp_path / "out.mp4"
    if file_state != "missing":
        path.write_bytes(b"encoded video" if file_state == "valid" else b"")
    metrics = RequestMetrics("test")
    metrics.denoising_stages.add("denoising")
    metrics.record_step(0.1)
    metrics.record_memory_snapshot("peak", MemorySnapshot(1, 2, 3, 4))
    monkeypatch.setattr(
        async_scheduler_client,
        "forward",
        AsyncMock(
            return_value=OutputBatch(
                output_file_paths=[str(path)],
                error=error,
                metrics=metrics,
                metrics_list=[metrics, None],
            )
        ),
    )
    sampling = SamplingParams(prompt="test")
    response = asyncio.run(
        forward_to_scheduler(Req(sampling_params=sampling), sampling)
    )
    response = json.loads(orjson_response(response).body)
    if error:
        assert response == {"error": error}
    elif file_state != "valid":
        assert set(response) == {"error"}
        assert str(path) in response["error"]
    else:
        assert base64.b64decode(response["output"]) == path.read_bytes()
        assert response["metrics"] == metrics.to_dict()
        assert response["metrics_list"] == [metrics.to_dict(), None]


def test_vertex_slow_output_read_does_not_block_liveness(monkeypatch, tmp_path):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"encoded video")
    started, release = threading.Event(), threading.Event()

    def slow_open(*args, **kwargs):
        started.set()
        assert release.wait(3), "HTTP loop could not release the output reader"
        return open(*args, **kwargs)

    monkeypatch.setattr(http_server, "open", slow_open, raising=False)
    monkeypatch.setattr(
        async_scheduler_client,
        "forward",
        AsyncMock(return_value=OutputBatch(output_file_paths=[str(path)])),
    )
    app = FastAPI()
    app.include_router(http_server.health_router)

    async def run():
        sampling = SamplingParams(prompt="test")
        task = asyncio.create_task(
            forward_to_scheduler(Req(sampling_params=sampling), sampling)
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert not task.done(), "output read blocked the HTTP event loop"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                assert (await client.get("/liveness")).status_code == 200
        finally:
            release.set()
            response = await task
        assert base64.b64decode(response["output"]) == path.read_bytes()

    asyncio.run(run())


@pytest.mark.parametrize(
    "invalid", [{}, {"prompt": 123}], ids=["missing-prompt", "numeric-prompt"]
)
def test_vertex_validates_batch_before_dispatch(monkeypatch, tmp_path, invalid):
    model_path = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    model_path.mkdir()
    (model_path / "model_index.json").write_text('{"_class_name": "WanPipeline"}')
    server_args = SimpleNamespace(
        model_path=str(model_path),
        pipeline_class_name="WanPipeline",
        pipeline_config=WanT2V480PConfig(),
        output_path=str(tmp_path),
        num_gpus=1,
        comfyui_mode=False,
        attention_backend_config={},
        enable_trace=False,
        backend="auto",
        model_id=None,
    )
    for module in (http_server, openai_utils):
        monkeypatch.setattr(module, "get_global_server_args", lambda: server_args)
    path = tmp_path / "out.mp4"
    path.write_bytes(b"encoded video")
    forward = AsyncMock(return_value=OutputBatch(output_file_paths=[str(path)]))
    monkeypatch.setattr(async_scheduler_client, "forward", forward)
    app = FastAPI()
    app.include_router(http_server.vertex_router)

    async def run():
        valid = {"prompt": "test"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            for instances in ([invalid], [valid, invalid], [invalid, valid]):
                response = await client.post(
                    http_server.VERTEX_ROUTE, json={"instances": instances}
                )
                assert response.status_code == 400, response.text
                assert "prompt" in response.json()["detail"]
                forward.assert_not_awaited()
            response = await client.post(
                http_server.VERTEX_ROUTE, json={"instances": [valid]}
            )
            assert response.status_code == 200, response.text
            assert (
                base64.b64decode(response.json()["predictions"][0]["output"])
                == path.read_bytes()
            )
            forward.assert_awaited_once()

    asyncio.run(run())


def test_request_fields_preserve_falsy_values_and_extra_precedence():
    for cls in (ImageGenerationsRequest, VideoGenerationsRequest):
        for container in ("extra_body", "extra_json", "extra_args", "extra_params"):
            for direct in (0, None):
                request = cls(
                    prompt="test",
                    seed=direct,
                    enable_cache_dit=False,
                    **{container: json.dumps({"seed": 42, "enable_cache_dit": True})},
                )
                before = request.model_dump()
                assert request_field_value(request, "seed") == (
                    42 if direct is None else 0
                )
                assert request_field_value(request, "enable_cache_dit") is False
                assert request.model_dump() == before


def test_video_model_kwargs_exclude_transport_only_aliases():
    request = VideoGenerationsRequest(
        prompt="test", extra_body={"task": "t2va", "audio_guidance_scale": 2.0}
    )
    assert request_model_kwargs(request, MiniMaxH3SamplingParams, "video") == {
        "task": "t2va"
    }
    assert request.model_extra["extra_body"]["audio_guidance_scale"] == 2.0
    assert request_model_kwargs(request, MiniMaxH3SamplingParams, "image") == {}


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_build_sampling_params_resolves_size_and_explicit_dimensions(monkeypatch):
    server_args = SimpleNamespace(model_path="zai-org/GLM-Image")
    monkeypatch.setattr(openai_utils, "get_global_server_args", lambda: server_args)
    captured = {}

    def fake_from_user_sampling_params_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        SamplingParams,
        "from_user_sampling_params_args",
        fake_from_user_sampling_params_args,
    )

    cases = [
        (
            {"size": "500x500", "width": None, "height": None},
            (500, 500),
        ),
        (
            {"size": "1024x1024", "width": None, "height": 600},
            (1024, 600),
        ),
        (
            {"size": "500x500", "width": None, "height": 600},
            (500, 600),
        ),
        (
            {"size": "500x500", "width": 600, "height": None},
            (600, 500),
        ),
        (
            {"size": "500x500", "width": 600, "height": 700},
            (600, 700),
        ),
    ]

    for request_fields, expected in cases:
        captured.clear()

        build_sampling_params("request-id", **request_fields)

        assert (captured["width"], captured["height"]) == expected
