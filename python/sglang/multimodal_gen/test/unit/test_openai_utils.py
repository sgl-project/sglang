# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
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
def test_vertex_response_does_not_publish_failed_output(monkeypatch, tmp_path, error):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"encoded video")
    monkeypatch.setattr(
        async_scheduler_client,
        "forward",
        AsyncMock(return_value=OutputBatch(output_file_paths=[str(path)], error=error)),
    )
    sampling = SamplingParams(prompt="test")
    response = asyncio.run(
        forward_to_scheduler(Req(sampling_params=sampling), sampling)
    )
    if error:
        assert response == {"error": error}
    else:
        assert base64.b64decode(response["output"]) == path.read_bytes()


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
