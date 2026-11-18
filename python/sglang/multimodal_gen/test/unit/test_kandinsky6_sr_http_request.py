# SPDX-License-Identifier: Apache-2.0
"""Prompt admission and SR fields through the real JSON and multipart endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    resolve_sampling_params_cls,
)


@pytest.mark.parametrize("multipart", [False, True], ids=["json", "multipart"])
@pytest.mark.parametrize("prompt_required", [False, True], ids=["sr", "generic"])
def test_prompt_admission_and_sr_fields(tmp_path, multipart, prompt_required):
    args = SimpleNamespace(
        pipeline_config=Kandinsky6SRPipelineConfig(),
        pipeline_class_name="Kandinsky6SRPipeline",
        backend="auto",
        model_id=None,
        model_path="kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
        served_model_name="kandinsky6-sr-test",
        input_save_path=str(tmp_path / "uploads"),
        output_path=str(tmp_path / "outputs"),
        attention_backend_config={},
        enable_trace=False,
        comfyui_mode=False,
        num_gpus=1,
    )
    assert resolve_sampling_params_cls(args) is Kandinsky6SRSamplingParams
    params_cls = SamplingParams if prompt_required else Kandinsky6SRSamplingParams
    fields = dict(
        sr_resolution_scale=4,
        sr_tiles_batch_size=3,
        sr_tile_min_overlap=0.3,
        sr_target_resolution="hd",
        sr_target_resize_mode="fit",
    )
    assert Kandinsky6SRSamplingParams.video_request_extra_fields() == fields.keys()
    assert SamplingParams.video_request_extra_fields() == frozenset()
    payload = dict(video_path="unused.mp4", **fields)
    request = (
        {"files": {key: (None, str(value)) for key, value in payload.items()}}
        if multipart
        else {"json": payload}
    )
    app = FastAPI()
    app.include_router(video_api.router)
    dispatch = AsyncMock()
    with (
        patch(
            "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
            args,
        ),
        patch.object(video_api, "resolve_sampling_params_cls", return_value=params_cls),
        patch.object(video_api, "_dispatch_job_async", dispatch),
        TestClient(app) as client,
    ):
        response = client.post("/v1/videos", **request)
    assert response.status_code == (400 if prompt_required else 200), response.text
    if prompt_required:
        dispatch.assert_not_called()
    else:
        assert response.json()["status"] == "queued"
        dispatch.assert_awaited_once()
        _, batch = dispatch.call_args.args
        assert isinstance(batch.sampling_params, Kandinsky6SRSamplingParams)
        assert batch.prompt == "" and batch.video_path == "unused.mp4"
        assert {key: vars(batch.sampling_params)[key] for key in fields} == fields


@pytest.mark.parametrize("params_cls", [Kandinsky6SRSamplingParams, SamplingParams])
def test_multipart_allowlist_drops_undeclared_fields(params_cls):
    extras = video_api._multipart_video_extras(
        {"sr_resolution_scale": "4", "sr_tiles_batch_size": "3", "undeclared": "1"},
        extra_body=None,
        extra_params=None,
        sampling_params_cls=params_cls,
    )
    assert extras == (
        {"sr_resolution_scale": 4, "sr_tiles_batch_size": 3}
        if params_cls is Kandinsky6SRSamplingParams
        else {}
    )
