# SPDX-License-Identifier: Apache-2.0
"""Capability selection and real HTTP parsing, with model execution stubbed."""

import argparse
import io
import pickle
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.configs.task_type import ModelTaskType as Task
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, utils, video_api
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


@dataclass
class MultiConfig(PipelineConfig):
    task_type: Task = Task.T2V
    supported_task_types: ClassVar[tuple[Task, ...]] = (
        Task.T2I,
        Task.I2I,
        Task.T2V,
        Task.I2V,
        Task.V2V,
        Task.F2V,
    )


def make_args(config, output=None):
    return SimpleNamespace(
        pipeline_config=config,
        model_path="test-model",
        backend="sglang",
        model_id=None,
        pipeline_class_name=None,
        output_path=output,
        input_save_path=None,
        served_model_name="test-model",
        num_gpus=1,
        comfyui_mode=False,
        attention_backend_config={},
        enable_trace=False,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        enable_cfg_parallel=False,
        warmup_steps=1,
    )


def test_legacy_enums_and_capabilities():
    from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        DataType as OldDataType,
    )

    assert ModelTaskType is Task
    assert OldDataType is DataType
    for task in Task:
        config = PipelineConfig(task_type=task)
        assert config.get_supported_task_types() == (task,)
        assert config.resolve_task_type() == task
        assert config.resolve_task_type(task.name.lower()) == task


@pytest.mark.parametrize("declared", [(), (Task.T2I,), (Task.T2V, Task.T2V)])
def test_invalid_capability_declarations(declared):
    class Invalid(MultiConfig):
        supported_task_types = declared

    with pytest.raises(ValueError, match="supported_task_types"):
        Invalid().get_supported_task_types()


def test_selection_and_ambiguity():
    config = MultiConfig()
    assert config.resolve_task_type() == Task.T2V
    assert config.resolve_task_type(data_type=DataType.IMAGE) == Task.T2I
    assert (
        config.resolve_task_type(data_type=DataType.IMAGE, has_image=True) == Task.I2I
    )
    assert config.resolve_task_type(has_image=True) == Task.I2V
    with pytest.raises(ValueError, match="Specify task_type"):
        config.resolve_task_type(has_video=True)
    assert config.resolve_task_type("f2v", has_video=True) == Task.F2V
    assert config.task_type == Task.T2V
    with pytest.raises(ValueError, match="endpoint produces IMAGE"):
        config.resolve_task_type("t2v", data_type=DataType.IMAGE)
    with pytest.raises(ValueError, match="Unsupported task_type"):
        config.resolve_task_type("i2m")
    with pytest.raises(ValueError, match="Unknown task_type"):
        config.resolve_task_type("imaginary")


@pytest.mark.parametrize(
    "task,image,video,output",
    [
        ("t2i", None, None, DataType.IMAGE),
        ("i2i", "reference.png", None, DataType.IMAGE),
        ("t2v", None, None, DataType.VIDEO),
        ("i2v", "reference.png", None, DataType.VIDEO),
        ("v2v", None, "reference.mp4", DataType.VIDEO),
        ("f2v", None, "reference.mp4", DataType.VIDEO),
    ],
)
def test_preparation_copy_transport_and_extension(task, image, video, output, tmp_path):
    config = MultiConfig()
    sp = SamplingParams(
        prompt="test",
        task_type=task,
        image_path=image,
        video_path=video,
        num_frames=25,
        output_file_name="result",
        adjust_frames=False,
    )
    sp._adjust(make_args(config, str(tmp_path)))
    sp._validate_with_pipeline_config(config)
    sp = pickle.loads(pickle.dumps(replace(sp, prompt="expanded prompt")))
    assert sp.task_type == Task.parse(task)
    assert sp.data_type == output
    assert sp.output_file_path().endswith(output.get_default_extension())
    assert sp.num_frames == (1 if output == DataType.IMAGE else 25)
    assert config.task_type == Task.T2V


@pytest.mark.parametrize(
    "task,image,video",
    [
        ("i2i", None, None),
        ("i2v", None, None),
        ("v2v", None, None),
        ("f2v", None, None),
        ("t2i", "a.png", None),
        ("t2v", None, "a.mp4"),
        ("v2v", "a.png", "a.mp4"),
    ],
)
def test_task_input_contracts(task, image, video):
    sp = SamplingParams(task_type=task, image_path=image, video_path=video)
    with pytest.raises(ValueError):
        sp._validate_with_pipeline_config(MultiConfig())


def test_cli_accepts_standard_task_type():
    parser = argparse.ArgumentParser()
    SamplingParams.add_cli_args(parser)
    assert parser.parse_args(["--task-type", "i2v"]).task_type == "i2v"


def test_batch_signature_separates_tasks():
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = make_args(MultiConfig())
    a = Req(sampling_params=SamplingParams(task_type=Task.V2V, video_path="same.mp4"))
    b = Req(sampling_params=SamplingParams(task_type=Task.F2V, video_path="same.mp4"))
    assert scheduler._build_dynamic_batch_signature(
        a
    ) != scheduler._build_dynamic_batch_signature(b)


def test_multi_task_warmup_uses_default_without_optional_image(monkeypatch):
    from sglang.multimodal_gen.runtime import warmup_request_builder as warmup

    monkeypatch.setattr(
        warmup, "get_model_sampling_defaults", lambda _: SamplingParams()
    )
    reqs = warmup.build_warmup_reqs(
        make_args(MultiConfig()),
        warmup_resolutions=["512x512"],
        server_based_warmup=True,
    )
    assert len(reqs) == 1
    assert reqs[0].task_type == Task.T2V
    assert reqs[0].data_type == DataType.VIDEO
    assert reqs[0].image_path is None


@pytest.fixture(params=[(MultiConfig, SamplingParams)])
def http_client(monkeypatch, tmp_path, request):
    config_cls, sampling_cls = request.param
    config = config_cls()
    args = make_args(config, str(tmp_path))
    for module in (image_api, video_api, utils):
        monkeypatch.setattr(module, "get_global_server_args", lambda: args)
    for module in (image_api, video_api):
        monkeypatch.setattr(
            module, "resolve_sampling_params_cls", lambda _: sampling_cls
        )
    monkeypatch.setattr(
        SamplingParams, "from_pretrained", classmethod(lambda cls, *a, **kw: cls())
    )
    admitted = []

    async def generate(client, req, **kwargs):
        admitted.append(req)
        path = req.output_file_path()
        Image.new("RGB", (64, 64)).save(path)
        return [path], OutputBatch()

    async def upload(paths):
        return [None] * len(paths)

    async def dispatch(job_id, req, **kwargs):
        admitted.append(req)

    # Run queued dispatch immediately so the tests can inspect admitted requests.
    def start(job_id, coroutine):
        import asyncio

        return asyncio.create_task(coroutine)

    monkeypatch.setattr(image_api, "process_generation_batch", generate)
    monkeypatch.setattr(image_api, "_upload_and_cleanup_images", upload)
    monkeypatch.setattr(video_api, "_dispatch_job_async", dispatch)
    monkeypatch.setattr(video_api, "_start_video_job", start)
    app = FastAPI()
    app.include_router(image_api.router)
    app.include_router(video_api.router)
    with TestClient(app) as client:
        yield client, admitted, config


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "http_client", [(Cosmos3Config, Cosmos3SamplingParams)], indirect=True
)
@pytest.mark.parametrize("task", [None, "t2i"])
def test_cosmos3_image_endpoint_selects_image_task(http_client, task):
    client, admitted, config = http_client
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "text", "size": "64x64", "task_type": task},
    )
    assert response.status_code == 200, response.text
    assert admitted[-1].task_type == Task.T2I
    assert admitted[-1].data_type == DataType.IMAGE
    assert admitted[-1].num_frames == 1
    assert config.task_type == Task.TI2V


@pytest.mark.parametrize(
    "http_client", [(Cosmos3Config, Cosmos3SamplingParams)], indirect=True
)
@pytest.mark.parametrize(
    "conditioning,task",
    [
        ({}, Task.TI2V),
        ({"image_path": "/tmp/image.png"}, Task.TI2V),
        ({"video_path": "/tmp/clip.mp4"}, Task.V2V),
    ],
)
def test_cosmos3_video_endpoint_preserves_conditioning(http_client, conditioning, task):
    client, admitted, config = http_client
    response = client.post(
        "/v1/videos", json={"prompt": "video", "size": "64x64", **conditioning}
    )
    assert response.status_code == 200, response.text
    assert admitted[-1].task_type == task
    assert admitted[-1].data_type == DataType.VIDEO
    assert config.task_type == Task.TI2V


@pytest.mark.parametrize(
    "frames,conditioning,task",
    [
        (1, {}, Task.T2I),
        (81, {}, Task.TI2V),
        (81, {"image_path": "/tmp/image.png"}, Task.TI2V),
        (81, {"video_path": "/tmp/clip.mp4"}, Task.V2V),
    ],
)
def test_cosmos3_sampling_resolves_existing_modes(frames, conditioning, task, tmp_path):
    config = Cosmos3Config()
    sp = Cosmos3SamplingParams(prompt="test", num_frames=frames, **conditioning)
    sp._adjust(make_args(config, str(tmp_path)))
    sp._validate_with_pipeline_config(config)
    assert sp.task_type == task
    assert sp.data_type == task.data_type()
    assert sp.output_file_path().endswith(sp.data_type.get_default_extension())
    assert config.task_type == Task.TI2V


def test_image_json_and_multipart_choose_tasks_on_video_default(http_client):
    client, admitted, config = http_client
    response = client.post(
        "/v1/images/generations",
        json={
            "prompt": "text",
            "response_format": "b64_json",
            "size": "64x64",
            "task_type": "t2i",
        },
    )
    assert response.status_code == 200, response.text
    assert admitted[-1].task_type == Task.T2I
    for mode in ({}, {"task_type": "i2i"}):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "edit",
                "size": "64x64",
                "response_format": "b64_json",
                **mode,
            },
            files={"image": ("image.png", png_bytes(), "image/png")},
        )
        assert response.status_code == 200, response.text
        assert admitted[-1].task_type == Task.I2I
    assert config.task_type == Task.T2V


@pytest.mark.parametrize("multipart", [False, True])
def test_video_transports_keep_selected_task(http_client, multipart):
    client, admitted, config = http_client
    payload = {
        "prompt": "video",
        "task_type": "f2v",
        "size": "64x64",
        "video_path": "/tmp/clip.mp4",
    }
    if multipart:
        response = client.post(
            "/v1/videos", files={k: (None, v) for k, v in payload.items()}
        )
    else:
        response = client.post("/v1/videos", json=payload)
    assert response.status_code == 200, response.text
    assert admitted[-1].task_type == Task.F2V
    assert admitted[-1].data_type == DataType.VIDEO
    assert config.task_type == Task.T2V


def test_http_rejects_invalid_tasks_before_model_execution(http_client):
    client, admitted, _ = http_client
    for endpoint, payload in [
        ("images/generations", {"task_type": "t2v"}),
        ("images/generations", {"task_type": "i2i"}),
        ("videos", {"task_type": "t2i"}),
        ("videos", {"task_type": "v2v"}),
        ("videos", {"task_type": "t2v", "video_path": "/tmp/clip.mp4"}),
        ("videos", {"task_type": "unknown"}),
    ]:
        response = client.post("/v1/" + endpoint, json={"prompt": "invalid", **payload})
        assert response.status_code == 400, response.text
    assert admitted == []


@pytest.mark.parametrize("multipart", [False, True])
def test_unconditioned_request_on_image_conditioned_default(http_client, multipart):
    client, admitted, config = http_client
    config.task_type = Task.I2V
    payload = {"prompt": "video", "task_type": "t2v", "size": "64x64"}
    if multipart:
        response = client.post(
            "/v1/videos", files={k: (None, v) for k, v in payload.items()}
        )
    else:
        response = client.post("/v1/videos", json=payload)
    assert response.status_code == 200, response.text
    assert admitted[-1].task_type == Task.T2V
    assert config.task_type == Task.I2V


def test_capability_discovery(monkeypatch):
    from sglang.multimodal_gen import registry
    from sglang.multimodal_gen.runtime.entrypoints import http_server
    from sglang.multimodal_gen.runtime.entrypoints.openai import common_api

    monkeypatch.setattr(registry, "get_model_info", lambda *a, **kw: None)
    monkeypatch.setattr(common_api, "get_model_info", lambda *a, **kw: None)
    args = make_args(MultiConfig())
    expected = [task.name for task in MultiConfig.supported_task_types]
    card = common_api._build_model_card(args, "test-model")
    assert card.task_type == "T2V"
    assert card.supported_task_types == expected
    app = FastAPI()
    app.state.server_args = args
    app.include_router(http_server.health_router)
    with TestClient(app) as client:
        for endpoint in ("/models", "/model_info"):
            response = client.get(endpoint)
            assert response.status_code == 200, response.text
            assert response.json()["supported_task_types"] == expected
        info = client.get("/model_info").json()
        assert info["task_type"] == "T2V"
        assert info["output_types"] == ["IMAGE", "VIDEO"]
        assert info["has_image_understanding"] is True
