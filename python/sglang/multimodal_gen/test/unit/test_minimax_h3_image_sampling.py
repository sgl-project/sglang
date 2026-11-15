# SPDX-License-Identifier: Apache-2.0
"""Image sampling params via the real factory and HTTP router; registry is stubbed."""

import io
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sample.minimax_h3_image import (
    MiniMaxH3ImageSamplingParams,
)
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as api_utils
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY as PLAN,
)


class ApiTestPipelineConfig(PipelineConfig):
    supported_task_types = (ModelTaskType.T2I, ModelTaskType.I2I)
    # Off by default here; tests enable it per case.
    min_generation_area = None

    def release_pdd_steps(self):
        # A 6-step PDD release.
        return 6


@pytest.mark.parametrize("images", [None, ["one.png", "two.png"]])
@pytest.mark.parametrize("is_warmup", [True, False])
def test_synthetic_warmup_builds_a_native_image_request(api_server, images, is_warmup):
    params = MiniMaxH3ImageSamplingParams(
        prompt="warmup",
        width=64,
        height=32,
        image_path=images,
        num_inference_steps=6,
        task_type=ModelTaskType.T2I,
    )
    req = Req(sampling_params=params)
    req.set_as_warmup(2)
    req.is_warmup = is_warmup
    if not is_warmup:
        req.extra["server_internal_prewarm"] = True
    params.prepare_synthetic_warmup_request_for_queue(req, api_server)
    assert req.num_inference_steps == (2 if is_warmup else 6)
    assert req.is_warmup == is_warmup and not req.save_output
    plan = req.extra[PLAN]
    assert plan.task == ("edit" if images else "t2i")
    assert req.task_type == (ModelTaskType.I2I if images else ModelTaskType.T2I)
    assert (plan.shape["width"], plan.shape["height"]) == (64, 32)


@pytest.mark.parametrize("quality", ["exact", "lossless"])
def test_reference_and_lossless_quality_are_accepted(quality):
    assert MiniMaxH3ImageSamplingParams(quality=quality).quality == quality


@pytest.mark.parametrize(
    "overrides",
    [
        {"quality": "high"},
        {"enable_teacache": True},
        {"enable_cache_dit": True},
        {"enable_spectrum": True},
        {"skip_softmax_params": {"threshold_scale_factor": 0.1}},
    ],
)
def test_unsupported_approximation_is_rejected(overrides):
    with pytest.raises(ValueError, match="MiniMax H3 image"):
        MiniMaxH3ImageSamplingParams(**overrides)


@pytest.fixture
def api_server(monkeypatch, tmp_path):
    import sglang.multimodal_gen.registry as registry

    server = SimpleNamespace(
        model_path="/not-a-real-model",
        pipeline_class_name="ApiTestOnly",
        pipeline_config=ApiTestPipelineConfig(task_type=ModelTaskType.T2I),
        backend="sglang",
        model_id=None,
        output_path=str(tmp_path),
        input_save_path=str(tmp_path),
        comfyui_mode=False,
        attention_backend_config=None,
        enable_trace=False,
        batching_max_size=1,
    )
    # Mock only class selection; exercise the real factory, merge and hooks.
    monkeypatch.setattr(
        registry,
        "get_pipeline_config_classes",
        lambda name: (ApiTestPipelineConfig, MiniMaxH3ImageSamplingParams),
    )
    monkeypatch.setattr(api_utils, "get_global_server_args", lambda: server)
    return server


def params(**kwargs):
    return MiniMaxH3ImageSamplingParams(
        **{"prompt": "  raw\n<text>  ", "width": 1024, "height": 768, **kwargs}
    )


@pytest.mark.parametrize("edit", [False, True])
def test_real_factory_merge_and_prepare_hooks(api_server, edit):
    sampling = api_utils.build_sampling_params(
        "req-test",
        prompt="  raw prompt  ",
        request_data_type=DataType.IMAGE,
        size="1056x800",
        seed=17,
        generator_device="cuda",
        image_path=["b.png", "a.png"] if edit else None,
    )
    req = prepare_request(api_server, sampling)
    assert isinstance(sampling, MiniMaxH3ImageSamplingParams)
    assert req.width == 1056 and req.height == 800
    assert req.prompt == "  raw prompt  "
    # The release's head count is the default step count.
    assert req.num_inference_steps == 6
    # One positive branch: no negative prompt, no guidance.
    assert req.negative_prompt is None
    assert (req.guidance_scale, req.true_cfg_scale) == (1.0, None)
    assert not req.do_classifier_free_guidance
    assert req.output_file_name.endswith(".png")
    assert req.extra[PLAN].seed == 17
    assert req.extra[PLAN].task == ("edit" if edit else "t2i")


@pytest.mark.parametrize(
    "override,steps",
    [
        # The constructor keeps a user count; the server checks it against the
        # release (and fills an omitted one) in _adjust.
        ({"num_inference_steps": 6}, 6),
        ({"num_inference_steps": 13, "guidance_scale": 1}, 13),
        # A scale of 1 is no guidance, so it is accepted.
        ({"true_cfg_scale": 1.0}, None),
    ],
)
def test_request_steps(override, steps):
    sampling = params(**override)
    assert sampling.num_inference_steps == steps
    assert sampling.guidance_scale == 1.0


@pytest.mark.parametrize(
    "override,match",
    [
        ({"num_inference_steps": 0}, "num_inference_steps >= 1"),
        ({"num_inference_steps": 2.0}, "num_inference_steps >= 1"),
        ({"negative_prompt": ""}, "negative_prompt is not supported"),
        ({"guidance_scale": 3.0}, "guidance_scale must be 1"),
        ({"guidance_scale": True}, "guidance_scale must be 1"),
        ({"true_cfg_scale": float("nan")}, "true_cfg_scale must be 1"),
        ({"num_frames": 2}, "one image"),
        ({"num_outputs_per_prompt": 2}, "one image"),
        ({"min_generation_area": 512}, "min_generation_area"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_image_requests_fail_at_constructor(override, match):
    with pytest.raises(ValueError, match=match):
        params(**override)


@pytest.fixture
def image_router(api_server, monkeypatch):
    admitted = []

    async def generate(client, req, **kwargs):
        admitted.append(req)
        path = req.output_file_path()
        Image.new("RGB", (16, 16)).save(path)
        with open(path, "rb") as f:
            return [path], OutputBatch(output_image_bytes=[f.read()])

    async def upload(paths):
        return [None] * len(paths)

    monkeypatch.setattr(image_api, "get_global_server_args", lambda: api_server)
    monkeypatch.setattr(image_api, "process_generation_batch", generate)
    monkeypatch.setattr(image_api.cloud_storage, "upload_all_and_cleanup", upload)
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app) as client:
        yield client, admitted


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_http_generations_and_edits_carry_steps(image_router):
    client, admitted = image_router
    fields = {"prompt": "p", "size": "64x32", "num_inference_steps": 6}
    response = client.post(
        "/v1/images/generations", json={**fields, "guidance_scale": 1}
    )
    assert response.status_code == 200, response.text
    response = client.post(
        "/v1/images/edits",
        data={**fields, "true_cfg_scale": "1"},
        files=[("image[]", ("ref.png", _png(), "image/png"))],
    )
    assert response.status_code == 200, response.text
    t2i, edit = admitted
    assert (t2i.extra[PLAN].task, edit.extra[PLAN].task) == ("t2i", "edit")
    assert t2i.num_inference_steps == edit.num_inference_steps == 6
    assert t2i.guidance_scale == edit.guidance_scale == 1.0


@pytest.mark.parametrize(
    "fields",
    [
        {"num_inference_steps": 0},
        # The 6-step release fixes the count.
        {"num_inference_steps": 9},
        {"prompt": " \n "},
        {"guidance_scale": 1.5},
        {"true_cfg_scale": 3.0},
        {"negative_prompt": "blurry"},
    ],
)
def test_http_rejects_invalid_steps_and_cfg_fields(image_router, fields):
    client, admitted = image_router
    response = client.post(
        "/v1/images/generations", json={"prompt": "p", "size": "64x32", **fields}
    )
    assert response.status_code == 400, response.text
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "p", "size": "64x32", **fields},
        files=[("image[]", ("ref.png", _png(), "image/png"))],
    )
    assert response.status_code == 400, response.text
    assert not admitted


def test_http_rejects_an_empty_prompt(image_router):
    client, admitted = image_router
    response = client.post(
        "/v1/images/generations", json={"prompt": "", "size": "64x32"}
    )
    assert response.status_code == 400, response.text
    assert "prompt must be a non-empty string" in response.text
    # The form parser already treats an empty edits prompt as missing.
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "", "size": "64x32"},
        files=[("image[]", ("ref.png", _png(), "image/png"))],
    )
    assert response.status_code == 422, response.text
    assert not admitted


def _post_both(client, fields, json_extra=None):
    """POST the same request to generations (JSON) and edits (multipart)."""
    t2i = client.post(
        "/v1/images/generations", json={"prompt": "p", **fields, **(json_extra or {})}
    )
    edit = client.post(
        "/v1/images/edits",
        data={"prompt": "p", **{k: str(v) for k, v in fields.items()}},
        files=[("image[]", ("ref.png", _png(), "image/png"))],
    )
    return t2i, edit


M = 2048 * 2048


def test_http_server_default_area_generates_on_the_canvas(image_router, api_server):
    client, admitted = image_router
    api_server.pipeline_config.min_generation_area = M
    for response in _post_both(client, {"size": "1000x1500"}):
        assert response.status_code == 200, response.text
    for req in admitted:
        shape = req.extra[PLAN].shape
        # The request keeps the delivered size; the plan carries the canvas.
        assert (req.width, req.height) == (1000, 1500)
        assert (shape["width"], shape["height"]) == (1664, 2496)
        assert (shape["output_width"], shape["output_height"]) == (1000, 1500)


def test_http_request_area_overrides_the_server_default(image_router, api_server):
    client, admitted = image_router
    api_server.pipeline_config.min_generation_area = M
    t2i, edit = _post_both(client, {"size": "1024x768", "min_generation_area": 1024})
    assert t2i.status_code == edit.status_code == 200, (t2i.text, edit.text)
    # extra_body is the OpenAI SDK's container for the same field.
    response = client.post(
        "/v1/images/generations",
        json={
            "prompt": "p",
            "size": "1024x768",
            "extra_body": {"min_generation_area": 4096 * 4096},
        },
    )
    assert response.status_code == 200, response.text
    shapes = [req.extra[PLAN].shape for req in admitted]
    assert [(s["width"], s["height"]) for s in shapes] == [
        (1024, 768),
        (1024, 768),
        (4736, 3552),
    ]
    assert shapes[0]["output_width"] is None


@pytest.mark.parametrize(
    "server_area,fields",
    [
        (M, {"size": "1024x768", "min_generation_area": 1023}),
        (M, {"size": "1024x768", "min_generation_area": "big"}),
        # Disabled: the field is rejected and sizes must stay 32px-aligned.
        (None, {"size": "1024x768", "min_generation_area": M}),
        (None, {"size": "1000x768"}),
    ],
)
def test_http_rejects_invalid_min_generation_area(
    image_router, api_server, server_area, fields
):
    client, admitted = image_router
    api_server.pipeline_config.min_generation_area = server_area
    for response in _post_both(client, fields):
        assert response.status_code == 400, response.text
    assert not admitted
