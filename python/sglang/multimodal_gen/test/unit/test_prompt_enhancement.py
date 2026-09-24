import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams
from sglang.multimodal_gen.configs.sample.zimage import ZImageTurboSamplingParams
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
    PromptEnhancer,
    PromptEnhancerConfig,
    maybe_enhance_prompt,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module


def completion(text="Rewritten prompt", finish_reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}


@pytest.mark.parametrize(
    "override",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"model": ""},
        {"base_url": "file:///tmp/server"},
        {"base_url": "https://user:secret@example.org/v1"},
        {"base_url": "https://example.org/v1?key=secret"},
        {"generation_kwargs": {"stream": True}},
        {"generation_kwargs": {"messages": []}},
        {"generation_kwargs": {"n": 2}},
        {"typo": True},
    ],
)
def test_config_rejects_invalid_or_conflicting_fields(override):
    with pytest.raises(ValidationError):
        PromptEnhancerConfig.model_validate(
            {"base_url": "http://localhost:30001/v1", "model": "enhancer", **override}
        )


def test_missing_auth_env_fails_at_startup(monkeypatch):
    monkeypatch.delenv("TEST_ENHANCER_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_ENHANCER_KEY is unset"):
        PromptEnhancer(
            PromptEnhancerConfig(
                base_url="http://localhost:30001/v1",
                model="enhancer",
                api_key_env="TEST_ENHANCER_KEY",
            )
        )


def test_disabled_path_does_not_need_a_client():
    assert (
        asyncio.run(maybe_enhance_prompt(None, "original", enabled=False, task="image"))
        == "original"
    )


@pytest.mark.parametrize(
    "upstream, status",
    [
        (completion(" "), 502),
        (completion(None), 502),
        (completion("partial", "length"), 502),
        (completion("refusal", "content_filter"), 502),
        ({"choices": []}, 502),
        ({"error": "sensitive upstream detail"}, 502),
        (httpx.Response(401, text="secret"), 502),
        (httpx.ReadTimeout("secret"), 504),
        (httpx.ConnectError("secret"), 502),
    ],
)
def test_upstream_failures_are_not_silent(upstream, status):
    async def run():
        enhancer = PromptEnhancer(
            PromptEnhancerConfig(base_url="http://localhost/v1", model="enhancer")
        )

        def respond(request):
            if isinstance(upstream, Exception):
                raise upstream
            if isinstance(upstream, httpx.Response):
                return upstream
            return httpx.Response(200, json=upstream)

        await enhancer.client.aclose()
        enhancer.client = httpx.AsyncClient(
            base_url="http://localhost/v1/", transport=httpx.MockTransport(respond)
        )
        try:
            with pytest.raises(HTTPException) as error:
                await enhancer.enhance("original", task="image", image_paths=[])
            assert error.value.status_code == status
            assert "secret" not in error.value.detail
            assert "sensitive" not in error.value.detail
        finally:
            await enhancer.close()

    asyncio.run(run())


@pytest.fixture
def server(monkeypatch, tmp_path):
    calls = []
    batches = []
    rewritten = '{"high_level_description": "A red teapot"}'
    config_path = tmp_path / "enhancer.json"
    config_path.write_text(
        json.dumps(
            {
                "base_url": "http://enhancer.local/v1",
                "model": "test-vlm",
                "include_images": True,
                "api_key_env": "TEST_ENHANCER_KEY",
                "system_prompt": "Return the target model's JSON caption.",
                "generation_kwargs": {
                    "max_tokens": 128,
                    "response_format": {"type": "json_object"},
                },
            }
        )
    )
    monkeypatch.setenv("TEST_ENHANCER_KEY", "test-key")

    def respond(request):
        assert request.url == "http://enhancer.local/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion(rewritten))

    # keep the real HTTP client and lifespan; replace only the external transport
    original_init = httpx.AsyncClient.__init__

    def client_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_init)
    monkeypatch.setattr(ServerArgs, "__post_init__", lambda self: None)
    args = ServerArgs(
        model_path="test-model",
        pipeline_config=ZImagePipelineConfig(),
        num_gpus=1,
        prompt_enhancer_config=str(config_path),
        warmup_mode="disabled",
        output_path=str(tmp_path / "outputs"),
        input_save_path=str(tmp_path / "inputs"),
    )
    monkeypatch.setattr(server_args_module, "_global_server_args", args)

    def model_info(*args_, **kwargs):
        cls = (
            ZImageTurboSamplingParams
            if args.pipeline_config.task_type.is_image_gen()
            else WanT2V_1_3B_SamplingParams
        )
        return SimpleNamespace(sampling_param_cls=cls)

    monkeypatch.setattr(registry, "get_model_info", model_info)
    monkeypatch.setattr(async_scheduler_client, "initialize", lambda args: None)
    monkeypatch.setattr(async_scheduler_client, "close", lambda: None)
    monkeypatch.setattr(http_server, "_wait_until_http_live", AsyncMock())
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.scheduler_client.run_zeromq_broker", AsyncMock()
    )

    async def forward(requests):
        batches.extend(requests)
        batch = requests[0]
        paths = []
        for idx in range(batch.num_outputs_per_prompt):
            path = tmp_path / f"{batch.request_id}_{idx}.png"
            Image.new("RGB", (16, 16), "red").save(path)
            paths.append(str(path))
        return OutputBatch(output_file_paths=paths)

    monkeypatch.setattr(async_scheduler_client, "forward", forward)
    app = http_server.create_app(args)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            args=args,
            calls=calls,
            batches=batches,
            rewritten=rewritten,
        )
    assert app.state.prompt_enhancer.client.is_closed


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("endpoint", ["image", "edit", "video_json", "video_form"])
def test_http_routes_enhance_once_before_sampling_and_preserve_options(
    server, endpoint, enabled
):
    original = "A red teapot"
    if endpoint == "image":
        response = server.client.post(
            "/v1/images/generations",
            json={
                "prompt": original,
                "enhance_prompt": enabled,
                "seed": 123,
                "negative_prompt": "blur",
                "n": 2,
                "response_format": "b64_json",
            },
        )
    elif endpoint == "edit":
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(buffer, format="PNG")
        response = server.client.post(
            "/v1/images/edits",
            data={
                "prompt": original,
                "enhance_prompt": str(enabled).lower(),
                "seed": "123",
                "negative_prompt": "blur",
                "n": "2",
            },
            files={"image": ("reference.png", buffer.getvalue(), "image/png")},
        )
    else:
        server.args.pipeline_config = WanT2V480PConfig()
        payload = {
            "prompt": original,
            "seed": 123,
            "negative_prompt": "blur",
            "n": 2,
            "num_frames": 5,
        }
        if endpoint == "video_json":
            response = server.client.post(
                "/v1/videos", json={**payload, "enhance_prompt": enabled}
            )
        else:
            response = server.client.post(
                "/v1/videos",
                data={**payload, "enhance_prompt": str(enabled).lower()},
                files={"unused": ("empty", b"")},
            )
    assert response.status_code == 200, response.text
    expected = server.rewritten if enabled else original
    assert len(server.calls) == int(enabled)
    if endpoint.startswith("video"):
        assert response.json()["revised_prompt"] == (expected if enabled else None)

        # wait for the real background job, not a mocked dispatch
        async def wait_jobs():
            if video_api._VIDEO_JOB_TASKS:
                await asyncio.gather(*list(video_api._VIDEO_JOB_TASKS.values()))

        server.client.portal.call(wait_jobs)
    else:
        assert len(response.json()["data"]) == 2
        assert all(
            item["revised_prompt"] == expected for item in response.json()["data"]
        )
    assert len(server.batches) == 1
    batch = server.batches[0]
    assert batch.prompt == expected
    assert batch.seed == 123
    assert batch.negative_prompt == "blur"
    assert batch.num_outputs_per_prompt == 2
    if enabled:
        payload = server.calls[0]
        assert payload["model"] == "test-vlm"
        assert payload["max_tokens"] == 128
        assert payload["response_format"] == {"type": "json_object"}
        if endpoint == "edit":
            assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        else:
            assert json.loads(payload["messages"][1]["content"])["prompt"] == original


def test_unconfigured_enhancer_rejects_opt_in_without_generation(server):
    enhancer = server.app.state.prompt_enhancer
    server.app.state.prompt_enhancer = None
    try:
        response = server.client.post(
            "/v1/images/generations",
            json={"prompt": "a teapot", "enhance_prompt": True},
        )
        assert response.status_code == 400
        assert not server.batches
    finally:
        server.app.state.prompt_enhancer = enhancer


def test_realtime_does_not_silently_accept_prompt_enhancement():
    with pytest.raises(ValidationError):
        RealtimeVideoGenerationsRequest(
            type="init", prompt="a teapot", enhance_prompt=True
        )
