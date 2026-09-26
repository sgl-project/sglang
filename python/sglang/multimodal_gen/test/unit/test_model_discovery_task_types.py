# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai import common_api


def test_explicit_pipeline_discovery_does_not_require_diffusers_index(monkeypatch):
    class CustomPipeline:
        pipeline_name = "CustomPipeline"

    class MultiConfig(PipelineConfig):
        supported_task_types = (ModelTaskType.T2V, ModelTaskType.I2I)

    args = SimpleNamespace(
        model_path="/models/custom/dit.safetensors",
        pipeline_class_name="CustomPipeline",
        pipeline_config=MultiConfig(task_type=ModelTaskType.T2V),
        num_gpus=1,
        served_model_name="custom",
        backend="sglang",
        model_id=None,
    )

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError(
            "Explicit pipeline must bypass Diffusers model_index lookup"
        )

    monkeypatch.setattr(common_api, "get_model_info", unexpected_lookup)
    monkeypatch.setattr(common_api, "get_pipeline_class", lambda name: CustomPipeline)
    monkeypatch.setattr(common_api, "get_global_server_args", lambda: args)
    app = FastAPI()
    app.state.server_args = args
    app.include_router(http_server.health_router)
    app.include_router(common_api.router)
    with TestClient(app) as client:
        for endpoint in ("/models", "/v1/models", "/model_info"):
            response = client.get(endpoint)
            assert response.status_code == 200, response.text
            data = response.json()
            if endpoint == "/v1/models":
                data = data["data"][0]
            assert data["task_type"] == "T2V"
            assert data["supported_task_types"] == ["T2V", "I2I"]
        info = client.get("/model_info").json()
        assert info["architectures"] == ["CustomPipeline"]
        assert info["output_types"] == ["IMAGE", "VIDEO"]
        assert info["has_image_understanding"] is True
