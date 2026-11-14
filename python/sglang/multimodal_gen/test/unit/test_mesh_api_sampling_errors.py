# SPDX-License-Identifier: Apache-2.0
"""An invalid mesh sampling parameter must answer 400, as the image and video routes do.

`/v1/images/generations` and `/v1/videos` route parameter validation through
`build_sampling_params`, which maps `ValueError`/`TypeError` to HTTP 400. The mesh route
builds the same `SamplingParams` object directly, so before this test it answered 500.
"""

from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import mesh_api


async def _fake_save_image_to_path(img_src, path, uploads_root=None):
    return "outputs/uploads/fake-input.png"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(mesh_api.router)
    return TestClient(app)


def test_invalid_sampling_parameter_answers_400():
    with (
        patch.object(
            mesh_api,
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="dummy"),
        ),
        patch.object(mesh_api, "save_image_to_path", _fake_save_image_to_path),
        patch.object(
            mesh_api.SamplingParams,
            "from_user_sampling_params_args",
            side_effect=ValueError("num_inference_steps must be a positive int, got 0"),
        ),
    ):
        response = _client().post(
            "/v1/meshes",
            json={"input_image": "data:image/png;base64,iVBORw0KGgo="},
        )

    assert response.status_code == 400
    assert "num_inference_steps" in response.json()["detail"]
