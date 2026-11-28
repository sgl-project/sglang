import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.entrypoints.openai import common_api
from sglang.multimodal_gen.runtime.entrypoints.openai.common_api import (
    DiffusionModelCard,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _video_job_from_sampling,
)


def test_model_list_and_retrieve_use_served_model_name():
    server_args = SimpleNamespace(
        model_path="/private/checkpoint",
        served_model_name="public-model",
    )

    def build_model_card(_server_args, model_name):
        return DiffusionModelCard(id=model_name, root=model_name)

    with (
        patch.object(common_api, "get_global_server_args", return_value=server_args),
        patch.object(common_api, "_build_model_card", side_effect=build_model_card),
    ):
        models = asyncio.run(common_api.available_models())
        model = asyncio.run(common_api.retrieve_model("public-model"))
        missing = asyncio.run(common_api.retrieve_model("/private/checkpoint"))

    assert models["data"][0]["id"] == "public-model"
    assert model["id"] == "public-model"
    assert missing.status_code == 404


def test_video_job_uses_served_model_name_unless_requested():
    sampling = SimpleNamespace(
        width=512,
        height=512,
        num_frames=49,
        fps=24,
        output_file_path=lambda: "/tmp/output.mp4",
    )
    request = VideoGenerationsRequest(prompt="test")

    job = _video_job_from_sampling("request-id", request, sampling, "public-model")
    explicit_job = _video_job_from_sampling(
        "request-id",
        request.model_copy(update={"model": "requested-model"}),
        sampling,
        "public-model",
    )

    assert job["model"] == "public-model"
    assert explicit_job["model"] == "requested-model"
