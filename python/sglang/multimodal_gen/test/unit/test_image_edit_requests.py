import asyncio
import base64
import io
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI
from PIL import Image
from starlette.requests import Request

from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    MULTI_FRAME_I2I_sampling_params,
)


def _png(mode):
    stream = io.BytesIO()
    Image.new(mode, (2, 2)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("direct_url", [True, False])
def test_edit_forwards_sampling_parameters(tmp_path, monkeypatch, direct_url):
    image = _png("RGBA")
    local = tmp_path / "input.png"
    local.write_bytes(image)
    urls = ["https://example.org/a.png", "https://example.org/b.png"]
    params = replace(
        MULTI_FRAME_I2I_sampling_params,
        image_path=urls if direct_url else local,
        direct_url_test=direct_url,
        extras={"quality": "exact", "num_inference_steps": 8, "seed": 0},
    )

    async def edit(request: httpx.Request):
        async def receive():
            return {"type": "http.request", "body": request.read()}

        headers = [(key.lower(), value) for key, value in request.headers.raw]
        assert request.url.path == "/v1/images/edits"
        async with Request(
            {"type": "http", "headers": headers}, receive
        ).form() as form:
            assert form["quality"] == "exact"
            assert form["num_inference_steps"] == "8"
            assert form["seed"] == "0"
            assert form["num_frames"] == "4"
            if direct_url:
                assert (form.getlist("url[]") or form.getlist("url")) == urls
            else:
                (uploaded,) = form.getlist("image[]") or form.getlist("image")
                assert await uploaded.read() == image
        return httpx.Response(
            200,
            json={
                "id": "result",
                "created": 1,
                "data": [{"b64_json": base64.b64encode(image).decode()}] * 4,
            },
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack",
        Mock(),
    )
    with OpenAI(
        base_url="http://testserver/v1",
        api_key="test",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda req: asyncio.run(edit(req)))
        ),
        max_retries=0,
    ) as client:
        generate = get_generate_fn("test-model", "image", params)
        assert generate("layered", client) == ("result", image)


@pytest.mark.parametrize(
    "modes, message",
    [(["RGBA"] * 3, "expected 4 layers"), (["RGBA"] * 3 + ["RGB"], "layer 3")],
)
def test_layered_edit_rejects_incomplete_output(modes, message):
    data = [
        SimpleNamespace(b64_json=base64.b64encode(_png(mode)).decode())
        for mode in modes
    ]
    client = Mock()
    client.images.with_raw_response.edit.return_value.parse.return_value.data = data
    generate = get_generate_fn("test-model", "image", MULTI_FRAME_I2I_sampling_params)
    with pytest.raises(AssertionError, match=message):
        generate("layered", client)
