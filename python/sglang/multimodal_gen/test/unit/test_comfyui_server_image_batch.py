# SPDX-License-Identifier: Apache-2.0
"""Server-mode image node: several images from one request as one IMAGE batch."""

import base64
import functools
import io
import sys
import types
from unittest import mock

import pytest
import torch
from PIL import Image


@functools.cache
def _load_nodes():
    """Import the real nodes.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    return nodes


def _png_b64(value: int, size=(8, 8)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _Client:
    """Records the request and answers with one solid image per entry."""

    def __init__(self, images):
        self.images = images
        self.requests = []

    def generate_image(self, **params):
        self.requests.append(params)
        return {"data": [{"b64_json": image} for image in self.images]}


def _generate(client, **kwargs):
    node = _load_nodes().SGLDiffusionGenerateImage()
    return node.generate_image(client, "a cat", **kwargs)[0]


def test_default_still_sends_one_image_request() -> None:
    client = _Client([_png_b64(10)])
    batch = _generate(client)
    assert client.requests[0]["n"] == 1
    assert batch.shape == (1, 8, 8, 3)


def test_num_images_returns_one_ordered_batch() -> None:
    client = _Client([_png_b64(v) for v in (0, 128, 255)])
    batch = _generate(client, num_images=3, seed=7)
    request = client.requests[0]
    assert request["n"] == 3
    assert request["seed"] == 7
    assert batch.shape == (3, 8, 8, 3)
    # Server order is kept: image i is the one generated with seed + i.
    assert batch[:, 0, 0, 0].tolist() == pytest.approx([0.0, 128 / 255, 1.0])


def test_fewer_images_than_requested_is_an_error() -> None:
    client = _Client([_png_b64(1), _png_b64(2)])
    with pytest.raises(
        RuntimeError, match="Requested 3 images but the server returned 2"
    ):
        _generate(client, num_images=3)


def test_images_of_different_sizes_are_an_error() -> None:
    client = _Client([_png_b64(1, (8, 8)), _png_b64(2, (16, 8))])
    with pytest.raises(RuntimeError, match="cannot be combined"):
        _generate(client, num_images=2)


@pytest.mark.parametrize("num_images", [0, 11])
def test_out_of_range_count_is_rejected_before_the_request(num_images) -> None:
    client = _Client([_png_b64(1)])
    with pytest.raises(ValueError, match="num_images must be between 1 and 10"):
        _generate(client, num_images=num_images)
    assert client.requests == []


def test_node_exposes_num_images_input() -> None:
    nodes = _load_nodes()
    spec = nodes.SGLDiffusionGenerateImage.INPUT_TYPES()["optional"]["num_images"]
    assert spec[0] == "INT"
    assert spec[1]["default"] == 1
    assert spec[1]["min"] == 1
    assert spec[1]["max"] == 10
    # ComfyUI IMAGE tensors are float32.
    batch = _generate(_Client([_png_b64(3)]))
    assert batch.dtype == torch.float32
