"""Checkpoint sweep: weights swapped in place, one prompt and seed, tiled grid."""

import base64
import io
import sys
from pathlib import Path
from unittest import mock

import pytest
import requests
import torch
from PIL import Image

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()

BASE_MODEL = "/models/base"
TEMPLATE = "/ckpts/run1/checkpoint-{step}"


def _png_b64(color):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeServer:
    """Records the order of weight swaps and generations; colors encode the step."""

    def __init__(self, fail_update_for=None, fail_generate_at=None):
        self.events = []
        self.generate_payloads = []
        self.update_urls = []
        self.fail_update_for = fail_update_for
        self.fail_generate_at = fail_generate_at
        self.loaded = BASE_MODEL

    def get(self, url, headers=None, timeout=None):
        assert url.endswith("/v1/model_info")
        return FakeResponse({"model_path": BASE_MODEL})

    def post(self, url, json=None, headers=None, timeout=None, **kw):
        if url.endswith("/update_weights_from_disk"):
            self.update_urls.append(url)
            self.events.append(("update", json["model_path"]))
            if json["model_path"] == self.fail_update_for:
                return FakeResponse({"success": False, "message": "no such dir"})
            self.loaded = json["model_path"]
            return FakeResponse({"success": True, "message": "ok"})
        assert url.endswith("/v1/images/generations")
        self.events.append(("generate", self.loaded))
        self.generate_payloads.append(json)
        if len(self.generate_payloads) == self.fail_generate_at:
            raise RuntimeError("server exploded")
        shade = len(self.generate_payloads) * 40
        return FakeResponse({"data": [{"b64_json": _png_b64((shade, 0, 0))}]})


def _sweep(server, **overrides):
    client = SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    kwargs = dict(
        sgld_client=client,
        positive_prompt="a lighthouse",
        checkpoint_template=TEMPLATE,
        checkpoint_steps="500, 1000, 1500",
        seed=7,
        width=256,
        height=256,
        columns=2,
    )
    kwargs.update(overrides)
    with (
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=server.get),
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=server.post),
    ):
        return NODES.SGLDiffusionCheckpointSweep().sweep(**kwargs)


def test_swaps_each_checkpoint_then_restores_the_original():
    server = FakeServer()

    _sweep(server)

    assert server.events == [
        ("update", "/ckpts/run1/checkpoint-500"),
        ("generate", "/ckpts/run1/checkpoint-500"),
        ("update", "/ckpts/run1/checkpoint-1000"),
        ("generate", "/ckpts/run1/checkpoint-1000"),
        ("update", "/ckpts/run1/checkpoint-1500"),
        ("generate", "/ckpts/run1/checkpoint-1500"),
        ("update", BASE_MODEL),
    ]


def test_every_cell_uses_the_same_prompt_and_seed():
    server = FakeServer()

    _sweep(server)

    assert len(server.generate_payloads) == 3
    assert all(p == server.generate_payloads[0] for p in server.generate_payloads)
    assert server.generate_payloads[0]["seed"] == 7
    assert server.generate_payloads[0]["prompt"] == "a lighthouse"


def test_weights_endpoint_has_no_v1_prefix():
    server = FakeServer()

    _sweep(server)

    assert all(
        url == "http://127.0.0.1:30010/update_weights_from_disk"
        for url in server.update_urls
    )


def test_grid_layout_follows_step_order_and_pads_empty_cells():
    server = FakeServer()

    grid, images, log = _sweep(server)

    assert images.shape == (3, 4, 4, 3)
    assert grid.shape == (1, 8, 8, 3)
    # step order is left to right, top to bottom; the 4th cell is padding
    assert grid[0, 0, 0, 0] < grid[0, 0, 4, 0] < grid[0, 4, 0, 0]
    assert grid[0, 4, 4].abs().max() == 0
    assert log.splitlines()[0] == "step 500: /ckpts/run1/checkpoint-500"
    assert log.splitlines()[-1] == f"restored: {BASE_MODEL}"


def test_original_weights_are_restored_when_generation_fails():
    server = FakeServer(fail_generate_at=2)

    with pytest.raises(RuntimeError, match="step 1000"):
        _sweep(server)

    assert server.events[-1] == ("update", BASE_MODEL)
    assert server.loaded == BASE_MODEL


def test_missing_checkpoint_names_the_step_and_still_restores():
    server = FakeServer(fail_update_for="/ckpts/run1/checkpoint-1000")

    with pytest.raises(RuntimeError, match=r"step 1000.*no such dir"):
        _sweep(server)

    assert server.events[-1] == ("update", BASE_MODEL)


def test_restore_failure_is_reported_without_losing_the_results():
    server = FakeServer(fail_update_for=BASE_MODEL)

    grid, images, log = _sweep(server)

    assert images.shape[0] == 3
    assert "WARNING: could not restore" in log


def test_restore_can_be_disabled():
    server = FakeServer()

    _sweep(server, restore_original=False)

    assert server.events[-1] == ("generate", "/ckpts/run1/checkpoint-1500")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"checkpoint_template": "/ckpts/final"}, "{step}"),
        ({"checkpoint_steps": ""}, "No checkpoint steps"),
        ({"checkpoint_steps": "500, abc"}, "Not a checkpoint step"),
        ({"seed": -1}, "seed"),
        ({"positive_prompt": ""}, "Prompt"),
    ],
)
def test_bad_input_is_rejected_before_touching_the_server(overrides, message):
    server = FakeServer()

    with pytest.raises(ValueError, match=message):
        _sweep(server, **overrides)

    assert server.events == []


def test_step_list_drops_repeats_and_keeps_order():
    assert UTILS.parse_step_list("1000, 500,1000\n1500") == [1000, 500, 1500]


def test_step_list_limit():
    with pytest.raises(ValueError, match="At most"):
        UTILS.parse_step_list(",".join(str(i) for i in range(100)))


def test_grid_clamps_columns_to_image_count():
    images = torch.ones(2, 3, 3, 3)

    assert UTILS.make_image_grid(images, columns=9).shape == (1, 3, 6, 3)


@pytest.mark.parametrize("body", [["not", "an", "object"], "plain text", None])
def test_non_object_error_body_does_not_hide_the_failure(body):
    response = FakeResponse(body, 500)
    response.raise_for_status = mock.Mock(
        side_effect=requests.exceptions.HTTPError("500 Server Error", response=response)
    )

    client = SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    with mock.patch(f"{PKG}.core.server_api.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="500 Server Error"):
            client.update_weights_from_disk("/ckpts/run1/checkpoint-500")
