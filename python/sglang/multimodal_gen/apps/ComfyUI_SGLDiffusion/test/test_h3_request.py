"""Tests for the MiniMax-H3 ComfyUI node's request shape.

These exercise the real node and the real server-API client together and mock
only HTTP, so a change on either side of the payload contract fails here.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_comfy_under_test"


def _install_comfy_stubs() -> None:
    """Stub the ComfyUI runtime modules the plugin imports at module scope.

    The plugin only ever runs inside ComfyUI, so these packages are absent in
    a plain checkout; stubbing them keeps the request-shape contract testable
    without a ComfyUI install or a GPU.
    """
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: "/tmp"
    sys.modules.setdefault("folder_paths", folder_paths)

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")

    class VideoInput:
        pass

    comfy_api_input.VideoInput = VideoInput
    comfy_api.input = comfy_api_input
    sys.modules.setdefault("comfy_api", comfy_api)
    sys.modules.setdefault("comfy_api.input", comfy_api_input)

    comfy = types.ModuleType("comfy")
    comfy.model_detection = types.ModuleType("comfy.model_detection")
    comfy.model_management = types.ModuleType("comfy.model_management")

    comfy_utils = types.ModuleType("comfy.utils")
    for name in (
        "calculate_parameters",
        "load_torch_file",
        "state_dict_prefix_replace",
        "unet_to_diffusers",
    ):
        setattr(comfy_utils, name, lambda *a, **k: None)
    comfy.utils = comfy_utils

    comfy_model_patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, *a, **k):
            pass

    comfy_model_patcher.ModelPatcher = ModelPatcher
    comfy.model_patcher = comfy_model_patcher

    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.model_detection", comfy.model_detection)
    sys.modules.setdefault("comfy.model_management", comfy.model_management)
    sys.modules.setdefault("comfy.utils", comfy_utils)
    sys.modules.setdefault("comfy.model_patcher", comfy_model_patcher)


def _load(module_name: str, relative_path: str):
    """Load one plugin source file into the synthetic package."""
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
    """Load the node and its client without importing the sglang package root.

    `sglang/__init__` pulls in the whole LLM serving stack, none of which this
    contract depends on. `core.generator` is replaced by a placeholder because
    it reaches for ComfyUI's model machinery and the node never calls it on the
    server path.
    """
    _install_comfy_stubs()

    package = types.ModuleType(PKG)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG] = package

    server_api = _load(f"{PKG}.core.server_api", "core/server_api.py")

    core = types.ModuleType(f"{PKG}.core")
    core.__path__ = [str(PLUGIN_DIR / "core")]
    core.SGLDiffusionServerAPI = server_api.SGLDiffusionServerAPI
    core.SGLDiffusionGenerator = object
    sys.modules[f"{PKG}.core"] = core

    _load(f"{PKG}.utils", "utils.py")
    nodes = _load(f"{PKG}.nodes", "nodes.py")
    return server_api, nodes


SERVER_API, NODES = _load_plugin()
SGLDiffusionServerAPI = SERVER_API.SGLDiffusionServerAPI
SGLDiffusionGenerateH3 = NODES.SGLDiffusionGenerateH3

RESOLVED_SIZE = "1344x768"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _run_node(**node_kwargs):
    """Drive the node through the real client and capture the POST payload."""
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return _Response({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None):
        return _Response(
            {
                "id": "job-1",
                "status": "completed",
                "size": RESOLVED_SIZE,
                "file_path": "/tmp/out.mp4",
            }
        )

    node = SGLDiffusionGenerateH3()
    with mock.patch(
        f"{PKG}.core.server_api.requests.post",
        side_effect=fake_post,
    ), mock.patch(
        f"{PKG}.core.server_api.requests.get",
        side_effect=fake_get,
    ), mock.patch(
        f"{PKG}.nodes.get_image_path",
        side_effect=lambda image: "/tmp/frame.png",
    ):
        result = node.generate(sgld_client=client, **node_kwargs)
    return captured, result


def _image():
    return torch.zeros(1, 8, 8, 3)


def test_t2va_sends_task_target_and_flow_shifts():
    payload, _ = _run_node(positive_prompt="a cat", task="t2va")

    assert payload["task"] == "t2va"
    assert payload["conditions"] == []
    assert payload["target"] == {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }
    assert payload["flow_shift"] == 12.0
    assert payload["audio_flow_shift"] == 3.0


def test_fl2va_maps_keyframes_to_frame_indices():
    payload, _ = _run_node(
        positive_prompt="continue the shot",
        task="fl2va",
        first_frame=_image(),
        last_frame=_image(),
    )

    assert [c["role"] for c in payload["conditions"]] == ["keyframe", "keyframe"]
    assert [c["frame_index"] for c in payload["conditions"]] == [0, -1]
    assert all(c["uri"].startswith("file:///") for c in payload["conditions"])


def test_ref2va_preserves_modality_order_for_prompt_tags():
    payload, _ = _run_node(
        positive_prompt="use <Picture 1> and <Audio 1>",
        task="ref2va",
        reference_image=_image(),
        reference_video="/data/clip.mp4",
        reference_audio="/data/voice.mp3",
    )

    assert [c["type"] for c in payload["conditions"]] == ["image", "video", "audio"]
    assert {c["role"] for c in payload["conditions"]} == {"reference"}


def test_remote_reference_urls_pass_through_unchanged():
    url = "https://example.com/clip.mp4"
    payload, _ = _run_node(
        positive_prompt="follow <Video 1>",
        task="ref2va",
        reference_video=url,
    )

    assert payload["conditions"][0]["uri"] == url


def test_node_reports_the_server_resolved_canvas():
    _, (video, video_path) = _run_node(positive_prompt="a cat", task="t2va")

    assert video.get_dimensions() == (1344, 768)
    assert video_path == "/tmp/out.mp4"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"task": "fl2va"}, "fl2va requires"),
        ({"task": "ref2va"}, "ref2va requires"),
        ({"task": "t2va", "reference_video": "/data/clip.mp4"}, "t2va takes no"),
    ],
)
def test_task_and_conditioning_must_agree(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _run_node(positive_prompt="a cat", **kwargs)


def test_extra_fields_win_over_generic_defaults():
    """A model's own field must not be shadowed by a same-named generic default."""
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return _Response({"id": "job-1"})

    with mock.patch(
        f"{PKG}.core.server_api.requests.post",
        side_effect=fake_post,
    ), mock.patch(
        f"{PKG}.core.server_api.requests.get",
        side_effect=lambda *a, **k: _Response(
            {"id": "job-1", "status": "completed", "size": RESOLVED_SIZE}
        ),
    ):
        client.generate_video(
            prompt="a cat",
            size="720x1280",
            extra_fields={"size": "1344x768", "task": "t2va"},
        )

    assert captured["size"] == "1344x768"
    assert captured["task"] == "t2va"


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_payload_validates_against_the_server_request_model(task):
    """The node's payload must satisfy the schema the server actually parses.

    The other tests mock HTTP, so they would still pass if a field were
    misnamed or mistyped. This one feeds the captured payload to
    VideoGenerationsRequest, closing that gap without a running server.
    """
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        VideoGenerationsRequest,
    )

    conditioning = {
        "t2va": {},
        "fl2va": {"first_frame": _image()},
        "ref2va": {"reference_image": _image()},
    }[task]
    payload, _ = _run_node(positive_prompt="a cat", task=task, **conditioning)

    request = VideoGenerationsRequest(**payload)

    # the H3 fields ride through as extras; losing them silently would leave a
    # valid request that generates the wrong thing
    assert request.task == task
    assert request.target["short_edge"] == payload["target"]["short_edge"]
    assert len(request.conditions) == len(payload["conditions"])
