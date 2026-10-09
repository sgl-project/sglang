"""Regression tests for issue #43190.

1. `SGLDVideoInput` must implement every abstract method current ComfyUI
   requires on `VideoInput` (as of ComfyUI #12107, `as_trimmed` is abstract).
2. The video-generating nodes must use the server's `url` when there is no
   local `file_path` (cloud upload) and must fail clearly rather than
   silently when neither is usable (remote server, no shared filesystem).
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path
from unittest import mock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_comfy_video_under_test"


def _install_comfy_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: "/tmp"
    folder_paths.folder_names_and_paths = {}
    sys.modules["folder_paths"] = folder_paths

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")

    class VideoInput(ABC):
        """Mirrors the real `comfy_api.input.VideoInput` abstract surface
        since ComfyUI #12107 added `as_trimmed` as abstract."""

        @abstractmethod
        def get_components(self):
            pass

        @abstractmethod
        def save_to(self, path, format=None, codec=None, metadata=None):
            pass

        @abstractmethod
        def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
            pass

    comfy_api_input.VideoInput = VideoInput
    comfy_api.input = comfy_api_input
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.input"] = comfy_api_input

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

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_detection"] = comfy.model_detection
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules["comfy.model_patcher"] = comfy_model_patcher


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
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

    utils = _load(f"{PKG}.utils", "utils.py")
    nodes = _load(f"{PKG}.nodes", "nodes.py")
    return server_api, utils, nodes


SERVER_API, UTILS, NODES = _load_plugin()
SGLDiffusionServerAPI = SERVER_API.SGLDiffusionServerAPI
SGLDiffusionGenerateVideo = NODES.SGLDiffusionGenerateVideo


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _DownloadResponse:
    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield b"fake-mp4-bytes"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_generate_video(response_payload, download_url=None):
    """Drive the real node+client; `requests.get` is shared process-wide, so
    one fake must serve both the job-status poll and the video download."""
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")

    def fake_post(url, json=None, headers=None, timeout=None):
        return _Response({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None, stream=False):
        if download_url is not None and url == download_url:
            return _DownloadResponse()
        return _Response({"id": "job-1", "status": "completed", **response_payload})

    node = SGLDiffusionGenerateVideo()
    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        return node.generate_video(sgld_client=client, positive_prompt="a cat")


def test_as_trimmed_is_required_by_current_comfyui():
    """Bug 1: SGLDVideoInput must be instantiable under current ComfyUI.

    ComfyUI #12107 made `as_trimmed` abstract; instantiation used to raise
    "Can't instantiate abstract class ... without ... 'as_trimmed'".
    """
    video_input = UTILS.SGLDVideoInput("/tmp/out.mp4", height=720, width=1280)
    # Negative resulting duration must return None per the VideoInput contract.
    assert video_input.as_trimmed(0.0, -1.0) is None


def test_as_trimmed_produces_a_real_trimmed_clip(tmp_path):
    """as_trimmed must cut the clip, not hand back the original untouched."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=2",
            str(src),
        ],
        capture_output=True,
        check=True,
    )

    with mock.patch(
        f"{PKG}.utils.folder_paths.get_temp_directory", return_value=str(tmp_path)
    ):
        video_input = UTILS.SGLDVideoInput(str(src), height=32, width=32)
        trimmed = video_input.as_trimmed(0.0, 1.0)

    assert trimmed is not None
    assert trimmed.video_path != str(src)
    assert os.path.exists(trimmed.video_path)


def test_cloud_upload_response_has_no_local_file_path():
    """Bug 2a: when cloud upload succeeds, file_path is None and url is set."""
    download_url = "https://cdn.example.com/out.mp4"
    video, video_path = _run_generate_video(
        {"file_path": None, "url": download_url}, download_url=download_url
    )

    # The node must not hand back a VideoInput pointed at a nonexistent local
    # path; it must fetch from `url` instead, into a real, readable file.
    assert video_path not in (None, "")
    assert os.path.exists(video_path)
    with open(video_path, "rb") as f:
        assert f.read() == b"fake-mp4-bytes"


def test_remote_server_file_path_is_not_locally_readable(tmp_path):
    """Bug 2b: file_path from a remote server is not on this machine.

    resolve_video_path trusts a present file_path (same-host servers are the
    common case), but the save step must fail clearly rather than silently
    writing nothing when that path turns out not to be readable here.
    """
    remote_only_path = "/srv/sglang/outputs/v2.mp4"
    video, video_path = _run_generate_video({"file_path": remote_only_path})
    assert video_path == remote_only_path

    dest = tmp_path / "saved.mp4"
    with pytest.raises(FileNotFoundError):
        video.save_to(str(dest))
    assert not dest.exists()
