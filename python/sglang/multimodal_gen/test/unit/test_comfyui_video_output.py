# SPDX-License-Identifier: Apache-2.0
"""SGLD video nodes: getting the generated video to a file ComfyUI can read."""

import importlib
import shutil
import subprocess
import sys
import types
from unittest import mock

import pytest

_UTILS = "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.utils"


@pytest.fixture
def utils(tmp_path):
    """Import the plugin's utils.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(tmp_path)
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.input = types.ModuleType("comfy_api.input")
    comfy_api.input.VideoInput = type("VideoInput", (), {})
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api.input,
    }
    with mock.patch.dict(sys.modules, stubs):
        # Re-import so utils binds these stubs, not ones another test installed.
        sys.modules.pop(_UTILS, None)
        yield importlib.import_module(_UTILS)
    sys.modules.pop(_UTILS, None)


def test_cloud_uploaded_video_is_fetched_from_its_url(utils, monkeypatch) -> None:
    """After a cloud upload the server drops the local file and returns
    file_path=None; the node used that empty path and produced no video."""
    response = mock.MagicMock()
    response.__enter__.return_value.iter_content = lambda chunk_size: [b"mp4"]
    monkeypatch.setattr(utils.requests, "get", lambda url, **kwargs: response)

    path = utils.resolve_video_path(None, "https://cdn.example.com/out.mp4?sig=1")
    assert path.endswith(".mp4")
    with open(path, "rb") as f:
        assert f.read() == b"mp4"
    assert utils.resolve_video_path("/local/out.mp4", None) == "/local/out.mp4"


def test_unreadable_video_path_fails_instead_of_saving_nothing(utils, tmp_path):
    """A remote server's file_path is not readable here; save_to silently
    wrote no file."""
    video = utils.SGLDVideoInput("/srv/remote/out.mp4", height=32, width=32)
    with pytest.raises(FileNotFoundError, match="readable from the ComfyUI process"):
        video.save_to(str(tmp_path / "saved.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_as_trimmed_returns_a_cut_copy(utils, tmp_path) -> None:
    """Current ComfyUI requires VideoInput.as_trimmed; it must cut the clip."""
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=2", str(src)],
        capture_output=True,
        check=True,
    )
    trimmed = utils.SGLDVideoInput(str(src), height=32, width=32).as_trimmed(0.0, 1.0)
    assert trimmed.video_path != str(src)
    assert (trimmed.height, trimmed.width) == (32, 32)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            trimmed.video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(probe.stdout) < 1.5
