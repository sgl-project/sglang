"""Video-to-video: the generic video node forwards a VIDEO input to the server.

A local clip is uploaded as `video_reference`; anything else goes as `video_path`.
"""

import sys
from pathlib import Path
from unittest import mock

import torch

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import OUTPUT_VIDEO, PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()


def _run_node(**node_kwargs):
    client = SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, files=None, data=None):
        captured.update(json or data)
        captured["uploads"] = {
            key: (name, fh.read()) for key, (name, fh, _) in (files or {}).items()
        }
        return FakeResponse({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(
            {"id": "job-1", "status": "completed", "file_path": OUTPUT_VIDEO}
        )

    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        NODES.SGLDiffusionGenerateVideo().generate_video(
            sgld_client=client, positive_prompt="restyle as oil painting", **node_kwargs
        )
    return captured


def test_local_video_input_is_uploaded(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    video = UTILS.SGLDVideoInput(str(src), height=480, width=832)

    payload = _run_node(video=video)

    assert payload["uploads"] == {"video_reference": ("clip.mp4", b"x")}
    assert "video_path" not in payload
    assert "input_reference" not in payload


def test_no_video_input_omits_video_path():
    payload = _run_node()
    assert "video_path" not in payload and not payload["uploads"]


def test_video_and_image_inputs_are_both_forwarded(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    video = UTILS.SGLDVideoInput(str(src), height=480, width=832)
    image = torch.rand(1, 8, 8, 3)

    with mock.patch(f"{PKG}.nodes.get_image_path", return_value="/tmp/ref.png"):
        payload = _run_node(video=video, image=image)

    assert payload["uploads"] == {"video_reference": ("clip.mp4", b"x")}
    # not a file on this machine, so it is sent as a string
    assert payload["input_reference"] == "/tmp/ref.png"


def test_foreign_video_is_written_to_temp_mp4():
    class OtherVideo(UTILS.VideoInput):
        def save_to(self, path, **kwargs):
            with open(path, "wb") as f:
                f.write(b"mp4")

    path = UTILS.get_video_path(OtherVideo())

    assert path.endswith(".mp4")
    assert open(path, "rb").read() == b"mp4"


def test_missing_sgld_video_file_is_rewritten_not_forwarded(tmp_path):
    gone = tmp_path / "gone.mp4"
    video = UTILS.SGLDVideoInput(str(gone), height=480, width=832)
    video.save_to = lambda path, **kw: open(path, "wb").write(b"mp4")

    assert UTILS.get_video_path(video) != str(gone)
