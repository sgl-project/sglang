"""enhance_prompt: sent to the server, rewritten prompt returned as a STRING."""

import sys
from pathlib import Path
from unittest import mock

import pytest
import requests

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import OUTPUT_VIDEO, PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _client():
    return SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")


def _video_run(job_extra=None, **node_kwargs):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return FakeResponse({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None):
        job = {"id": "job-1", "status": "completed", "file_path": OUTPUT_VIDEO}
        job.update(job_extra or {})
        return FakeResponse(job)

    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        result = NODES.SGLDiffusionGenerateVideo().generate_video(
            sgld_client=_client(), positive_prompt="a cat", **node_kwargs
        )
    return captured, result


def _image_run(revised_prompt=None, **node_kwargs):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        captured.update(json)
        data = {"b64_json": PNG_B64, "revised_prompt": revised_prompt}
        return FakeResponse({"data": [data]})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        result = NODES.SGLDiffusionGenerateImage().generate_image(
            sgld_client=_client(), positive_prompt="a cat", **node_kwargs
        )
    return captured, result


def test_video_enhance_sends_flag_and_returns_rewrite():
    payload, result = _video_run(
        {"revised_prompt": "a tabby cat stretching on a sunlit windowsill"},
        enhance_prompt=True,
    )

    assert payload["enhance_prompt"] is True
    assert result[2] == "a tabby cat stretching on a sunlit windowsill"


def test_video_without_enhance_omits_flag_and_echoes_prompt():
    payload, result = _video_run()

    assert "enhance_prompt" not in payload
    assert result[2] == "a cat"


def test_image_enhance_sends_flag_and_returns_rewrite():
    payload, result = _image_run("a fluffy tabby, studio light", enhance_prompt=True)

    assert payload["enhance_prompt"] is True
    assert result[1] == "a fluffy tabby, studio light"


def test_image_without_enhance_omits_flag():
    payload, result = _image_run(revised_prompt=None)

    assert "enhance_prompt" not in payload
    assert result[1] == "a cat"


def test_audio_video_node_returns_revised_prompt_last():
    assert NODES.SGLDiffusionGenerateAudioVideo.RETURN_NAMES[-1] == "revised_prompt"
    assert (
        "enhance_prompt"
        in NODES.SGLDiffusionGenerateAudioVideo.INPUT_TYPES()["optional"]
    )


def test_new_outputs_are_appended_after_existing_ones():
    assert NODES.SGLDiffusionGenerateVideo.RETURN_NAMES[:2] == ("video", "video_path")
    assert NODES.SGLDiffusionGenerateImage.RETURN_NAMES[0] == "image"


def test_server_rejection_reason_reaches_the_user():
    response = FakeResponse(
        {"detail": "Prompt enhancement requires --prompt-enhancer-config"}, 400
    )
    response.raise_for_status = mock.Mock(
        side_effect=requests.exceptions.HTTPError("400 Bad Request", response=response)
    )

    with mock.patch(f"{PKG}.core.server_api.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="--prompt-enhancer-config"):
            NODES.SGLDiffusionGenerateVideo().generate_video(
                sgld_client=_client(), positive_prompt="a cat", enhance_prompt=True
            )
