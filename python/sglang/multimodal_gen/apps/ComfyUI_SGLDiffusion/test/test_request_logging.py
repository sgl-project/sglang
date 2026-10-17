"""Server Mode nodes log the exact request they send, without the API key."""

import json
import logging
import sys
from pathlib import Path
from unittest import mock

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import OUTPUT_VIDEO, PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()

SECRET = "sk-secret-key"
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _client():
    return SERVER_API.SGLDiffusionServerAPI(
        base_url="http://127.0.0.1:30010", api_key=SECRET
    )


def _logged(caplog):
    lines = [
        r.getMessage()
        for r in caplog.records
        if "SGLDiffusion request" in r.getMessage()
    ]
    assert len(lines) == 1
    return lines[0]


def test_image_request_is_logged(caplog):
    resp = FakeResponse({"data": [{"b64_json": PNG_B64}]})
    with (
        caplog.at_level(logging.INFO),
        mock.patch(f"{PKG}.core.server_api.requests.post", return_value=resp),
    ):
        NODES.SGLDiffusionGenerateImage().generate_image(
            sgld_client=_client(), positive_prompt="a cat", seed=7
        )
    line = _logged(caplog)
    assert "/images/generations" in line
    body = json.loads(line[line.index("{") :])
    assert body["prompt"] == "a cat" and body["seed"] == 7
    assert SECRET not in line


def test_video_request_is_logged(caplog):
    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(
            {"id": "job-1", "status": "completed", "file_path": OUTPUT_VIDEO}
        )

    with (
        caplog.at_level(logging.INFO),
        mock.patch(
            f"{PKG}.core.server_api.requests.post",
            return_value=FakeResponse({"id": "job-1"}),
        ),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        NODES.SGLDiffusionGenerateVideo().generate_video(
            sgld_client=_client(), positive_prompt="a cat"
        )
    line = _logged(caplog)
    assert "/videos" in line
    assert json.loads(line[line.index("{") :])["prompt"] == "a cat"
    assert SECRET not in line


def test_long_values_are_shortened(caplog):
    long_value = "A" * 5000
    with caplog.at_level(logging.INFO):
        SERVER_API._log_request("http://x/videos", {"input_reference": long_value})
    line = _logged(caplog)
    assert long_value not in line and "5000 chars" in line
