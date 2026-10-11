"""Generic audio-video node: AUDIO extracted from the returned MP4, extra fields merged."""

import fractions
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

av = pytest.importorskip("av")

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()

SAMPLE_RATE = 24000


def _write_mp4(path, with_audio):
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=8)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        audio = out.add_stream("aac", rate=SAMPLE_RATE) if with_audio else None
        for _ in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        if audio is not None:
            tone = np.sin(np.linspace(0, 440 * 2 * np.pi, SAMPLE_RATE)).astype(
                np.float32
            )
            frame = av.AudioFrame.from_ndarray(
                tone[None, :], format="fltp", layout="mono"
            )
            frame.sample_rate = SAMPLE_RATE
            frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
            for packet in audio.encode(frame):
                out.mux(packet)
            for packet in audio.encode(None):
                out.mux(packet)


def _run_node(video_file, **node_kwargs):
    client = SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return FakeResponse({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(
            {"id": "job-1", "status": "completed", "file_path": str(video_file)}
        )

    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        result = NODES.SGLDiffusionGenerateAudioVideo().generate_audio_video(
            sgld_client=client, positive_prompt="a drummer", **node_kwargs
        )
    return captured, result


def test_returns_video_audio_and_path(tmp_path):
    clip = tmp_path / "out.mp4"
    _write_mp4(clip, with_audio=True)

    _, (video, audio, video_path) = _run_node(clip)

    assert video_path == str(clip)
    assert audio["sample_rate"] == SAMPLE_RATE
    waveform = audio["waveform"]
    assert waveform.dim() == 3 and waveform.shape[0] == 1
    assert waveform.shape[2] > SAMPLE_RATE // 2
    assert waveform.abs().max() > 0.1


def test_clip_without_audio_yields_silent_audio(tmp_path):
    clip = tmp_path / "silent.mp4"
    _write_mp4(clip, with_audio=False)

    _, (_, audio, _) = _run_node(clip)

    assert audio["waveform"].shape == (1, 1, 1)
    assert audio["waveform"].abs().max() == 0


def test_extra_request_json_is_merged_and_wins(tmp_path):
    clip = tmp_path / "out.mp4"
    _write_mp4(clip, with_audio=False)

    payload, _ = _run_node(
        clip,
        steps=20,
        extra_request_json='{"flow_shift": 12.0, "num_inference_steps": 4}',
    )

    assert payload["flow_shift"] == 12.0
    assert payload["num_inference_steps"] == 4


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"text"'])
def test_bad_extra_request_json_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="extra_request_json"):
        _run_node(tmp_path / "unused.mp4", extra_request_json=bad)


def test_node_keeps_video_node_inputs_and_is_registered():
    inputs = NODES.SGLDiffusionGenerateAudioVideo.INPUT_TYPES()
    assert {"image", "video", "extra_request_json"} <= set(inputs["optional"])
    assert (
        "extra_request_json"
        not in NODES.SGLDiffusionGenerateVideo.INPUT_TYPES()["optional"]
    )
    assert "SGLDiffusionGenerateAudioVideo" in NODES.NODE_CLASS_MAPPINGS
    assert NODES.SGLDiffusionGenerateVideo.RETURN_TYPES == ("VIDEO", "STRING")
