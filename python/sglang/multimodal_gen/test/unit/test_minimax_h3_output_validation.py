# SPDX-License-Identifier: Apache-2.0
"""The final MiniMax H3 MP4 check must reach ffprobe's verdict without spawning it.

Each fixture breaks one property the check reads, so matching verdicts show the
in-process reader supplies every field that ffprobe did.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
needs_pyav = pytest.mark.skipif(video_adapter.av is None, reason="needs PyAV")

EXPECTED = {"expected_frame_count": 48, "expected_size": (320, 192)}
VALID_FIELDS = {"size": "320x192", "seconds": "2"}
X264 = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")

# name -> (encoder overrides, expected error substring or None when valid)
CASES = {
    "valid": ({}, None),
    "sample_rate": ({"audio_rate": 48000}, "sample rate"),
    "mono": ({"channels": 1}, "stereo"),
    "silent": ({"audio": False}, "exactly one video stream"),
    "frame_rate": ({"rate": 25}, "frame rate"),
    "frame_count": ({"frames": 47, "audio_seconds": 47 / 24}, "frame count"),
    "size": ({"size": "336x192"}, "size does not match"),
    "av_drift": ({"audio_seconds": 1.0}, "drift"),
    "video_codec": ({"video": ("-c:v", "mpeg4")}, "video codec"),
    "audio_codec": ({"audio_codec": "ac3"}, "audio codec"),
    "pixel_format": (
        {"video": ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p")},
        "pixel format",
    ),
    "matroska": ({"suffix": ".mkv"}, "frame count"),
    "subtitle_stream": ({"subtitle": True}, "exactly one video stream"),
}
UNREADABLE = ("truncated", "missing")


def _encode(
    directory: Path,
    name: str,
    *,
    size: str = "320x192",
    rate: int = 24,
    frames: int = 48,
    audio: bool = True,
    audio_rate: int = 32000,
    channels: int = 2,
    audio_seconds: float = 2.0,
    audio_codec: str = "aac",
    video: tuple[str, ...] = X264,
    subtitle: bool = False,
    suffix: str = ".mp4",
) -> str:
    path = directory / (name + suffix)
    command = ["ffmpeg", "-y", "-v", "error"]
    # end the video input; -frames:v can stop ffmpeg before the audio drains
    command += [
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={rate},trim=end_frame={frames}",
    ]
    maps = ["-map", "0:v"]
    if audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={audio_rate}:duration={audio_seconds}",
        ]
        maps += ["-map", "1:a"]
    if subtitle:
        srt = directory / "caption.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\ncaption\n")
        command += ["-i", str(srt)]
        maps += ["-map", f"{2 if audio else 1}:s", "-c:s", "mov_text"]
    command += maps + [*video]
    if audio:
        command += ["-c:a", audio_codec, "-ac", str(channels)]
    subprocess.run(command + [str(path)], check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("h3_outputs")
    paths = {
        name: _encode(directory, name, **overrides)
        for name, (overrides, _) in CASES.items()
    }
    data = Path(paths["valid"]).read_bytes()
    (directory / "truncated.mp4").write_bytes(data[: len(data) // 2])
    paths["truncated"] = str(directory / "truncated.mp4")
    paths["missing"] = str(directory / "missing.mp4")
    return paths


def _verdict(path: str):
    try:
        return video_adapter._probe_minimax_h3_output_fields(path, **EXPECTED)
    except RuntimeError as exc:
        return f"RuntimeError: {exc}"


def _use_ffprobe(monkeypatch):
    monkeypatch.setattr(
        video_adapter, "_read_output_media", video_adapter._ffprobe_output_media
    )


@pytest.mark.parametrize("name", list(CASES))
def test_each_fixture_trips_its_check(outputs, name, monkeypatch):
    _use_ffprobe(monkeypatch)
    verdict = _verdict(outputs[name])
    error = CASES[name][1]
    if error is None:
        assert verdict == VALID_FIELDS
    else:
        assert error in verdict


@needs_pyav
@pytest.mark.parametrize("name", [*CASES, *UNREADABLE])
def test_in_process_read_matches_ffprobe(outputs, name, monkeypatch):
    in_process = _verdict(outputs[name])
    _use_ffprobe(monkeypatch)
    ffprobe = _verdict(outputs[name])
    if name in UNREADABLE:
        # the two readers word their parse errors differently
        assert str(in_process).startswith("RuntimeError")
        assert str(ffprobe).startswith("RuntimeError")
    else:
        assert in_process == ffprobe


@needs_pyav
def test_validation_spawns_no_process(outputs, monkeypatch):
    def run(command, *args, **kwargs):
        raise AssertionError(f"spawned {command[0]}")

    adapter = video_adapter.MiniMaxH3VideoModelAdapter()
    shape = {"frame_count": 48, "width": 320, "height": 192}
    monkeypatch.setattr(adapter, "_resolved_shape", lambda batch: shape)
    monkeypatch.setattr(video_adapter.subprocess, "run", run)
    fields = adapter.validate_final_outputs_sync(
        [outputs["valid"]] * 2, SimpleNamespace(num_outputs_per_prompt=2)
    )
    assert fields == VALID_FIELDS


def test_falls_back_to_ffprobe_without_pyav(outputs, monkeypatch):
    spawned = []
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        spawned.append(command[0])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(video_adapter, "av", None)
    monkeypatch.setattr(video_adapter.subprocess, "run", run)
    assert _verdict(outputs["valid"]) == VALID_FIELDS
    assert spawned == ["ffprobe"]
