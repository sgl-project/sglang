# SPDX-License-Identifier: Apache-2.0
"""The configured x264 preset has to reach the encoder, not just the command.

libx264 records the options it resolved into the mp4 it writes, so a real encode
can be checked against a reference encode made with the preset spelled out. That
catches the failure this guards against -- the preset never being passed, and
ffmpeg silently applying its own default.
"""

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    X264_PRESET,
    MaterializedOutput,
    save_materialized_output,
    save_outputs,
)

FPS = 8
FRAMES = 8
SIZE = 64
# Options libx264 derives from the preset alone, so a reference encode pins them
# without hard-coding values that move with the x264 build.
PRESET_DERIVED_KEYS = ("subme", "ref", "rc_lookahead", "me", "trellis")


def _x264_options(path) -> dict[str, str]:
    """The `options:` line libx264 embeds in its own output."""
    blob = path.read_bytes()
    start = blob.find(b"x264 - core")
    assert start >= 0, "libx264 did not stamp its settings into the file"
    text = blob[start : blob.find(b"\x00", start)].decode("utf-8", "replace")
    _, _, options = text.partition("options: ")
    return dict(kv.split("=", 1) for kv in options.split() if "=" in kv)


def _reference_encode(tmp_path, frames, preset):
    out = tmp_path / f"ref_{preset}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{SIZE}x{SIZE}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        input=frames.tobytes(),
        check=True,
    )
    return out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_saved_video_carries_the_configured_preset(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    sample = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

    saved = tmp_path / "clip.mp4"
    paths = save_outputs([sample], DataType.VIDEO, FPS, True, lambda _idx: str(saved))

    assert paths == [str(saved)] and saved.exists()
    got = _x264_options(saved)
    expected = _x264_options(_reference_encode(tmp_path, frames, X264_PRESET))
    for key in PRESET_DERIVED_KEYS:
        assert got.get(key) == expected.get(key), f"{key} does not match {X264_PRESET}"

    if X264_PRESET != "medium":
        # ffmpeg's implicit default, i.e. what a dropped -preset would give.
        default = _x264_options(_reference_encode(tmp_path, frames, "medium"))
        assert any(got.get(k) != default.get(k) for k in PRESET_DERIVED_KEYS), (
            "encode is indistinguishable from ffmpeg's default preset"
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize(
    "device",
    [
        # host and CUDA tensors use the same system ffmpeg encoder
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="direct save needs CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("preset", ["ultrafast", "slow"])
def test_request_preset_reaches_the_encoder(tmp_path, device, preset):
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, (FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    sample = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

    saved = tmp_path / "clip.mp4"
    save_outputs(
        [sample.to(device)],
        DataType.VIDEO,
        FPS,
        True,
        lambda _idx: str(saved),
        x264_preset=preset,
    )

    got = _x264_options(saved)
    expected = _x264_options(_reference_encode(tmp_path, frames, preset))
    for key in PRESET_DERIVED_KEYS:
        assert got.get(key) == expected.get(key), f"{key} does not match {preset}"


def test_unknown_preset_is_rejected():
    assert SamplingParams(x264_preset="ultrafast").x264_preset == "ultrafast"
    assert SamplingParams().x264_preset is None
    with pytest.raises(ValueError, match="x264_preset must be one of"):
        SamplingParams(x264_preset="lightspeed")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_saved_video_decodes_to_every_frame(tmp_path):
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    sample = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

    saved = tmp_path / "clip.mp4"
    save_outputs([sample], DataType.VIDEO, FPS, True, lambda _idx: str(saved))

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,codec_name",
            "-of",
            "csv=p=0",
            str(saved),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    codec, count = probe.split(",")
    assert codec == "h264"
    assert int(count) == FRAMES


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("compression", [10, 100])
def test_materialized_video_uses_system_ffmpeg(
    tmp_path, monkeypatch, with_audio, compression
):
    def reject_imageio(*args, **kwargs):
        raise AssertionError("video output must use system ffmpeg")

    monkeypatch.setattr(output_utils.imageio, "mimsave", reject_imageio)
    rng = np.random.default_rng(3)
    # non-contiguous RGB and non-macroblock dimensions exercise the raw pipe and scaling
    frames = rng.integers(0, 256, (FRAMES, 30, 34, 3), dtype=np.uint8)[:, :, ::-1]
    saved = tmp_path / "materialized.mp4"
    save_materialized_output(
        MaterializedOutput(
            sample=None,
            frames=list(frames),
            audio=np.zeros((32000, 2), dtype=np.float32) if with_audio else None,
            fps=FPS,
        ),
        DataType.VIDEO,
        str(saved),
        audio_sample_rate=32000,
        output_compression=compression,
        x264_preset="ultrafast",
    )
    streams = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(saved),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (48, 32)
    assert video["codec_name"] == "h264"
    assert int(video["nb_read_frames"]) == FRAMES
    assert video["r_frame_rate"] == f"{FPS}/1"
    options = _x264_options(saved)
    if compression == 100:
        assert options["qp"] == "0"
    else:
        assert options["crf"] == f"{int((1 - compression / 100) * 51):.1f}"
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    assert bool(audio) == with_audio
    if with_audio:
        assert audio[0]["codec_name"] == "aac"
        assert audio[0]["sample_rate"] == "32000"
        assert audio[0]["channels"] == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_failure_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg video encoding failed"):
        output_utils._save_video_ffmpeg(
            str(tmp_path / "invalid.mp4"),
            [np.zeros((16, 16, 3), dtype=np.uint8)],
            fps=FPS,
            quality=5,
            x264_preset="invalid",
        )
