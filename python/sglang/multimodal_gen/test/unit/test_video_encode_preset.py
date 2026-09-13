# SPDX-License-Identifier: Apache-2.0
"""The configured x264 preset has to reach the encoder, not just the command.

libx264 records the options it resolved into the mp4 it writes, so a real encode
can be checked against a reference encode made with the preset spelled out. That
catches the failure this guards against -- the preset never being passed, and
ffmpeg silently applying its own default.
"""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import X264_PRESET, save_outputs

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
