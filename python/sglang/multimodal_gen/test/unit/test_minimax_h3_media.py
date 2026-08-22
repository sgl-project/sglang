# SPDX-License-Identifier: Apache-2.0
"""Numerical boundaries for the one-pass Ref2VA media path."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    reference_encoding,
)


def test_ffprobe_falls_back_when_stream_side_data_is_unknown(monkeypatch):
    material_io._ffprobe_entries = None
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        entries = command[command.index("-show_entries") + 1]
        if "stream_side_data" in entries:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="ffprobe: No match for section 'stream_side_data'",
            )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "audio", "sample_rate": "44100"}],
                    "format": {"format_name": "mp3", "duration": "1.0"},
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    payload = material_io._ffprobe_media("/input/ref.mp3")

    assert len(calls) == 2
    assert "stream_side_data" in calls[0][calls[0].index("-show_entries") + 1]
    assert "stream_side_data" not in calls[1][calls[1].index("-show_entries") + 1]
    assert payload["format"]["format_name"] == "mp3"
    assert material_io._ffprobe_entries is not None
    assert "stream_side_data" not in material_io._ffprobe_entries


def test_video_transform_runs_once_and_qwen_samples_shared_rgb(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[-1] == "pipe:1":
            return SimpleNamespace(stdout=expected.tobytes())
        output_fd = int(command[-1].removeprefix("pipe:"))
        assert kwargs["pass_fds"] == (output_fd,)
        os.write(output_fd, expected.tobytes())
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    frames = reference_encoding.minimax_h3_decode_reference_video_frames(
        "/input/ref.mp4",
        target_width=6,
        target_height=4,
        target_frame_count=25,
        fps=24.0,
        start_time_seconds=2.25,
    )
    sampled = reference_encoding.minimax_h3_sample_reference_video_frames(frames)

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-vf") + 1] == (
        "fps=24,scale=6:4:flags=lanczos,setsar=1"
    )
    assert command[command.index("-frames:v") + 1] == "25"
    assert command[command.index("-ss") + 1] == "2.25"
    assert command.index("-ss") < command.index("-i")
    assert command[-5:-1] == ["-f", "rawvideo", "-pix_fmt", "rgb24"]
    assert command[-1].startswith("pipe:")
    assert "libx264" not in command
    if command[-1] != "pipe:1":
        assert frames.flags.writeable
    assert all(np.shares_memory(frame, frames) for frame in sampled["frames"])
    assert [int(frame[0, 0, 0]) for frame in sampled["frames"]] == [
        int(expected[index, 0, 0, 0]) for index in (0, 12, 24)
    ]
    assert sampled["block_timestamps"] == [0.25, 1.0]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_video_transform_can_share_one_host_decode(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    class FakeGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = object()

        def barrier(self):
            return None

    def all_gather_object(outputs, value, **_kwargs):
        outputs[:] = [value, value]

    def run(command, **_kwargs):
        commands.append(command)
        output_fd = int(command[-1].removeprefix("pipe:"))
        os.write(output_fd, expected.tobytes())
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(subprocess, "run", run)
    reference_encoding._reference_video_host_leader.cache_clear()

    try:
        frames = reference_encoding.minimax_h3_decode_reference_video_frames(
            "/input/ref.mp4",
            target_width=6,
            target_height=4,
            target_frame_count=25,
            share_across_replicas=True,
        )
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    assert np.array_equal(frames, expected)
    assert frames.flags.writeable
    assert len(commands) == 1
    assert commands[0][-1].startswith("pipe:")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_shared_video_transform_falls_back_when_proc_fd_is_blocked(monkeypatch):
    expected = np.arange(25 * 4 * 6 * 3, dtype=np.uint8).reshape(25, 4, 6, 3)
    commands = []

    class FakeGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = object()

    def all_gather_object(outputs, value, **_kwargs):
        outputs[:] = [value, value]

    def run(command, **_kwargs):
        commands.append(command)
        os.write(int(command[-1].removeprefix("pipe:")), expected.tobytes())
        return SimpleNamespace(stderr=b"")

    real_open = os.open

    def guarded_open(path, flags):
        if str(path).startswith("/proc/"):
            raise PermissionError("blocked by test policy")
        return real_open(path, flags)

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "open", guarded_open)
    reference_encoding._reference_video_host_leader.cache_clear()
    try:
        frames = reference_encoding.minimax_h3_decode_reference_video_frames(
            "/input/ref.mp4",
            target_width=6,
            target_height=4,
            target_frame_count=25,
            share_across_replicas=True,
        )
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    assert np.array_equal(frames, expected)
    assert len(commands) == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux memfd")
def test_shared_video_transform_propagates_any_host_decode_failure(monkeypatch):
    class FakeGroup:
        world_size = 4
        rank_in_group = 0
        cpu_group = object()

    gather_index = 0

    def all_gather_object(outputs, value, **_kwargs):
        nonlocal gather_index
        if gather_index == 0:
            outputs[:] = ["host-a", "host-a", "host-b", "host-b"]
        else:
            outputs[:] = [
                value,
                None,
                (None, 0, "CalledProcessError: remote decode failed"),
                None,
            ]
        gather_index += 1

    monkeypatch.setattr(reference_encoding, "get_world_group", FakeGroup)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        reference_encoding,
        "_write_reference_video_to_fd",
        lambda _command, _fd: 1,
    )
    reference_encoding._reference_video_host_leader.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="remote decode failed"):
            reference_encoding._decode_reference_video_shared(["ffmpeg"])
    finally:
        reference_encoding._reference_video_host_leader.cache_clear()

    # The failure is resolved immediately after the shared state exchange;
    # no rank enters a mapping collective that another host skipped.
    assert gather_index == 2


def test_audio_decode_is_bounded_float_pcm_without_temp_files(monkeypatch):
    pcm = torch.arange(8, dtype=torch.float32).numpy().tobytes()
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(
                stdout=json.dumps(
                    {"streams": [{"channels": 6, "sample_rate": "44100"}]}
                )
            )
        return SimpleNamespace(stdout=pcm)

    monkeypatch.setattr(subprocess, "run", run)
    waveform, source_rate = reference_encoding._load_waveform(
        "/input/ref.mp4",
        material_chain="video.reference_preserve",
        max_duration_seconds=3.5,
        start_time_seconds=2.25,
    )

    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert source_rate == 44100
    torch.testing.assert_close(
        waveform,
        torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.float32),
    )
    assert ffmpeg[ffmpeg.index("-t") + 1] == "3.5"
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "2.25"
    assert ffmpeg.index("-ss") < ffmpeg.index("-i")
    assert ffmpeg[-3:] == ["-f", "f32le", "pipe:1"]


def test_reference_audio_encode_sets_forward_context(monkeypatch):
    class FakeAudioVAE(torch.nn.Module):
        attn_proj = True

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.mean_proj = torch.nn.Identity()

        def preprocess(self, waveform, _sample_rate):
            return waveform

        def encoder(self, _audio_data):
            return torch.ones(2, 32, 4)

        def pre_block(self, hidden_states):
            assert get_forward_context().current_timestep == 0
            return hidden_states

    monkeypatch.setattr(
        reference_encoding,
        "_load_waveform",
        lambda *_args, **_kwargs: (torch.ones(2, 320), 32000),
    )

    result = reference_encoding.minimax_h3_encode_reference_audio_rows(
        FakeAudioVAE(),
        "/input/ref.wav",
        SimpleNamespace(
            latent_channels=32,
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        ),
    )

    assert result["rows"].shape == (8, 32)
    assert result["ref_audio_t"] == 4
