# SPDX-License-Identifier: Apache-2.0
"""Where the MiniMax-H3 decoding stage decodes and shares the audio (CPU)."""

from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    decoding,
)


@pytest.fixture
def decoder_setup(monkeypatch):
    frames = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2)
    events = []

    def decode_video(latents, **kwargs):
        events.append("video")
        if "on_frames" in kwargs:
            kwargs["on_frames"](frames)
        return frames.clone()

    video_vae = SimpleNamespace(
        training=False,
        decode_base=decode_video,
        processor=SimpleNamespace(revert_tensor=lambda x, **kwargs: x),
    )
    stage = decoding.MiniMaxH3DecodingStage.__new__(decoding.MiniMaxH3DecodingStage)
    stage.video_vae = video_vae
    stage.audio_vae = SimpleNamespace(sample_rate=32000)
    monkeypatch.setattr(
        stage, "use_declared_component", lambda **kwargs: nullcontext(video_vae)
    )
    monkeypatch.setattr(
        stage,
        "_get_vae_decode_fn",
        lambda _module, _args, **kwargs: kwargs["decode_fn"],
    )
    monkeypatch.setattr(
        decoding, "resolve_decode_precision", lambda *args: torch.float32
    )
    monkeypatch.setattr(decoding, "autocast_enabled_for_device", lambda *args: False)
    monkeypatch.setattr(
        decoding, "autocast_context", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(decoding, "use_vae_fast_path", lambda *args: nullcontext())
    monkeypatch.setattr(stage, "_audio_tile_plan", Mock(return_value=(None, 0, 1)))
    args = SimpleNamespace(
        disable_autocast=True,
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    latents_mean=[0, 0, 0], latents_std=[1, 1, 1]
                )
            ),
            trim_audio_to_video=False,
            post_decoding=lambda x, _args: x,
        ),
    )
    batch = SimpleNamespace(
        latents=torch.zeros_like(frames),
        audio_latents=torch.randn(2, 3, 4),
        extra={},
        sampling_params=SimpleNamespace(quality="lossless"),
        trajectory_timesteps=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
        metrics=None,
    )
    return stage, batch, args, frames, events


def test_audio_decoded_first_is_shared_after_the_video(monkeypatch, decoder_setup):
    stage, batch, args, frames, events = decoder_setup
    waveform = torch.randn(2, 1, 16)
    payload = {"waveform": waveform, "sample_rate": 32000}
    monkeypatch.setattr(stage, "_overlaps_audio_decode", lambda _args: False)
    monkeypatch.setattr(
        stage,
        "_decode_owner_audio",
        lambda *args: events.append("audio") or (payload, None),
    )
    monkeypatch.setattr(
        stage,
        "_share_audio",
        lambda *args: events.append("share") or (waveform, 32000),
    )
    monkeypatch.setattr(decoding, "streamed_video_output_path", lambda *args: None)
    output = stage.forward(batch, args)
    assert events == ["audio", "video", "share"]
    stage._audio_tile_plan.assert_called_once_with(False)
    assert torch.equal(output.output, frames)
    assert torch.equal(output.audio, waveform.permute(1, 0, 2))
    assert output.audio_sample_rate == 32000


@pytest.mark.parametrize("trim_audio", [False, True])
def test_overlapped_audio_feeds_the_encoder_and_is_shared_before_it_closes(
    monkeypatch, decoder_setup, trim_audio
):
    stage, batch, args, frames, events = decoder_setup
    batch.num_frames, batch.fps = 1, 10
    args.pipeline_config.trim_audio_to_video = trim_audio
    waveform = torch.randn(2, 1, 3216)
    output_waveform = waveform.permute(1, 0, 2)
    if trim_audio:
        output_waveform = output_waveform[..., :3200]
    payload = {"waveform": waveform, "sample_rate": 32000}
    job = Future()

    def start_decode(_latent, _args, finish):
        decoded = finish(waveform, payload["sample_rate"])
        job.set_result((payload, decoded, decoded[0]))
        return job

    start = Mock(side_effect=start_decode)
    monkeypatch.setattr(stage, "_overlaps_audio_decode", lambda _args: True)
    monkeypatch.setattr(stage, "_start_owner_audio", start)
    monkeypatch.setattr(
        stage,
        "_join_owner_audio",
        lambda started: events.append("join") or (payload, started.result()[1], None),
    )
    monkeypatch.setattr(
        stage,
        "_share_audio",
        lambda *args: events.append("share") or (waveform, 32000),
    )
    monkeypatch.setattr(decoding, "streamed_video_output_path", lambda *args: "a.mp4")

    class Stream:
        def __init__(self, path, request, finish_frames, audio, sample_rate):
            # the encoder resolves the audio on its own thread
            assert callable(audio) and sample_rate == 32000
            assert torch.equal(audio(), output_waveform[0])

        def __call__(self, part):
            pass

        def finish(self):
            events.append("finish")
            return ["a.mp4"]

        def abort(self):
            pytest.fail("unexpected stream abort")

    monkeypatch.setattr(decoding, "EncodeWhileDecoding", Stream)
    output = stage.forward(batch, args)
    start.assert_called_once()
    stage._audio_tile_plan.assert_called_once_with(True)
    assert events == ["video", "join", "share", "finish"]
    assert output.output_file_paths == ["a.mp4"]
    assert torch.equal(output.audio, output_waveform)
    assert output.audio_sample_rate == 32000
