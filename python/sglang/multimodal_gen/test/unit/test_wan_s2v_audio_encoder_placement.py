# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V's before-denoising stage: audio placement, output rate, remote-input downloads."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    wan_s2v as stage_module,
)

DEVICE = torch.device("meta")  # stands in for the accelerator; never computed on


class _RecordingEncoder:
    def __init__(self, device):
        self.device = torch.device(device)
        self.encoded_on = []

    def to(self, device):
        self.device = torch.device(device)
        return self

    def encode_audio(self, audio_path, *, infer_frames, fps, dtype, m):
        self.encoded_on.append(self.device)
        return torch.zeros(1), 1


def test_audio_encodes_on_the_device_after_being_parked_on_the_host():
    """--audio-encoder-cpu-offload parks the encoder on the host after each request; the next
    request must still encode on the device, not where it was parked."""
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    stage.transformer = SimpleNamespace(
        fps=16, param_dtype=torch.bfloat16, audio_sample_m=0
    )
    stage.audio_encoder = _RecordingEncoder("cpu")  # as request 1's offload leaves it
    with patch.object(stage_module, "get_local_torch_device", return_value=DEVICE):
        stage._encode_audio("talk.wav", infer_frames=81)
        stage.audio_encoder = stage.audio_encoder.to("cpu")  # parked again
        stage._encode_audio("talk.wav", infer_frames=81)
    assert stage.audio_encoder.encoded_on == [DEVICE, DEVICE]


class _StopAfterFps(Exception):
    pass


def _stop(*args, **kwargs):
    raise _StopAfterFps


def test_stage_saves_at_the_checkpoint_rate_not_the_requested_one(tmp_path):
    """A served request defaults fps to 24 while S2V paces audio at the checkpoint's 16;
    saved at 24, the video ends at 3.2 s of a 5 s clip and freezes on its last frame."""
    image, audio = tmp_path / "ref.png", tmp_path / "talk.wav"
    image.touch()
    audio.touch()
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    # Stops forward right after the fps is resolved, before any model work.
    stage.transformer = SimpleNamespace(
        supports_standard_denoising=True, fps=16, _normalize_infer_frames=_stop
    )
    batch = SimpleNamespace(
        image_path=str(image),
        audio_path=str(audio),
        pose_video_path=None,
        num_clip=None,
        num_frames=81,
        fps=24,
    )
    with (
        patch.object(stage, "_generate_seed_and_generator"),
        pytest.raises(_StopAfterFps),
    ):
        stage.forward(batch, server_args=SimpleNamespace())
    assert batch.fps == 16


class _Response:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_retry_after_failed_download_gets_the_whole_file(tmp_path):
    """The download cache returns any file already at its path, so a download that times out
    mid-body must not leave one there for the retry to pick up."""
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    responses = iter(
        [_Response(TimeoutError("read timed out")), _Response(b"RIFF-complete")]
    )
    with (
        patch.object(stage_module.tempfile, "gettempdir", return_value=str(tmp_path)),
        patch.object(stage_module, "urlopen", lambda *a, **kw: next(responses)),
    ):
        with pytest.raises(TimeoutError):
            stage._download_remote_input("https://example.com/talk.wav", "audio_path")
        path = stage._download_remote_input(
            "https://example.com/talk.wav", "audio_path"
        )
    with open(path, "rb") as f:
        assert f.read() == b"RIFF-complete"
