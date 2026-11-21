# SPDX-License-Identifier: Apache-2.0
"""LTX-2 ``num_frames`` must be ``8k + 1`` so audio matches the decoded video.

The video VAE only decodes ``(latent_frames - 1) * 8 + 1`` frames, while the
audio length comes from ``num_frames / fps``. CPU only -- no weights, no GPU.
"""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    LTX23PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import (
    LTX25PipelineConfig,
)

CONFIGS = [
    LTX2PipelineConfig,
    LTX23PipelineConfig,
    LTX25PipelineConfig,
    JoyEchoPipelineConfig,
]


def _decoded_video_frames(config, num_frames):
    ratio = config.vae_temporal_compression
    latent_frames = (num_frames - 1) // ratio + 1
    return (latent_frames - 1) * ratio + 1


def _audio_latent_length(config, num_frames, fps):
    batch = SimpleNamespace(fps=fps)
    return config.prepare_audio_latent_shape(batch, 1, num_frames)[2]


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_temporal_ratio_is_eight(config_cls):
    assert config_cls().vae_temporal_compression == 8


@pytest.mark.parametrize("config_cls", CONFIGS)
@pytest.mark.parametrize("num_frames", [1, 9, 33, 121, 241])
def test_aligned_num_frames_unchanged(config_cls, num_frames):
    assert config_cls().adjust_num_frames(num_frames) == num_frames


@pytest.mark.parametrize("config_cls", CONFIGS)
@pytest.mark.parametrize(
    "num_frames, expected",
    [(2, 1), (8, 1), (10, 9), (122, 121), (128, 121), (130, 129)],
)
def test_misaligned_num_frames_round_down(config_cls, num_frames, expected):
    config = config_cls()
    adjusted = config.adjust_num_frames(num_frames, log_adjustment=False)
    assert adjusted == expected
    # Rounding down must not change what the VAE actually decodes.
    assert adjusted == _decoded_video_frames(config, num_frames)


@pytest.mark.parametrize("config_cls", CONFIGS)
@pytest.mark.parametrize("num_frames", [122, 125, 128])
def test_audio_length_matches_decoded_video(config_cls, num_frames):
    config = config_cls()
    fps = 24
    decoded = _decoded_video_frames(config, num_frames)
    adjusted = config.adjust_num_frames(num_frames, log_adjustment=False)

    # Before the fix the audio was sized from the raw request and ran long.
    assert _audio_latent_length(config, num_frames, fps) > _audio_latent_length(
        config, decoded, fps
    )
    assert _audio_latent_length(config, adjusted, fps) == _audio_latent_length(
        config, decoded, fps
    )


def test_adjustment_logs_only_when_requested(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs import ltx_2

    calls = []
    monkeypatch.setattr(ltx_2.logger, "warning", lambda *a, **k: calls.append(a))
    config = LTX2PipelineConfig()

    config.adjust_num_frames(128, log_adjustment=False)
    assert calls == []
    config.adjust_num_frames(121)
    assert calls == []
    config.adjust_num_frames(128)
    assert len(calls) == 1
