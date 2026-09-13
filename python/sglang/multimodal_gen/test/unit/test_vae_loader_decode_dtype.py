"""The loader rounds VAE decoder weights to the decode dtype it will compute in.

The decode stage persists these frozen weights in the autocast dtype on first
use, so the rounding is already part of every output; what the tests pin down
is when the loader is allowed to do it early and when it must leave the
checkpoint dtype alone.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _hold_decoder_weights_in_decode_dtype,
)


class _RecordingVAE:
    def __init__(self):
        self.prepared_with = None

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        self.prepared_with = dtype
        return 144


def _server_args(decode_precision="fp16", disable_autocast=False):
    return SimpleNamespace(
        component_precisions={},
        pipeline_config=SimpleNamespace(
            vae_decode_precision=decode_precision,
            vae_precision="fp32",
        ),
        disable_autocast=disable_autocast,
    )


@pytest.fixture(autouse=True)
def _amp_supported(monkeypatch):
    from sglang.multimodal_gen.runtime.utils import precision

    monkeypatch.setattr(precision.current_platform, "is_amp_supported", lambda: True)


def test_the_decoder_is_rounded_to_the_decode_dtype_at_load():
    vae = _RecordingVAE()
    _hold_decoder_weights_in_decode_dtype(vae, _server_args(), "video_vae")
    assert vae.prepared_with == torch.float16


def test_disabling_autocast_keeps_the_checkpoint_dtype():
    vae = _RecordingVAE()
    _hold_decoder_weights_in_decode_dtype(
        vae, _server_args(disable_autocast=True), "video_vae"
    )
    assert vae.prepared_with is None


def test_the_kill_switch_keeps_the_checkpoint_dtype(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST", "1")
    vae = _RecordingVAE()
    _hold_decoder_weights_in_decode_dtype(vae, _server_args(), "video_vae")
    assert vae.prepared_with is None


def test_an_fp32_decode_precision_is_left_alone():
    vae = _RecordingVAE()
    _hold_decoder_weights_in_decode_dtype(
        vae, _server_args(decode_precision="fp32"), "video_vae"
    )
    assert vae.prepared_with is None


def test_the_audio_vae_is_not_touched():
    vae = _RecordingVAE()
    _hold_decoder_weights_in_decode_dtype(vae, _server_args(), "audio_vae")
    assert vae.prepared_with is None


def test_a_vae_without_the_hook_is_skipped():
    _hold_decoder_weights_in_decode_dtype(object(), _server_args(), "video_vae")
