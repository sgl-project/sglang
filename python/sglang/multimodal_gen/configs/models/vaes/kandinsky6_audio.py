# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration for the bundled MMAudio codec and BigVGAN vocoder."""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Kandinsky6AudioVAEArchConfig(VAEArchConfig):
    mode: str = "44k"
    # The two flags below are not read by the runtime module: the official
    # checkpoints carry the encoder half's weights, so ``Kandinsky6AudioVAE``
    # always builds both halves of the nested MMAudioVAE.
    need_vae_decoder: bool = True
    need_vae_encoder: bool = False
    scaling_factor: float = 0.5302
    # Reverses a checkpoint's own audio-latent shift, matching the diffusers
    # reference's postprocess_audio (getattr(audio_vae, "mean_value", 0.0));
    # no real checkpoint has been observed setting this to anything but 0.0
    # yet.
    mean_value: float = 0.0
    # BigVGAN-v2 hyperparameters, passed straight through to the runtime
    # BigVGANV2(...) constructor unmodified.
    vocoder_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Kandinsky6AudioVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=Kandinsky6AudioVAEArchConfig)
    # Generation only decodes audio; the runtime module does not read these.
    load_encoder: bool = False
    load_decoder: bool = True


__all__ = ["Kandinsky6AudioVAEArchConfig", "Kandinsky6AudioVAEConfig"]
