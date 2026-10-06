# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 audio VAE + vocoder configuration.

Mirrors the official checkpoints' ``audio_vae/config.json``
(``kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers`` and
``kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers``, which differ only in
``scaling_factor``: 0.5302 and 0.417)::

    {
      "_class_name": "MMAudioVAE", "mode": "44k", "sample_rate": 44100,
      "downsample_factor": 1024, "scaling_factor": 0.5302,
      "vocoder_config": {...BigVGAN-v2 hyperparams...}
    }

The runtime ``Kandinsky6AudioVAE`` module (``runtime/models/vaes``, resolved
from the ``MMAudioVAE`` class name) bundles two independently-trained
components in one checkpoint: an MMAudioVAE (mel<->latent codec) and a BigVGANV2
vocoder (mel->waveform). This config is a thin passthrough for both:
``vocoder_config`` is forwarded to the runtime ``BigVGANV2(...)`` constructor
unmodified, ``mode`` selects the nested MMAudioVAE variant and
``scaling_factor`` denormalizes the audio latents before decoding.
"""

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
