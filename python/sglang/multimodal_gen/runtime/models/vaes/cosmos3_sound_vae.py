# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sound tokenizer (decode-only AVAE).

The on-disk ``sound_tokenizer`` ships only the decoder (no encoder), so this
wraps the diffusers Oobleck decoder, whose module tree and SnakeBeta activation
(alpha + beta, log-scale) match the checkpoint key-for-key. It maps a
``[B, sound_dim, T]`` audio latent at ``sound_latent_fps`` to a
``[B, audio_channels, T * hop_size]`` waveform at ``sampling_rate``.
"""

import json
import os

import torch
from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
from safetensors.torch import load_file
from torch import nn

_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_CONFIG_NAME = "config.json"
_DECODER_PREFIX = "decoder."


class Cosmos3SoundVAE(nn.Module):
    """Decode-only Oobleck AVAE for Cosmos3 sound generation."""

    def __init__(self, config: dict):
        super().__init__()
        self.sample_rate = config["sampling_rate"]
        self.audio_channels = config["dec_out_channels"]
        self.latent_dim = config["vocoder_input_dim"]
        self.hop_size = config["hop_size"]
        # The decoder upsamples in the reverse of the encoder strides.
        self.decoder = OobleckDecoder(
            channels=config["dec_dim"],
            input_channels=self.latent_dim,
            audio_channels=self.audio_channels,
            upsampling_ratios=list(reversed(config["dec_strides"])),
            channel_multiples=config["dec_c_mults"],
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "Cosmos3SoundVAE":
        with open(os.path.join(path, _CONFIG_NAME)) as f:
            config = json.load(f)
        model = cls(config)
        state_dict = load_file(os.path.join(path, _WEIGHTS_NAME))
        decoder_sd = {
            k[len(_DECODER_PREFIX) :]: v
            for k, v in state_dict.items()
            if k.startswith(_DECODER_PREFIX)
        }
        model.decoder.load_state_dict(decoder_sd, strict=True)
        return model

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, latent_dim, T]`` to ``[B, audio_channels, T * hop_size]``."""
        return self.decoder(latent)
