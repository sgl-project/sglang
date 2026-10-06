# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 mel VAE and waveform vocoder.

The vae.*, vocoder.* and mel_converter.* names match the official checkpoint,
allowing the generic VAELoader to load the full tree without key remapping."""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.bigvgan import (
    BigVGANV2,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.mel_converter import (
    MelConverter,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio_vae.mmaudio_vae import (
    MMAudioVAE,
)


class Kandinsky6AudioVAE(nn.Module, LayerwiseOffloadableModuleMixin):
    """MMAudio mel codec and BigVGAN vocoder with checkpoint-compatible nesting.

    decode: [B, embed_dim, A] -> [B, num_mels, T]
    vocode: [B, num_mels, T] -> [B, 1, samples]
    wrapped_decode and wrapped_encode compose the codec and waveform stages."""

    # official checkpoint class name
    _aliases = ["MMAudioVAE"]

    # Decode-only small model relative to the DiT; no benefit from DiT-group
    # offload tuning knobs. Mirrors MiniMaxH3AudioVAE's choice.
    layerwise_offload_dit_group_enabled = False

    def __init__(self, config: Kandinsky6AudioVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.scaling_factor = float(arch.scaling_factor)
        self.mean_value = float(arch.mean_value)
        # audio latent stride; frame-count alignment is owned by the pipeline config
        self.downsample_factor = 1024

        self.mel_converter = MelConverter(
            sampling_rate=44100,
            n_fft=2048,
            num_mels=128,
            hop_size=512,
            win_size=2048,
            fmin=0,
            fmax=22050,
        )
        # checkpoint weights include the encoder regardless of need_vae_encoder
        self.vae = MMAudioVAE(mode=arch.mode, need_encoder=True)
        # normalize random initialization before loading; checkpoint weights are
        # already normalized and must never be renormalized
        self.vae.remove_weight_norm()
        self.vocoder = BigVGANV2(dict(arch.vocoder_config))

        # resolution containers are not called; hooks belong on their blocks
        self.layer_names = (
            [
                f"{path}.{index}.{group}"
                for path, levels in (
                    ("vae.encoder.down", self.vae.encoder.down),
                    ("vae.decoder.up", self.vae.decoder.up),
                )
                for index in range(len(levels))
                for group in ("block", "attn")
            ]
            + [f"vocoder.ups.{index}" for index in range(len(self.vocoder.ups))]
            + ["vocoder.resblocks"]
        )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: [B, embed_dim, A] (1D-conv channel-first) -> mel: [B, num_mels, T]."""
        return self.vae.decode(latents, unnormalize_output=True)

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, num_mels, T] -> waveform: [B, 1, samples]."""
        # mel denormalization returns fp32; cast to the loaded vocoder's dtype
        vocoder_dtype = next(self.vocoder.parameters()).dtype
        return self.vocoder(mel.to(dtype=vocoder_dtype))

    def wrapped_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents -> waveform in one call, matching the diffusers
        reference's MMAudioVAE.wrapped_decode."""
        return self.vocode(self.decode(latents))

    def encode_audio(self, waveform: torch.Tensor):
        """waveform -> mel -> VAE posterior. See MelConverter's module
        docstring: not exercised by / verified against the current
        (decode-only) inference path."""
        mel = self.mel_converter(waveform)
        return self.vae.encode(mel)

    def wrapped_encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform -> mean audio latent in one call, matching the
        diffusers reference's MMAudioVAE.wrapped_encode."""
        return self.encode_audio(waveform).mean

    def remove_weight_norm(self) -> Kandinsky6AudioVAE:
        """Idempotent MMAudio weight normalization, normally applied before loading."""
        self.vae.remove_weight_norm()
        return self


EntryClass = Kandinsky6AudioVAE

__all__ = ["Kandinsky6AudioVAE"]
