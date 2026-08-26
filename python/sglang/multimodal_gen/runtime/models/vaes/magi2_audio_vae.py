# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.magi2 import Magi2AudioVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_avae import OobleckDecoder

# Module order matches sglang's OobleckDecoder, so this is an index -> name relabel.
_CHECKPOINT_PREFIX = "pretransform.model."
_DECODER_BLOCK_NAMES = ("snake1", "conv_t1", "res_unit1", "res_unit2", "res_unit3")
_RESIDUAL_UNIT_NAMES = ("snake1", "conv1", "snake2", "conv2")


def _remap_decoder_key(key: str, *, num_blocks: int) -> str:
    head, index, tail = key.split(".", 2)
    assert head == "layers", f"unexpected oobleck decoder key {key}"
    index = int(index)
    if index == 0:
        return f"conv1.{tail}"
    if index == num_blocks + 1:
        return f"snake1.{tail}"
    if index == num_blocks + 2:
        return f"conv2.{tail}"

    inner_head, inner_index, inner_tail = tail.split(".", 2)
    assert inner_head == "layers", f"unexpected oobleck decoder key {key}"
    name = _DECODER_BLOCK_NAMES[int(inner_index)]
    if name.startswith("res_unit"):
        unit_head, unit_index, unit_tail = inner_tail.split(".", 2)
        assert unit_head == "layers", f"unexpected oobleck decoder key {key}"
        inner_tail = f"{_RESIDUAL_UNIT_NAMES[int(unit_index)]}.{unit_tail}"
    return f"block.{index - 1}.{name}.{inner_tail}"


def remap_stable_audio_decoder_state_dict(
    state_dict: dict[str, torch.Tensor], *, num_blocks: int
) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        name = key.removeprefix(_CHECKPOINT_PREFIX)
        if not name.startswith("decoder.layers."):
            continue
        target = "decoder." + _remap_decoder_key(
            name.removeprefix("decoder."), num_blocks=num_blocks
        )
        # SnakeBeta keeps alpha/beta as (C,); sglang's Snake1d as (1, C, 1).
        if target.endswith(("alpha", "beta")):
            tensor = tensor.reshape(1, -1, 1)
        remapped[target] = tensor
    return remapped


def resample_fft(x: torch.Tensor, *, orig_freq: int, new_freq: int) -> torch.Tensor:
    """FFT-domain, matching the reference's scipy.signal.resample (audio_decoder.py:31-33), not torchaudio's windowed sinc."""
    n_x = x.shape[-1]
    num = n_x * new_freq // orig_freq
    if num == n_x:
        return x

    spectrum = torch.fft.rfft(x, dim=-1)
    m = min(num, n_x)
    spectrum = spectrum[..., : m // 2 + 1].clone()
    if m % 2 == 0:
        # Bin m//2 is unpaired: scipy folds its mirror in downsampling, splits it up.
        spectrum[..., m // 2] *= 2.0 if num < n_x else 0.5
    return torch.fft.irfft(spectrum * (num / n_x), n=num, dim=-1)


class Magi2AudioVAE(nn.Module):
    """Latents run at 25 fps, so 2048x upsampling lands at 51200 Hz, resampled to 44100."""

    def __init__(self, config: Magi2AudioVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config

        stride_product = math.prod(arch.strides)
        if stride_product != arch.downsampling_ratio:
            raise ValueError(
                "MAGI-2 audio VAE strides must multiply to downsampling_ratio: "
                f"product={stride_product}, downsampling_ratio={arch.downsampling_ratio}."
            )
        if not arch.use_snake:
            raise ValueError("MAGI-2 audio VAE is only shipped with snake activations.")
        # Odd strides would need ConvTranspose1d output_padding; shipped ones are even.
        if any(stride % 2 for stride in arch.strides):
            raise ValueError(f"Odd oobleck stride is unsupported: {arch.strides}.")

        self.latent_dim = arch.latent_dim
        self.audio_channels = arch.io_channels
        self.downsampling_ratio = arch.downsampling_ratio
        self.native_sample_rate = arch.native_sample_rate
        self.sample_rate = arch.output_sample_rate
        self.final_tanh = arch.final_tanh
        self.num_blocks = len(arch.strides)

        self.decoder = OobleckDecoder(
            channels=arch.channels,
            input_channels=arch.latent_dim,
            audio_channels=arch.io_channels,
            # sglang's decoder walks strides coarse-to-fine.
            upsampling_ratios=list(reversed(arch.strides)),
            channel_multiples=list(arch.c_mults),
        )

    def num_output_samples(self, num_latent_frames: int) -> int:
        native = num_latent_frames * self.downsampling_ratio
        return native * self.sample_rate // self.native_sample_rate

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state_dict = remap_stable_audio_decoder_state_dict(
            dict(weights), num_blocks=self.num_blocks
        )
        self.load_state_dict(state_dict, strict=True)
        return set(state_dict)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Left unclamped so the caller can normalize once at save time."""
        unbatched = latents.ndim == 2
        if unbatched:
            latents = latents.unsqueeze(0)

        param = next(self.decoder.parameters())
        # Bottleneck is pass-through (audio_decoder.py:27-28, :115).
        waveform = self.decoder(latents.to(device=param.device, dtype=param.dtype))
        if self.final_tanh:
            waveform = torch.tanh(waveform)
        waveform = resample_fft(
            waveform.float(),
            orig_freq=self.native_sample_rate,
            new_freq=self.sample_rate,
        )
        return waveform.squeeze(0) if unbatched else waveform


EntryClass = Magi2AudioVAE
