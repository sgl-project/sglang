# Copied and adapted from: https://github.com/meituan-longcat/LongCat-AudioDiT
# SPDX-License-Identifier: Apache-2.0
"""WAV-VAE for LongCat-AudioDiT (1D audio autoencoder)."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTVaeConfig,
)


def randn_like_with_generator(
    tensor: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``torch.randn_like`` that can take a CPU generator (serve-safe)."""
    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(
        tensor.shape,
        dtype=torch.float32,
        generator=generator,
    ).to(device=tensor.device, dtype=tensor.dtype)


def _snake_beta(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class LongCatAudioDiTSnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


def _get_vae_activation(activation: str, channels: int | None = None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    elif activation == "snake":
        return LongCatAudioDiTSnakeBeta(channels)
    elif activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


def _wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _pixel_unshuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    return (
        x.view(b, c, w // factor, factor)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(b, c * factor, w // factor)
    )


def _pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    c = c // factor
    return (
        x.view(b, c, factor, w).permute(0, 1, 3, 2).contiguous().view(b, c, w * factor)
    )


class _DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.group_size = in_channels * factor // out_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pixel_unshuffle_1d(x, self.factor)
        b, c, n = x.shape
        return x.view(b, self.out_channels, self.group_size, n).mean(dim=2)


class _UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return _pixel_shuffle_1d(x, self.factor)


class _VaeResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 7,
        use_snake: bool = False,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class _VaeEncoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = False,
        downsample_shortcut: str = "none",
    ):
        super().__init__()
        layers = []
        for d in [1, 3, 9]:
            layers.append(
                _VaeResidualUnit(in_ch, in_ch, dilation=d, use_snake=use_snake)
            )
        act = "snake" if use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=in_ch))
        layers.append(
            _wn_conv1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.layers = nn.Sequential(*layers)
        self.res = (
            _DownsampleShortcut(in_ch, out_ch, stride)
            if downsample_shortcut == "averaging"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class _VaeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = False,
        upsample_shortcut: str = "none",
    ):
        super().__init__()
        act = "snake" if use_snake else "elu"
        layers = [
            _get_vae_activation(act, channels=in_ch),
            _wn_conv_transpose1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]
        for d in [1, 3, 9]:
            layers.append(
                _VaeResidualUnit(out_ch, out_ch, dilation=d, use_snake=use_snake)
            )
        self.layers = nn.Sequential(*layers)
        self.res = (
            _UpsampleShortcut(in_ch, out_ch, stride)
            if upsample_shortcut == "duplicating"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class LongCatAudioDiTVaeEncoder(nn.Module):
    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels
        layers = [
            _wn_conv1d(config.in_channels, c_mults[0] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1):
            layers.append(
                _VaeEncoderBlock(
                    c_mults[i] * ch,
                    c_mults[i + 1] * ch,
                    config.strides[i],
                    use_snake=config.use_snake,
                    downsample_shortcut=config.downsample_shortcut,
                )
            )
        layers.append(
            _wn_conv1d(
                c_mults[-1] * ch, config.encoder_latent_dim, kernel_size=3, padding=1
            )
        )
        self.layers = nn.Sequential(*layers)

        if config.out_shortcut == "averaging":
            self.shortcut = _DownsampleShortcut(
                c_mults[-1] * ch, config.encoder_latent_dim, 1
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x = self.layers[:-1](x)
        return self.layers[-1](x) + self.shortcut(x)


class LongCatAudioDiTVaeDecoder(nn.Module):
    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels

        if config.in_shortcut == "duplicating":
            self.shortcut = _UpsampleShortcut(config.latent_dim, c_mults[-1] * ch, 1)
        else:
            self.shortcut = None

        layers = [
            _wn_conv1d(config.latent_dim, c_mults[-1] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(
                _VaeDecoderBlock(
                    c_mults[i] * ch,
                    c_mults[i - 1] * ch,
                    config.strides[i - 1],
                    use_snake=config.use_snake,
                    upsample_shortcut=config.upsample_shortcut,
                )
            )
        act = "snake" if config.use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=c_mults[0] * ch))
        layers.append(
            _wn_conv1d(
                c_mults[0] * ch,
                config.in_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            )
        )
        if config.final_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x_short = self.shortcut(x) + self.layers[0](x)
        return self.layers[1:](x_short)


class LongCatAudioDiTVae(nn.Module):
    """WAV-VAE audio autoencoder with VAE bottleneck and scale factor.

    The original checkpoint runs encode/decode in **float16** (``model_half=True``
    in ``AutoencoderPretransform``).  We replicate this behaviour so that the
    outputs are numerically identical to the original codebase.
    """

    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        self.config = config
        self.encoder = LongCatAudioDiTVaeEncoder(config)
        self.decoder = LongCatAudioDiTVaeDecoder(config)
        self.scale = config.scale
        self.downsampling_ratio = config.downsampling_ratio

    def to_half(self):
        """Convert encoder and decoder weights to float16 (matching original behaviour)."""
        self.encoder.half()
        self.decoder.half()
        return self

    def encode(
        self,
        audio: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode audio to latent space.

        Runs encoder **and** VAE bottleneck in float16 when weights are float16,
        matching the original ``AutoencoderPretransform(model_half=True)`` +
        ``AudioAutoencoder.encode`` behaviour where the bottleneck operates on
        the fp16 encoder output before the final ``.float()`` conversion.

        Args:
            audio: ``(batch, 1, num_samples)`` raw waveform.
            generator: Optional CPU generator for the VAE posterior sample.

        Returns:
            Latent tensor ``(batch, latent_dim, num_frames)`` in float32.
        """
        is_half = next(self.encoder.parameters()).dtype == torch.float16
        if is_half:
            audio = audio.half()
        latents = self.encoder(audio)
        # VAE bottleneck runs in the same dtype as encoder output (fp16)
        # to match original: bottleneck.encode(latents) happens before .float()
        mean, scale_param = latents.chunk(2, dim=1)
        stdev = F.softplus(scale_param) + 1e-4
        latents = randn_like_with_generator(mean, generator=generator) * stdev + mean
        # Convert to fp32 after bottleneck, matching original AutoencoderPretransform
        if is_half:
            latents = latents.float()
        return latents / self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to audio waveform.

        Runs decoder in float16 when weights are float16, matching the original
        ``AutoencoderPretransform(model_half=True)`` behaviour.

        Args:
            latents: ``(batch, latent_dim, num_frames)``.

        Returns:
            Waveform tensor ``(batch, 1, num_samples)`` in float32.
        """
        z = latents * self.scale
        is_half = next(self.decoder.parameters()).dtype == torch.float16
        if is_half:
            z = z.half()
        decoded = self.decoder(z)
        if is_half:
            decoded = decoded.float()
        return decoded


# Entry class for model registry
EntryClass = LongCatAudioDiTVae
