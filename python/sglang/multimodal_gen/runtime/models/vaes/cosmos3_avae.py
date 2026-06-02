# SPDX-License-Identifier: Apache-2.0
"""Decoder-only audio tokenizer for the Cosmos3 sound modality."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import weight_norm

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Snake1d(nn.Module):
    def __init__(self, hidden_dim: int, logscale: bool = True) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)


class OobleckResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = weight_norm(
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        )
        self.snake2 = Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.snake1(x))
        y = self.conv2(self.snake2(y))
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class OobleckDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        output_padding: int,
    ) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ) -> None:
        super().__init__()
        strides = upsampling_ratios
        mults = [1] + list(channel_multiples)

        self.conv1 = weight_norm(
            nn.Conv1d(input_channels, channels * mults[-1], kernel_size=7, padding=3)
        )

        blocks = []
        for i, stride in enumerate(strides):
            blocks.append(
                OobleckDecoderBlock(
                    input_dim=channels * mults[len(strides) - i],
                    output_dim=channels * mults[len(strides) - i - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            )
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(
            nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.snake1(x)
        return self.conv2(x)


def _cfg(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


class Cosmos3AVAEAudioTokenizer(nn.Module):
    """Cosmos3 audio tokenizer: latents → waveform via an Oobleck decoder stack."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.sample_rate = int(_cfg(config, "sampling_rate", "sample_rate", default=48000))
        self.audio_channels = int(
            _cfg(
                config,
                "dec_out_channels",
                "audio_channels",
                default=2 if bool(config.get("stereo", True)) else 1,
            )
        )
        self.latent_channels = int(
            _cfg(config, "vocoder_input_dim", "io_channels", "latent_ch", default=64)
        )
        dec_strides = [int(s) for s in _cfg(config, "dec_strides", default=[2, 4, 5, 6, 8])]
        self.hop_size = int(
            _cfg(config, "hop_size", default=math.prod(dec_strides) if dec_strides else 1920)
        )
        stride_product = math.prod(dec_strides)
        if stride_product != self.hop_size:
            raise ValueError(
                "Cosmos3 AVAE dec_strides product must equal hop_size: "
                f"product={stride_product}, hop_size={self.hop_size}."
            )

        norm = str(_cfg(config, "normalization_type", default="none"))
        if bool(_cfg(config, "normalize_latents", default=False)) and norm == "none":
            norm = "tanh"
        self.normalization_type = norm
        self.tanh_input_scale = float(_cfg(config, "tanh_input_scale", default=1.5))
        self.tanh_output_scale = float(_cfg(config, "tanh_output_scale", default=3.5))
        self.tanh_clamp = float(_cfg(config, "tanh_clamp", default=0.995))

        self.decoder = OobleckDecoder(
            channels=int(_cfg(config, "dec_dim", default=320)),
            input_channels=self.latent_channels,
            audio_channels=self.audio_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(_cfg(config, "dec_c_mults", default=[1, 2, 4, 8, 16])),
        )

    @property
    def temporal_compression_factor(self) -> int:
        return self.hop_size

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.hop_size

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.hop_size

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            x = torch.clamp(
                latent.float() / self.tanh_output_scale,
                -self.tanh_clamp,
                self.tanh_clamp,
            )
            return (torch.atanh(x) * self.tanh_input_scale).to(in_dtype)
        if self.normalization_type != "none":
            raise ValueError(
                f"Unsupported AVAE normalization_type={self.normalization_type!r}."
            )
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        squeeze = latent.ndim == 2
        if squeeze:
            latent = latent.unsqueeze(0)
        decoder_dtype = next(self.decoder.parameters()).dtype
        decoder_device = next(self.decoder.parameters()).device
        z = self._denormalize_latent(latent.to(decoder_device)).to(decoder_dtype)
        audio = self.decoder(z).clamp(-1.0, 1.0).to(latent.dtype)
        return audio.squeeze(0) if squeeze else audio

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> "Cosmos3AVAEAudioTokenizer":
        path = Path(path)
        with open(path / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config)
        state_dict = load_file(
            str(path / "diffusion_pytorch_model.safetensors"), device="cpu"
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("encoder.")]
        if missing:
            logger.warning("Cosmos3 AVAE missing keys: %s", missing[:8])
        if unexpected:
            logger.warning("Cosmos3 AVAE unexpected keys: %s", unexpected[:8])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device=device, dtype=dtype)
        return model


EntryClass = Cosmos3AVAEAudioTokenizer
