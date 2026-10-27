"""Torch-native waveform encoder used by Muse 1.2 Omni checkpoints."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkAudioConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight, self.eps)


def patchify(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, length, channels = value.shape
    if length % patch_size:
        raise ValueError(f"audio length {length} is not divisible by {patch_size}")
    return value.reshape(batch, length // patch_size, patch_size * channels)


class Patchifier(nn.Module):
    def __init__(
        self, patch_size: int, input_size: int, output_size: int, eps: float
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_size * patch_size, output_size, bias=True)
        self.norm = RMSNorm(output_size, eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(patchify(value, self.patch_size)))


class DepthwiseCausalConv(nn.Conv1d):
    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )
        self.causal_kernel_size = kernel_size

    def forward(
        self, value: torch.Tensor, cache: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        value = value.transpose(1, 2)
        if cache is None:
            value = F.pad(value, (self.causal_kernel_size - 1, 0))
            new_cache = None
        else:
            value = torch.cat((cache, value), dim=-1)
            new_cache = value[:, :, -self.causal_kernel_size + 1 :]
        return super().forward(value).transpose(1, 2), new_cache


class Block(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = DepthwiseCausalConv(hidden_size, kernel_size)
        self.norm = RMSNorm(hidden_size, eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.mlp.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(
        self, value: torch.Tensor, cache: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = value
        value, cache = self.conv(value, cache)
        value = self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm(value))))
        return value + residual, cache


class AudioEncoder(nn.Module):
    """Encode mono 24 kHz waveforms into one token per 1,920 samples."""

    def __init__(self, config: MuseSparkAudioConfig) -> None:
        super().__init__()
        if not (
            len(config.patch_sizes)
            == len(config.hidden_sizes)
            == len(config.depths)
            == config.num_hidden_layers
        ):
            raise ValueError("audio stage settings must have num_hidden_layers entries")
        if config.hidden_act != "gelu":
            raise ValueError("Muse audio encoder supports gelu only")
        self.output_size = config.out_hidden_size
        self.patch_size = math.prod(config.patch_sizes)
        self.input_channels = 1
        stages: list[nn.Module] = []
        stage_input = self.input_channels
        for patch, hidden, layer_count in zip(
            config.patch_sizes, config.hidden_sizes, config.depths
        ):
            stage = nn.Module()
            stage.patch_embedder = Patchifier(
                patch, stage_input, hidden, config.rms_norm_eps
            )
            stage.conv_blocks = nn.ModuleList(
                Block(
                    hidden,
                    int(hidden * config.mlp_ratio),
                    config.conv_kernel_size,
                    config.rms_norm_eps,
                )
                for _ in range(layer_count)
            )
            stages.append(stage)
            stage_input = hidden
        self.layers = nn.ModuleList(stages)
        self.norm = RMSNorm(stage_input, config.rms_norm_eps)
        self.proj = nn.Linear(stage_input, self.output_size, bias=True)
        self.post_proj_norm = RMSNorm(self.output_size, config.rms_norm_eps)

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        for stage in self.layers:
            value = stage.patch_embedder(value)
            for block in stage.conv_blocks:
                value, _ = block(value)
        return self.post_proj_norm(self.proj(self.norm(value)))

    def forward(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for waveform in waveforms:
            waveform = waveform.reshape(-1, self.input_channels)
            padding = (-waveform.shape[0]) % self.patch_size
            waveform = F.pad(waveform, (0, 0, 0, padding)).unsqueeze(0)
            outputs.append(self._encode(waveform).squeeze(0))
        return outputs


__all__ = ["AudioEncoder"]
