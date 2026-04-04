import math
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    LTX2VideoCausalConv3d,
    LTX2VideoResnetBlock3d,
    LTXVideoDownsampler3d,
)


def _patchify_video(sample: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return sample
    batch_size, channels, num_frames, height, width = sample.shape
    sample = sample.reshape(
        batch_size,
        channels,
        num_frames,
        1,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return sample.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)


class LTX23VideoPixelNorm(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


class LTX23PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x - mean) / std


class LTX23VideoResBlockStack(nn.Module):
    def __init__(self, channels: int, num_layers: int, spatial_padding_mode: str) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                LTX2VideoResnetBlock3d(
                    in_channels=channels,
                    out_channels=channels,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states, causal=True)
        return hidden_states


def _make_ltx23_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    spatial_padding_mode: str,
) -> tuple[nn.Module, int]:
    if block_name == "res_x":
        return (
            LTX23VideoResBlockStack(
                channels=in_channels,
                num_layers=int(block_config["num_layers"]),
                spatial_padding_mode=spatial_padding_mode,
            ),
            in_channels,
        )

    multiplier = int(block_config.get("multiplier", 2))
    stride_map = {
        "compress_space_res": (1, 2, 2),
        "compress_time_res": (2, 1, 1),
        "compress_all_res": (2, 2, 2),
    }
    stride = stride_map.get(block_name)
    if stride is None:
        raise ValueError(f"Unsupported LTX-2.3 encoder block: {block_name}")
    out_channels = in_channels * multiplier
    return (
        LTXVideoDownsampler3d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            spatial_padding_mode=spatial_padding_mode,
        ),
        out_channels,
    )


class LTX23VideoConditionEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        vae_config = config.get("vae", config)
        latent_channels = int(vae_config["latent_channels"])
        patch_size = int(vae_config.get("patch_size", 4))
        spatial_padding_mode = str(vae_config.get("spatial_padding_mode", "zeros"))
        encoder_blocks = list(vae_config["encoder_blocks"])
        latent_log_var = str(vae_config.get("latent_log_var", "uniform"))

        self.patch_size = patch_size
        self.latency_channels = latent_channels
        self.latent_log_var = latent_log_var
        self.per_channel_statistics = LTX23PerChannelStatistics(latent_channels)

        feature_channels = latent_channels
        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=int(vae_config.get("in_channels", 3)) * patch_size**2,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList()
        for block_name, block_params in encoder_blocks:
            block_config = (
                {"num_layers": block_params}
                if isinstance(block_params, int)
                else dict(block_params)
            )
            block, feature_channels = _make_ltx23_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                spatial_padding_mode=spatial_padding_mode,
            )
            self.down_blocks.append(block)

        self.conv_norm_out = LTX23VideoPixelNorm(dim=1, eps=1e-8)
        self.conv_act = nn.SiLU()

        conv_out_channels = latent_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var in {"uniform", "constant"}:
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Unsupported latent_log_var: {latent_log_var}")

        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        frames_count = int(sample.shape[2])
        if (frames_count - 1) % 8 != 0:
            frames_to_crop = (frames_count - 1) % 8
            sample = sample[:, :, :-frames_to_crop, ...]

        hidden_states = _patchify_video(sample, self.patch_size)
        hidden_states = self.conv_in(hidden_states, causal=True)

        for block in self.down_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=True)

        if self.latent_log_var == "uniform":
            means = hidden_states[:, :-1, ...]
            logvar = hidden_states[:, -1:, ...]
            hidden_states = torch.cat(
                [
                    means,
                    logvar.repeat(1, means.shape[1], *([1] * (means.ndim - 2))),
                ],
                dim=1,
            )
        elif self.latent_log_var == "constant":
            means = hidden_states[:, :-1, ...]
            logvar = torch.full_like(means, -30.0)
            hidden_states = torch.cat([means, logvar], dim=1)

        means, _ = torch.chunk(hidden_states, 2, dim=1)
        return self.per_channel_statistics.normalize(means)
