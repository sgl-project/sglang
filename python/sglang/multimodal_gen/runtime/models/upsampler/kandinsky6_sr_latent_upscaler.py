# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 SR latent upscaler (LU): a bank of convolutional upsamplers of scaled KVAE latents.

Each bank entry is a cascade of two 2x stages over ``[B, C, T, h, w]`` latents::

    x4: input_proj -> pre_blocks -> upsample_1 -> mid_blocks -> upsample_2 -> post_blocks -> output_proj
    x2: mid_input_proj -> x2_branch (adapter -> finisher -> mid_blocks -> second stage)

Every norm is an RMSNorm FiLM-modulated by the input latent itself (``zq``, nearest-resized to
the feature grid), the 3x3x3 convs pad time by repeating the edge frame, and the 2x upsample is
``Conv1x1(up + Conv(1,3,3)(up))`` with ``up`` the nearest-neighbour 2x resize.  The bank holds one
entry per served scale under ``_models.<index>`` in the order declared by ``scales``.

Module registration order equals the checkpoint's parameter order, so ``state_dict()`` lists keys
in file order.

This single file replaces the vendored ``kandinsky6_sr_lu`` subpackage (pydantic-style
flat/multi_scale config, natten/shifted-window motion-attention training leftovers never present
in the released checkpoint -- 12 files, ~3.2k lines): it mirrors FastVideo's consolidated
``fastvideo/models/upsamplers/kandinsky6_sr.py`` (commit 950a5edb, "bitwise identical to the
previous port on the real weights").

The registered module names match the current Diffusers checkpoint layout directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.multimodal_gen.configs.models.upsamplers.kandinsky6_sr import (
    Kandinsky6SRLatentUpscalerEntryConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ReplicateTimeConv3d(nn.Conv3d):
    """``Conv3d`` with 'same' padding that repeats the edge frame along T and zero-pads H and W.

    ``nn.Conv3d`` has a single padding mode for all axes, so T is padded explicitly before the
    convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        pad = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=(0, pad, pad))
        self.temporal_pad = pad

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(
            x, (0, 0, 0, 0, self.temporal_pad, self.temporal_pad), mode="replicate"
        )
        return super().forward(x)


class RMSNorm(nn.Module):
    """Channel RMS norm of ``[B, C, T, H, W]`` features with a learnable per-channel gain."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Upcast to float32 around the normalize (matches the Diffusers reference's
        # Kandinsky6SRLatentUpscalerRMSNorm): without it, bf16 activations drift from the
        # reference across the many ResidualBlock/X2Branch calls that route through this norm.
        normalized = F.normalize(x.float(), dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma


class ModulatedRMSNorm(nn.Module):
    """``RMSNorm(x) * conv_y(zq) + conv_b(zq)`` with 1x1x1 convs of the conditioning latent ``zq``."""

    def __init__(self, dim: int, zq_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv_y = nn.Conv3d(zq_dim, dim, kernel_size=1)
        self.conv_b = nn.Conv3d(zq_dim, dim, kernel_size=1)

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        if zq.shape[2:] != x.shape[2:]:
            zq = F.interpolate(zq, size=x.shape[2:], mode="nearest")
        return self.norm(x) * self.conv_y(zq) + self.conv_b(zq)


class ResidualBlock(nn.Module):
    """Pre-activation block ``norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2`` plus a (1x1x1 if
    narrowing) skip."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int, zq_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = ModulatedRMSNorm(in_channels, zq_dim)
        self.conv1 = ReplicateTimeConv3d(in_channels, mid_channels, kernel_size=3)
        self.norm2 = ModulatedRMSNorm(mid_channels, zq_dim)
        self.conv2 = ReplicateTimeConv3d(mid_channels, out_channels, kernel_size=3)
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else ReplicateTimeConv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        h = F.silu(self.norm1(x, zq))
        h = self.conv1(h)
        h = F.silu(self.norm2(h, zq))
        h = self.conv2(h)
        return self.shortcut(x) + h


def nearest_2x(x: Tensor) -> Tensor:
    """Nearest-neighbour 2x resize of H and W of ``[B, C, T, H, W]`` (T folded into the batch)."""
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    return x.reshape(b, t, c, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)


class PXSUpsample(nn.Module):
    """2x spatial upsample ``linear(up + spatial_conv(up))`` with ``up = nearest_2x(x)``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.linear = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        up = nearest_2x(x)
        return self.linear(up + self.spatial_conv(up))


class X2Finisher(nn.Module):
    """The ``PXSUpsample`` conv pair applied on the unchanged grid (no resize)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.linear = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x + self.spatial_conv(x))


def _stem(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(ReplicateTimeConv3d(in_channels, out_channels, kernel_size=3))


def _mid_output_head(channels: int, out_channels: int) -> nn.Sequential:
    """Unused training head whose checkpoint layout remains positional."""
    return nn.Sequential(
        RMSNorm(channels),
        nn.SiLU(),
        ReplicateTimeConv3d(channels, out_channels, kernel_size=3),
    )


class Kandinsky6SRLatentUpscalerOutputHead(nn.Module):
    """Named ``norm -> activation -> conv`` projection used by current checkpoints."""

    def __init__(self, channels: int, out_channels: int, zq_dim: int) -> None:
        super().__init__()
        self.norm = ModulatedRMSNorm(channels, zq_dim)
        self.activation = nn.SiLU()
        self.conv = ReplicateTimeConv3d(channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        return self.conv(self.activation(self.norm(x, zq)))


def _residual_stack(
    count: int, in_channels: int, channels: int, expand_ratio: int, zq_dim: int
) -> nn.Sequential:
    """``count`` blocks at width ``channels``; the first one narrows from ``in_channels``."""
    return nn.Sequential(
        *(
            ResidualBlock(
                in_channels if index == 0 else channels,
                channels,
                channels * expand_ratio,
                zq_dim,
            )
            for index in range(count)
        )
    )


def _run_blocks(blocks: nn.Sequential, x: Tensor, zq: Tensor) -> Tensor:
    for block in blocks:
        x = block(x, zq)
    return x


def _second_stage(
    upsample: nn.Module,
    blocks: nn.Sequential,
    head: Kandinsky6SRLatentUpscalerOutputHead,
    x: Tensor,
    zq: Tensor,
) -> Tensor:
    """``upsample -> blocks -> head``, with the head's first (norm) layer conditioned on ``zq``."""
    x = _run_blocks(blocks, upsample(x), zq)
    return head(x, zq)


class X2Branch(nn.Module):
    """Weights exclusive to the x2 path."""

    def __init__(self, config: Kandinsky6SRLatentUpscalerEntryConfig) -> None:
        super().__init__()
        c = config.in_channels
        w1, w2, w3 = config.stage_channels
        er = config.expand_ratio
        self.adapter = _residual_stack(config.x2_adapter_blocks, w1, w1, er, c)
        self.finisher = X2Finisher(w1)
        self.mid_blocks = _residual_stack(config.num_mid_blocks, w1, w2, er, c)
        self.upsample = PXSUpsample(w2)
        self.blocks = _residual_stack(config.num_post_blocks, w2, w3, er, c)
        self.output_proj = Kandinsky6SRLatentUpscalerOutputHead(w3, c, zq_dim=c)

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        x = _run_blocks(self.adapter, x, zq)
        x = self.finisher(x)
        x = _run_blocks(self.mid_blocks, x, zq)
        return _second_stage(self.upsample, self.blocks, self.output_proj, x, zq)


class Kandinsky6SRLatentUpscaler(nn.Module):
    """One bank entry: ``[B, C, T, h, w]`` -> ``[B, C, T, s*h, s*w]`` for ``s`` = 4, or 2 with
    ``enable_x2_entry``."""

    def __init__(self, config: Kandinsky6SRLatentUpscalerEntryConfig) -> None:
        super().__init__()
        c = config.in_channels
        w1, w2, w3 = config.stage_channels
        er = config.expand_ratio
        self.input_proj = _stem(c, w1)
        # The 2x deep-supervision head of training: part of the checkpoint, never run at inference.
        self.mid_output_head = _mid_output_head(w2, c)
        self.output_proj = Kandinsky6SRLatentUpscalerOutputHead(w3, c, zq_dim=c)
        self.upsample_1 = PXSUpsample(w1)
        self.upsample_2 = PXSUpsample(w2)
        self.pre_blocks = _residual_stack(config.num_pre_blocks, w1, w1, er, c)
        self.mid_blocks = _residual_stack(config.num_mid_blocks, w1, w2, er, c)
        self.post_blocks = _residual_stack(config.num_post_blocks, w2, w3, er, c)
        self.mid_input_proj: nn.Sequential | None = None
        self.x2_branch: X2Branch | None = None
        if config.enable_x2_entry:
            self.mid_input_proj = _stem(c, config.hidden_channels)
            self.x2_branch = X2Branch(config)

    def forward(self, z: Tensor, scale: int) -> Tensor:
        zq = z
        if scale == 2:
            if self.x2_branch is None or self.mid_input_proj is None:
                raise ValueError(
                    "this latent-upscaler entry has no x2 path (enable_x2_entry=false)"
                )
            return self.x2_branch(self.mid_input_proj(z), zq)
        if scale != 4:
            raise ValueError(
                f"latent-upscaler entries upsample by 2 or 4, got scale={scale!r}"
            )
        x = _run_blocks(self.pre_blocks, self.input_proj(z), zq)
        x = _run_blocks(self.mid_blocks, self.upsample_1(x), zq)
        return _second_stage(self.upsample_2, self.post_blocks, self.output_proj, x, zq)


class Kandinsky6SRLatentUpscalerBank(nn.Module, LayerwiseOffloadableModuleMixin):
    """The x2 / x4 latent-upscaler bank of a Kandinsky 6 SR bundle (``latent_upscaler/``).

    ``upscale(z, scale=...)`` upsamples an already scaled (``latent * scaling_factor``) KVAE latent
    ``[B, C, T, h, w]`` by ``scale`` in H and W with the entry serving that scale.

    """

    layerwise_offload_dit_group_enabled = False

    def __init__(
        self,
        models: Sequence[Mapping[str, Any]],
        scaling_factor: float = 1.0,
        scales: Sequence[int] = (2, 4),
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("latent upscaler config declares no `models` entries")
        entries = [
            Kandinsky6SRLatentUpscalerEntryConfig.from_dict(spec, index)
            for index, spec in enumerate(models)
        ]
        entries_by_scale = {entry.target_scale: entry for entry in entries}
        if len(entries_by_scale) != len(entries):
            raise ValueError("latent upscaler `models` has duplicate target scales")
        self._scales = tuple(int(scale) for scale in scales)
        if len(set(self._scales)) != len(self._scales) or set(self._scales) != set(
            entries_by_scale
        ):
            raise ValueError(
                "latent upscaler `scales` must match the target scales in `models`: "
                f"{self._scales} vs {tuple(entries_by_scale)}"
            )
        self.scaling_factor = float(scaling_factor)
        self._models = nn.ModuleList(
            [
                Kandinsky6SRLatentUpscaler(entries_by_scale[scale])
                for scale in self._scales
            ]
        )

        self.layer_names = []
        for index, entry in enumerate(self._models):
            self.layer_names += [
                f"_models.{index}.pre_blocks",
                f"_models.{index}.mid_blocks",
                f"_models.{index}.post_blocks",
            ]
            if entry.x2_branch is not None:
                self.layer_names += [
                    f"_models.{index}.x2_branch.adapter",
                    f"_models.{index}.x2_branch.mid_blocks",
                    f"_models.{index}.x2_branch.blocks",
                ]

    @property
    def scales(self) -> tuple[int, ...]:
        """The upscale factors this bank serves, ascending."""
        return tuple(sorted(self._scales))

    def for_scale(self, scale: int) -> nn.Module | None:
        """Return the bank entry matching ``scale``, or ``None`` if the bank does not serve it."""
        scale = int(scale)
        return (
            self._models[self._scales.index(scale)] if scale in self._scales else None
        )

    def upscale(self, latent: Tensor, *, scale: int) -> Tensor:
        """Run the entry for ``scale`` on a *scaled* latent ``[B, C, T, h, w]``."""
        entry = self.for_scale(scale)
        if entry is None:
            raise ValueError(
                f"the latent-upscaler bank has no x{scale} entry (scales: {self.scales})"
            )
        return entry(latent, scale)

    def forward(self, z: Tensor, *, scale: int) -> Tensor:
        return self.upscale(z, scale=scale)


EntryClass = Kandinsky6SRLatentUpscalerBank
