# SPDX-License-Identifier: Apache-2.0
"""Causal Kandinsky 6 SR KVAE: spatial x16 and temporal x4.

Encode/decode segments preserve causal-convolution caches; segmentation affects
numerics. Pixels normalize as x/128-1, encode returns an unscaled posterior mean,
and decode returns normalized pixels. encoder.* and decoder.* match the checkpoint."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEArchConfig,
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv3d,
    disable_spatial_parallel_decode,
    gather_and_trim_height,
    split_height_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    can_install_spatial_shard_parallel_decode,
    should_run_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanRMS_norm

# pixel frames per segment after the extra causal first frame
_SEGMENT_FRAMES = 16
# temporal chunks bound workspace and avoid int32 indexing overflow at 4x decode
_MAX_CONV_NUMEL = 2 * 10**9


def _segment_sizes(frames: int, stride: int) -> list[int]:
    """The first causal segment has one extra frame; the final segment may be short."""
    return [min(frames, stride + 1)] + [
        min(stride, frames - start) for start in range(stride + 1, frames, stride)
    ]


class _SegmentCache:
    """Causal state carried from one temporal segment to the next during a single encode / decode call."""

    def __init__(self) -> None:
        self.first = True
        self.padding: dict[nn.Module, torch.Tensor] = {}


def _silu(x: torch.Tensor) -> torch.Tensor:
    # output norms require this form; F.silu in resblocks rounds differently in bf16
    return x * torch.sigmoid(x)


class _K6RMSNorm(WanRMS_norm):
    """Wan RMS norm with the reference's unconditional fp32 reduction."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(x.float(), dim=1).to(x.dtype)
        return normed * self.scale * self.gamma + self.bias


class _ChunkedConv3d(nn.Conv3d):
    """``nn.Conv3d`` that convolves inputs larger than ``_MAX_CONV_NUMEL`` in overlapping
    temporal chunks (plain ``F.conv3d`` silently corrupts most elements once the input exceeds
    ~2**31 elements: it appears to do 32-bit index arithmetic internally)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_parts = math.ceil(x.numel() / _MAX_CONV_NUMEL)
        if num_parts <= 1:
            return super().forward(x)
        k, frames = self.kernel_size[0], x.size(2)
        if self.padding[0] != 0:
            raise ValueError(
                f"temporal chunking needs an unpadded time axis, got kernel {self.kernel_size}, "
                f"padding {self.padding}"
            )
        if self.stride[0] != 1:
            # strided outputs may cross chunk boundaries: use one kernel window per output
            stride = self.stride[0]
            window_numel = x.shape[0] * x.shape[1] * k * x.shape[3] * x.shape[4]
            if window_numel >= _MAX_CONV_NUMEL:
                raise ValueError(
                    f"frames are too big for Conv3d even one stride-{stride} window at a time "
                    f"({window_numel} elements per window)"
                )
            window_outputs = []
            for i in range(0, frames - k + 1, stride):
                window_outputs.append(super().forward(x[:, :, i : i + k]))
            return torch.cat(window_outputs, dim=2)
        step = math.ceil(frames / num_parts)
        last = frames - step * (math.ceil(frames / step) - 1)
        if k > 1 and min(step, last) < k:
            # A chunk would be shorter than the kernel: convolve one output frame at a time instead.
            windows = [(i, i + k) for i in range(frames - k + 1)]
        else:
            # Each input chunk [start, end) is prefixed with the previous chunk's last k - 1 frames.
            windows = [
                (max(start - k + 1, 0), min(start + step, frames))
                for start in range(0, frames, step)
            ]
        out: torch.Tensor | None = None
        for lo, hi in windows:
            y = super().forward(x[:, :, lo:hi])
            if out is None:
                out = y.new_empty((*y.shape[:2], frames - k + 1, *y.shape[3:]))
            out[:, :, lo : lo + y.size(2)] = y
        assert out is not None
        return out


class _SpatialChunkedConv3d(SpatialParallelConv3d, _ChunkedConv3d):
    # exchange halos once before temporal chunking; uneven height shards may need
    # different chunk counts, so collectives must stay outside that loop
    @classmethod
    def from_conv(cls, conv: _ChunkedConv3d, height_padding=None):
        with torch.device("meta"):
            parallel = cls(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
                height_padding=height_padding,
            )
        parallel.weight, parallel.bias = conv.weight, conv.bias
        return parallel

    def _direct_forward(self, x):
        x = F.pad(x, (0, 0, self.height_pad_top, self.height_pad_bottom, 0, 0))
        return _ChunkedConv3d.forward(self, x)


class _CausalConv3d(nn.Module):
    """Conv3d that is causal in time: the first segment is left-padded with copies of its
    first frame, later segments with the tail of the previous one; height / width are zero
    padded."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int | tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        time_kernel, height_kernel, width_kernel = kernel_size
        self.height_pad = height_kernel // 2
        self.width_pad = width_kernel // 2
        self.time_pad = time_kernel - 1
        self.time_kernel = time_kernel
        self.time_stride = stride[0]
        self.conv = _ChunkedConv3d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        s, k = self.time_stride, self.time_kernel
        batch, _, frames, height, width = x.shape
        x = F.pad(
            x, (self.width_pad, self.width_pad, self.height_pad, self.height_pad, 0, 0)
        )
        if cache.first:
            first_frame = x[:, :, :1]
            padding = first_frame.expand(-1, -1, self.time_pad, -1, -1)
        else:
            padding = cache.padding[self]

        # conv(cat([padding, x])) without materializing the concatenation of the whole segment:
        # only the outputs whose window overlaps ``padding`` are computed on a small
        # concatenated head.
        out_frames = frames if s == 1 else (frames + 1) // 2
        output = x.new_empty((batch, self.conv.out_channels, out_frames, height, width))
        head_out = math.ceil(padding.size(2) / s)
        head_in = head_out * s - padding.size(2)
        if head_out > 0:
            output[:, :, :head_out] = self.conv(
                torch.cat([padding, x[:, :, : head_in + k - s]], dim=2)
            )
        if head_out < out_frames:
            output[:, :, head_out:] = self.conv(x[:, :, head_in:])

        # Keep exactly the input frames the next segment's first output window needs.
        pad_offset = head_in + s * math.trunc((frames - head_in - k) / s) + s
        if pad_offset < 0:  # segment shorter than the kernel
            cache.padding[self] = torch.cat([padding[:, :, pad_offset:], x], dim=2)
        else:
            cache.padding[self] = x[:, :, pad_offset:].clone()
        return output


class _ResnetBlock3D(nn.Module):
    """norm -> SiLU -> causal conv, twice, plus a 1x1 shortcut when the width changes. Decoder
    blocks use the latent-conditioned ``_SpatialNorm3D`` (``zq_channels`` set)."""

    def __init__(
        self, in_channels: int, out_channels: int, zq_channels: int | None = None
    ) -> None:
        super().__init__()
        if zq_channels is None:
            self.norm1: nn.Module = _K6RMSNorm(in_channels, images=False)
            self.norm2: nn.Module = _K6RMSNorm(out_channels, images=False)
        else:
            self.norm1 = _SpatialNorm3D(in_channels, zq_channels)
            self.norm2 = _SpatialNorm3D(out_channels, zq_channels)
        self.conv1 = _CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.conv2 = _CausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.nin_shortcut = _ChunkedConv3d(in_channels, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, cache: _SegmentCache, zq: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x) if zq is None else self.norm1(x, zq, cache)
        h = F.silu(h, inplace=True)
        h = self.conv1(h, cache)
        h = self.norm2(h) if zq is None else self.norm2(h, zq, cache)
        h = F.silu(h, inplace=True)
        h = self.conv2(h, cache)
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


def _chunked_interpolate_nearest(
    x: torch.Tensor, size: tuple[int, int, int], channels: int = 32
) -> torch.Tensor:
    """Nearest interpolation in channel chunks, bounding memory and int32 indexing."""
    if x.shape[1] <= channels:
        return F.interpolate(x, size=size, mode="nearest")
    return torch.cat(
        [
            F.interpolate(chunk, size=size, mode="nearest")
            for chunk in torch.split(x, channels, dim=1)
        ],
        dim=1,
    )


class _SpatialNorm3D(nn.Module):
    """RMS norm modulated by the latent: ``norm(f) * conv_y(zq) + conv_b(zq)`` with ``zq``
    nearest-resized to ``f``."""

    def __init__(self, f_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm_layer = _K6RMSNorm(f_channels, images=False)
        self.conv_y = _ChunkedConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = _ChunkedConv3d(zq_channels, f_channels, kernel_size=1)

    def forward(
        self, f: torch.Tensor, zq: torch.Tensor, cache: _SegmentCache
    ) -> torch.Tensor:
        frames, height, width = f.shape[-3:]
        if cache.first:
            # The causal first latent frame maps to exactly one pixel frame; the rest cover the
            # remaining frames.
            zq_first = _chunked_interpolate_nearest(zq[:, :, :1], (1, height, width))
            if zq.size(2) > 1:
                zq_rest = _chunked_interpolate_nearest(
                    zq[:, :, 1:], (frames - 1, height, width)
                )
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            zq = _chunked_interpolate_nearest(zq, (frames, height, width))
        norm_f = self.norm_layer(f)
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))
        return norm_f


class _PXSDownsample(nn.Module):
    """x2 spatial (strided conv + channel-averaged pixel-unshuffle) and optional x2 causal
    temporal (two causal convs + average pooling) downsample; doubles the channels."""

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        out_channels = 2 * in_channels
        self.spatial_conv = _ChunkedConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.compress_time = compress_time
        if compress_time:
            self.temporal_conv = nn.Sequential(
                _CausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1)),
                _CausalConv3d(
                    out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)
                ),
            )
        self.linear = _ChunkedConv3d(out_channels, out_channels, kernel_size=1)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        pxs = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), 2)
        pxs = pxs.view(b * t, 2 * c, 2, h // 2, w // 2).mean(dim=2)
        pxs = pxs.reshape(b, t, 2 * c, h // 2, w // 2).permute(0, 2, 1, 3, 4)
        return self.spatial_conv(x) + pxs

    def _temporal(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        b, c, t, h, w = x.shape
        pooled = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
        if cache.first:  # the causal first frame is kept as is
            first, rest = pooled[..., :1], pooled[..., 1:]
            pooled = (
                torch.cat([first, F.avg_pool1d(rest, kernel_size=2, stride=2)], dim=-1)
                if t > 1
                else first
            )
        else:
            pooled = F.avg_pool1d(pooled, kernel_size=2, stride=2)
        pooled = pooled.reshape(b, h, w, c, -1).permute(0, 3, 4, 1, 2)
        conv = self.temporal_conv[1](self.temporal_conv[0](x, cache), cache)
        return conv + pooled

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        out = self._spatial(x)
        if self.compress_time:
            out = self._temporal(out, cache)
        return self.linear(out)


class _PXSUpsample(nn.Module):
    """Optional x2 causal temporal (frame repeat + causal conv residual) then x2 spatial
    (nearest + conv residual) upsample."""

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        self.spatial_conv = _ChunkedConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.compress_time = compress_time
        if compress_time:
            self.temporal_conv = _CausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1)
            )
        self.linear = _ChunkedConv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        if self.compress_time:
            x = x.repeat_interleave(2, dim=2)
            if cache.first:  # the causal first frame is not duplicated
                x = x[:, :, 1:]
            x = self.temporal_conv(x, cache) + x
        frames, height, width = x.shape[-3:]
        x = _chunked_interpolate_nearest(x, (frames, height * 2, width * 2))
        x.add_(self.spatial_conv(x))
        return self.linear(x)


def _level_module(**children: nn.Module) -> nn.Module:
    module = nn.Module()
    for name, child in children.items():
        setattr(module, name, child)
    return module


class _Encoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: list[float],
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        temporal_compress_times: int,
        temporal_compress_start_level: int,
    ) -> None:
        super().__init__()
        time_levels = range(
            temporal_compress_start_level,
            temporal_compress_start_level + int(math.log2(temporal_compress_times)),
        )
        self.conv_in = _CausalConv3d(in_channels, round(ch * ch_mult[0]), kernel_size=3)
        self.down = nn.ModuleList()
        block_in = round(ch * ch_mult[0])
        for level, mult in enumerate(ch_mult):
            block_out = round(ch * mult)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(_ResnetBlock3D(block_in, block_out))
                block_in = block_out
            down = _level_module(block=blocks)
            if level != len(ch_mult) - 1:
                down.downsample = _PXSDownsample(
                    block_in, compress_time=level in time_levels
                )
                block_in *= 2
            self.down.append(down)
        self.mid = _level_module(
            block_1=_ResnetBlock3D(block_in, block_in),
            block_2=_ResnetBlock3D(block_in, block_in),
        )
        self.norm_out = _K6RMSNorm(block_in, images=False)
        self.conv_out = _CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        h = self.conv_in(x, cache)
        for down in self.down:
            for block in down.block:
                h = block(h, cache)
            if hasattr(down, "downsample"):
                h = down.downsample(h, cache)
        h = self.mid.block_2(self.mid.block_1(h, cache), cache)
        return self.conv_out(_silu(self.norm_out(h)), cache)


class _Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: list[float],
        num_res_blocks: int,
        out_ch: int,
        z_channels: int,
        temporal_compress_times: int,
        temporal_compress_start_level: int,
    ) -> None:
        super().__init__()
        num_levels = len(ch_mult)
        # Mirror of the encoder's temporally compressing levels.
        time_levels = range(
            num_levels
            - temporal_compress_start_level
            - int(math.log2(temporal_compress_times)),
            num_levels - temporal_compress_start_level,
        )
        block_in = round(ch * ch_mult[-1])
        self.conv_in = _CausalConv3d(z_channels, block_in, kernel_size=3)
        self.mid = _level_module(
            block_1=_ResnetBlock3D(block_in, block_in, z_channels),
            block_2=_ResnetBlock3D(block_in, block_in, z_channels),
        )
        up_levels: list[nn.Module] = []
        for level in reversed(range(num_levels)):
            block_out = round(ch * ch_mult[level])
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(_ResnetBlock3D(block_in, block_out, z_channels))
                block_in = block_out
            up = _level_module(block=blocks)
            if level != 0:
                up.upsample = _PXSUpsample(block_in, compress_time=level in time_levels)
            up_levels.insert(0, up)
        self.up = nn.ModuleList(up_levels)
        self.norm_out = _SpatialNorm3D(block_in, z_channels)
        self.conv_out = _CausalConv3d(block_in, out_ch, kernel_size=3)

    def forward(self, z: torch.Tensor, cache: _SegmentCache) -> torch.Tensor:
        h = self.conv_in(z, cache)
        h = self.mid.block_2(self.mid.block_1(h, cache, z), cache, z)
        for up in reversed(self.up):
            for block in up.block:
                h = block(h, cache, z)
            if hasattr(up, "upsample"):
                h = up.upsample(h, cache)
        return self.conv_out(_silu(self.norm_out(h, z, cache)), cache)


# constructor fields; resolution and training-only metadata do not affect the architecture
_ARCH_KEYS = (
    "ch",
    "ch_mult",
    "num_res_blocks",
    "z_channels",
    "temporal_compress_times",
)


def _arch_kwargs(conf: dict[str, Any], extra: str, which: str) -> dict[str, Any]:
    # never silently substitute RMSNorm for an unsupported group-norm checkpoint
    norm_type = conf.get("norm_type", "rms_norm")
    if norm_type != "rms_norm":
        raise ValueError(
            f"Kandinsky6SRVAE supports only {which}_config['norm_type'] == 'rms_norm' "
            f"(group-norm KVAEs are not implemented by this port), got {norm_type!r}"
        )
    missing = [key for key in (*_ARCH_KEYS, extra) if key not in conf]
    if missing:
        raise ValueError(f"Kandinsky6SRVAE {which}_config is missing {missing}")
    kwargs = {key: conf[key] for key in (*_ARCH_KEYS, extra)}
    kwargs["temporal_compress_start_level"] = conf.get(
        "temporal_compress_start_level", 0
    )
    return kwargs


class Kandinsky6SRVAE(nn.Module, LayerwiseOffloadableModuleMixin):
    """Segment-cached causal video KVAE used by the SR pipeline."""

    layerwise_offload_dit_group_enabled = False

    def __init__(self, config: Kandinsky6SRVAEConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.arch_config
        assert isinstance(arch, Kandinsky6SRVAEArchConfig)
        if not arch.encoder_config or not arch.decoder_config:
            raise ValueError(
                "Kandinsky6SRVAE needs `encoder_config` and `decoder_config` (KVAE "
                "architecture) in the component config.json."
            )
        enc = _arch_kwargs(dict(arch.encoder_config), "in_channels", "encoder")
        dec = _arch_kwargs(dict(arch.decoder_config), "out_ch", "decoder")
        self.temporal_compression = enc["temporal_compress_times"]
        self.encoder = _Encoder3D(**enc)
        self.decoder = _Decoder3D(**dec)
        self._spatial_parallel_decode_enabled = (
            can_install_spatial_shard_parallel_decode(config)
        )
        if self._spatial_parallel_decode_enabled:
            for module in self.decoder.modules():
                if isinstance(module, _CausalConv3d) and module.height_pad:
                    module.conv = _SpatialChunkedConv3d.from_conv(
                        module.conv, (module.height_pad, module.height_pad)
                    )
                    module.height_pad = 0
                elif isinstance(module, _PXSUpsample):
                    module.spatial_conv = _SpatialChunkedConv3d.from_conv(
                        module.spatial_conv
                    )
        # encode/decode invoke each block, not the surrounding level container
        self.layer_names = [
            f"{path}.{index}.block"
            for path, levels in (
                ("encoder.down", self.encoder.down),
                ("decoder.up", self.decoder.up),
            )
            for index in range(len(levels))
        ]

    @property
    def scaling_factor(self) -> float:
        return float(self.config.arch_config.scaling_factor)

    @property
    def spatial_factor(self) -> int:
        return int(self.config.arch_config.spatial_factor)

    @property
    def temporal_factor(self) -> int:
        return int(self.config.arch_config.temporal_factor)

    @staticmethod
    def normalize_data(data: torch.Tensor) -> torch.Tensor:
        return data / 128 - 1.0

    @staticmethod
    def denormalize_data(data: torch.Tensor) -> torch.Tensor:
        return (data + 1) * 128

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """``[B, C, T, H, W]`` normalized pixels -> ``(raw latent mean, pixel-frame segment sizes)``."""
        split_list = _segment_sizes(x.size(2), _SEGMENT_FRAMES)
        cache = _SegmentCache()
        latents = []
        for segment in torch.split(x, split_list, dim=2):
            moments = self.encoder(segment, cache)
            cache.first = False
            latents.append(
                moments.chunk(2, dim=1)[0]
            )  # deterministic posterior: the mean
        return torch.cat(latents, dim=2), split_list

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """``[B, C, T', h, w]`` raw latent -> ``.sample`` ``[B, 3, T, H, W]`` in the normalized pixel range."""
        if self._spatial_parallel_decode_enabled:
            world_size = get_decode_parallel_world_size()
            if (
                should_run_spatial_shard_parallel_decode(self.config, z)
                and z.shape[-2] >= world_size
            ):
                z, height = split_height_for_parallel_decode(
                    z,
                    expected_height=z.shape[-2] * self.spatial_factor,
                    world_size=world_size,
                    rank=get_decode_parallel_rank(),
                )
                return DecoderOutput(
                    sample=gather_and_trim_height(self._decode(z), height)
                )
            with disable_spatial_parallel_decode():
                return DecoderOutput(sample=self._decode(z))
        return DecoderOutput(sample=self._decode(z))

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        split_list = _segment_sizes(
            z.size(2), _SEGMENT_FRAMES // self.temporal_compression
        )
        cache = _SegmentCache()
        samples = []
        for chunk in torch.split(z, split_list, dim=2):
            samples.append(self.decoder(chunk, cache))
            cache.first = False
        return torch.cat(samples, dim=2)


EntryClass = Kandinsky6SRVAE

__all__ = ["Kandinsky6SRVAE"]
