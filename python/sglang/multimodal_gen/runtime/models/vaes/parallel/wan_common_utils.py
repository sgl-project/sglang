from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )

        _first_chunk = first_chunk.get() if first_chunk is not None else None
        if _first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.padding: tuple[int, int, int]
        # Set up causal padding
        self._padding: tuple[int, ...] = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        x = (
            x.to(self.weight.dtype) if current_platform.is_mps() else x
        )  # casting needed for mps since amp isn't supported
        return super().forward(x)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.
    """

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


is_first_frame = None
feat_cache = None
feat_idx = None
cache_t = None
first_chunk = None


def bind_context(
    is_first_frame_var,
    feat_cache_var,
    feat_idx_var,
    cache_t_value,
    first_chunk_var,
):
    global is_first_frame
    global feat_cache
    global feat_idx
    global cache_t
    global first_chunk
    is_first_frame = is_first_frame_var
    feat_cache = feat_cache_var
    feat_idx = feat_idx_var
    cache_t = cache_t_value
    first_chunk = first_chunk_var


def _ensure_bound():
    if (
        is_first_frame is None
        or feat_cache is None
        or feat_idx is None
        or cache_t is None
        or first_chunk is None
    ):
        raise RuntimeError("common_utils.bind_context() must be called before use.")


def resample_forward(self, x):
    _ensure_bound()
    b, c, t, h, w = x.size()
    first_frame = is_first_frame.get()
    if first_frame:
        assert t == 1
    _feat_cache = feat_cache.get()
    _feat_idx = feat_idx.get()
    if self.mode == "upsample3d":
        if _feat_cache is not None:
            idx = _feat_idx
            if _feat_cache[idx] is None:
                _feat_cache[idx] = "Rep"
                _feat_idx += 1
            else:
                cache_x = x[:, :, -cache_t:, :, :].clone()
                if (
                    cache_x.shape[2] < 2
                    and _feat_cache[idx] is not None
                    and _feat_cache[idx] != "Rep"
                ):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            _feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if (
                    cache_x.shape[2] < 2
                    and _feat_cache[idx] is not None
                    and _feat_cache[idx] == "Rep"
                ):
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                        dim=2,
                    )
                if _feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, _feat_cache[idx])
                _feat_cache[idx] = cache_x
                _feat_idx += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        elif not first_frame and hasattr(self, "time_conv"):
            x = self.time_conv(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
            x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    _feat_cache = feat_cache.get()
    _feat_idx = feat_idx.get()
    if self.mode == "downsample3d":
        if _feat_cache is not None:
            idx = _feat_idx
            if _feat_cache[idx] is None:
                _feat_cache[idx] = x.clone()
                _feat_idx += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([_feat_cache[idx][:, :, -1:, :, :], x], 2))
                _feat_cache[idx] = cache_x
                _feat_idx += 1
            feat_cache.set(_feat_cache)
            feat_idx.set(_feat_idx)
        elif not first_frame and hasattr(self, "time_conv"):
            x = self.time_conv(x)
    return x


def residual_block_forward(self, x):
    _ensure_bound()
    # Apply shortcut connection
    h = self.conv_shortcut(x)

    # First normalization and activation
    x = self.norm1(x)
    x = self.nonlinearity(x)

    _feat_cache = feat_cache.get()
    _feat_idx = feat_idx.get()
    if _feat_cache is not None:
        idx = _feat_idx
        cache_x = x[:, :, -cache_t:, :, :].clone()
        if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
            cache_x = torch.cat(
                [
                    _feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )

        x = self.conv1(x, _feat_cache[idx])
        _feat_cache[idx] = cache_x
        _feat_idx += 1
        feat_cache.set(_feat_cache)
        feat_idx.set(_feat_idx)
    else:
        x = self.conv1(x)

    # Second normalization and activation
    x = self.norm2(x)
    x = self.nonlinearity(x)

    # Dropout
    x = self.dropout(x)

    _feat_cache = feat_cache.get()
    _feat_idx = feat_idx.get()
    if _feat_cache is not None:
        idx = _feat_idx
        cache_x = x[:, :, -cache_t:, :, :].clone()
        if cache_x.shape[2] < 2 and _feat_cache[idx] is not None:
            cache_x = torch.cat(
                [
                    _feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )

        x = self.conv2(x, _feat_cache[idx])
        _feat_cache[idx] = cache_x
        _feat_idx += 1
        feat_cache.set(_feat_cache)
        feat_idx.set(_feat_idx)
    else:
        x = self.conv2(x)

    # Add residual connection
    return x + h


def attention_block_forward(self, x):
    identity = x
    batch_size, channels, num_frames, height, width = x.size()
    x = x.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, channels, height, width
    )
    x = self.norm(x)

    # compute query, key, value
    qkv = self.to_qkv(x)
    qkv = qkv.reshape(batch_size * num_frames, 1, channels * 3, -1)
    qkv = qkv.permute(0, 1, 3, 2).contiguous()
    q, k, v = qkv.chunk(3, dim=-1)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    x = (
        x.squeeze(1)
        .permute(0, 2, 1)
        .reshape(batch_size * num_frames, channels, height, width)
    )

    # output projection
    x = self.proj(x)

    # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
    x = x.view(batch_size, num_frames, channels, height, width)
    x = x.permute(0, 2, 1, 3, 4)

    return x + identity


def mid_block_forward(self, x):
    # First residual block
    x = self.resnets[0](x)

    # Process through attention and residual blocks
    for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
        if attn is not None:
            x = attn(x)

        x = resnet(x)

    return x


def residual_down_block_forward(self, x):
    x_copy = x
    for resnet in self.resnets:
        x = resnet(x)
    if self.downsampler is not None:
        x = self.downsampler(x)

    return x + self.avg_shortcut(x_copy)


def residual_up_block_forward(self, x):
    if self.avg_shortcut is not None:
        x_copy = x

    for resnet in self.resnets:
        x = resnet(x)

    if self.upsampler is not None:
        x = self.upsampler(x)

    if self.avg_shortcut is not None:
        x = x + self.avg_shortcut(x_copy)

    return x


def up_block_forward(self, x):
    for resnet in self.resnets:
        x = resnet(x)

    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x
