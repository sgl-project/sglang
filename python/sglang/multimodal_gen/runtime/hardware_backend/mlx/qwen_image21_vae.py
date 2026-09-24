# Copyright 2026 Qwen Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn


def silu(x):
    return nn.silu(x.astype(mx.float32)).astype(x.dtype)


class Conv2d(nn.Conv2d):
    def __call__(self, x):
        # torch convolutions add bias before rounding their accumulated output
        y = mx.conv2d(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return (y + self.bias.astype(mx.float32)).astype(x.dtype)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = mx.ones(channels)
        self.scale = channels**0.5

    def __call__(self, x):
        value = x.astype(mx.float32)
        denominator = mx.maximum(
            mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True)), 1e-12
        )
        # preserve the checkpoint's cast before scale and learned channel weights
        normalized = (value / denominator).astype(x.dtype)
        scaled = (normalized.astype(mx.float32) * self.scale).astype(x.dtype)
        return scaled * self.gamma


class AverageDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_factor, spatial_factor):
        super().__init__()
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.group_size = (
            in_channels * temporal_factor * spatial_factor**2 // out_channels
        )

    def __call__(self, x):
        batch, height, width, channels = x.shape
        factor = self.spatial_factor
        x = x.reshape(
            batch, height // factor, factor, width // factor, factor, channels
        )
        x = x.transpose(0, 1, 3, 5, 2, 4)
        # single-frame temporal downsampling includes left zero padding
        x = mx.expand_dims(x, axis=-3)
        x = mx.pad(x, [(0, 0)] * 4 + [(self.temporal_factor - 1, 0), (0, 0), (0, 0)])
        return x.reshape(
            batch, height // factor, width // factor, self.out_channels, self.group_size
        ).mean(axis=-1)


class DuplicateUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_factor, spatial_factor=2):
        super().__init__()
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.repeats = out_channels * temporal_factor * spatial_factor**2 // in_channels

    def __call__(self, x):
        batch, height, width, _ = x.shape
        factor = self.spatial_factor
        x = mx.repeat(x, self.repeats, axis=-1).reshape(
            batch,
            height,
            width,
            self.out_channels,
            self.temporal_factor,
            factor,
            factor,
        )
        # the single image is the last frame of the first temporal chunk
        x = x[:, :, :, :, -1].transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(batch, height * factor, width * factor, self.out_channels)


class Resample(nn.Module):
    def __init__(self, channels, up):
        super().__init__()
        self.up = up
        self.conv = Conv2d(
            channels, channels, 3, stride=1 if up else 2, padding=int(up)
        )

    def __call__(self, x):
        if self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        else:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = ChannelNorm(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = ChannelNorm(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def __call__(self, x):
        residual = self.conv_shortcut(x)
        x = self.conv1(silu(self.norm1(x)))
        return self.conv2(silu(self.norm2(x))) + residual


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.to_qkv = Conv2d(channels, channels * 3, 1)
        self.proj = Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def __call__(self, x):
        qkv = self.to_qkv(self.norm(x)).reshape(x.shape[0], 1, -1, x.shape[-1] * 3)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # the VAE's 768/1152-wide head uses the unfused path, whose intermediates
        # otherwise round to bf16 between the score and value matrix products
        attended = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
        ).astype(x.dtype)
        return x + self.proj(attended.reshape(x.shape))


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = [ResidualBlock(channels, channels) for _ in range(2)]
        self.attentions = [SpatialAttention(channels)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, temporal, down):
        super().__init__()
        self.resnets = [
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ]
        self.downsampler = Resample(out_channels, up=False) if down else nn.Identity()
        self.avg_shortcut = AverageDownsample(
            in_channels, out_channels, 2 if temporal else 1, 2 if down else 1
        )

    def __call__(self, x):
        residual = self.avg_shortcut(x)
        for block in self.resnets:
            x = block(x)
        return self.downsampler(x) + residual


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, temporal, up):
        super().__init__()
        self.resnets = [
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks + 1)
        ]
        self.upsampler = Resample(out_channels, up=True) if up else nn.Identity()
        self.avg_shortcut = (
            DuplicateUpsample(in_channels, out_channels, 2 if temporal else 1)
            if up
            else None
        )

    def __call__(self, x):
        residual = self.avg_shortcut(x) if self.avg_shortcut is not None else None
        for block in self.resnets:
            x = block(x)
        x = self.upsampler(x)
        return x + residual if residual is not None else x


class Encoder(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, temporal, in_channels):
        super().__init__()
        dims = [dim * factor for factor in (1, *dim_mult)]
        self.conv_in = Conv2d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = [
            DownBlock(
                a,
                b,
                num_res_blocks,
                temporal[i] if i < len(temporal) else False,
                i < len(dim_mult) - 1,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.mid_block = MidBlock(dims[-1])
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = Conv2d(dims[-1], z_dim * 2, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(silu(self.norm_out(self.mid_block(x))))


class Decoder(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, temporal, out_channels):
        super().__init__()
        dims = [dim * factor for factor in (dim_mult[-1], *dim_mult[::-1])]
        self.conv_in = Conv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        self.up_blocks = [
            UpBlock(
                a,
                b,
                num_res_blocks,
                temporal[i] if i < len(temporal) else False,
                i < len(dim_mult) - 1,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        ]
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = Conv2d(dims[-1], out_channels, 3, padding=1)

    def before_attention(self, x):
        return self.mid_block.resnets[0](self.conv_in(x))

    def after_attention(self, x):
        x = self.mid_block.resnets[1](x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(silu(self.norm_out(x)))

    def __call__(self, x):
        return self.after_attention(
            self.mid_block.attentions[0](self.before_attention(x))
        )


class QwenImage21VAE(nn.Module):
    """Single-image NHWC tensor core; latent scaling belongs to the pipeline."""

    def __init__(
        self,
        base_dim=96,
        decoder_base_dim=144,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        temperal_downsample=(False, True, True, True),
        in_channels=4,
        out_channels=4,
    ):
        super().__init__()
        self.encoder = Encoder(
            base_dim, z_dim, dim_mult, num_res_blocks, temperal_downsample, in_channels
        )
        self.quant_conv = Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder(
            decoder_base_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            temperal_downsample[::-1],
            out_channels,
        )

    def encode(self, image):
        return mx.split(self.quant_conv(self.encoder(image)), 2, axis=-1)[0]

    def decode(self, latents):
        return mx.clip(self.decoder(self.post_quant_conv(latents)), -1, 1)

    def compile_decode(self):
        """Build after loading weights; retain fp32 attention's evaluation boundaries."""
        before = mx.compile(
            lambda x: self.decoder.before_attention(self.post_quant_conv(x))
        )
        after = mx.compile(lambda x: mx.clip(self.decoder.after_attention(x), -1, 1))

        def decode(latents):
            # whole-graph compilation rewrites the unfused wide-head attention
            return after(self.decoder.mid_block.attentions[0](before(latents)))

        return decode
