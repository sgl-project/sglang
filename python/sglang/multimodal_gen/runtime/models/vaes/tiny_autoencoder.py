# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2023 Ollin Boer Bohan
# Copyright (c) 2025 Ollin Boer Bohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tiny AutoEncoder networks adapted from madebyollin/taesd and taehv.

Image tensors use BCHW and video tensors use BCTHW. Encoder inputs and decoder
outputs use unit-range pixels; latent inputs follow each trained checkpoint's
diffusion-normalized latent space. Video memories belong to a single call.
"""

from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = (
                lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False),
                n_in * 4,
            )
            self.pool = nn.Sequential(
                conv1x1(n_in, n_gn),
                nn.GroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                conv1x1(n_gn, n_in),
            )

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder(latent_channels=4, use_midblock_gn=False, image_channels=3):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(image_channels, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4, use_midblock_gn=False, image_channels=3):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 64),
        nn.ReLU(),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        Block(64, 64, **mb_kw),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, image_channels),
    )


def F16Encoder(latent_channels=64, image_channels=3):
    """Encoder variant with 16x spatial downscaling, 2x2 input patchify, and wider low-res stages."""
    return nn.Sequential(
        nn.PixelUnshuffle(2),
        conv(image_channels * 4, 64),
        nn.ReLU(inplace=True),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 128, stride=2, bias=False),
        Block(128, 128),
        Block(128, 128),
        Block(128, 128),
        conv(128, 256, stride=2, bias=False),
        Block(256, 256),
        Block(256, 256),
        Block(256, 256),
        conv(256, latent_channels),
    )


def F16Decoder(latent_channels=64, image_channels=3):
    """Decoder variant with 16x spatial upscaling, wider low-res stages, and 2x2 output patchify."""
    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 256),
        nn.ReLU(),
        Block(256, 256),
        Block(256, 256),
        Block(256, 256),
        nn.Upsample(scale_factor=2),
        conv(256, 128, bias=False),
        Block(128, 128),
        Block(128, 128),
        Block(128, 128),
        nn.Upsample(scale_factor=2),
        conv(128, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, image_channels * 4),
        nn.PixelShuffle(2),
    )


class TinyImageAutoencoder(nn.Module):
    """TAESD family operating on diffusion-normalized image latents."""

    def __init__(self, latent_channels=4, load_encoder=True, load_decoder=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_channels = 4 if latent_channels == 64 else 3
        channels = 32 if latent_channels == 128 else latent_channels
        encoder = F16Encoder if latent_channels == 64 else Encoder
        decoder = F16Decoder if latent_channels == 64 else Decoder
        kwargs = {"image_channels": self.image_channels}
        if latent_channels == 128:
            kwargs["use_midblock_gn"] = True
        self.encoder = encoder(channels, **kwargs) if load_encoder else None
        self.decoder = decoder(channels, **kwargs) if load_decoder else None

    def encode(self, pixels):
        latents = self.encoder(pixels)
        if self.latent_channels == 128:
            b, c, h, w = latents.shape
            latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
            latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(b, 128, h // 2, w // 2)
        return latents

    def decode(self, latents):
        video = latents.ndim == 5
        if video:
            b, c, t, h, w = latents.shape
            latents = latents.transpose(1, 2).reshape(b * t, c, h, w)
        if self.latent_channels == 128:
            b_images, _, h, w = latents.shape
            latents = latents.reshape(b_images, 32, 2, 2, h, w)
            latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(
                b_images, 32, h * 2, w * 2
            )
        pixels = self.decoder(latents)
        if video:
            pixels = pixels.reshape(b, t, *pixels.shape[1:]).transpose(1, 2)
        return pixels


class MemBlock(nn.Module):
    def __init__(self, channels, leaky=False):
        super().__init__()

        def activation():
            return nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            conv(channels * 2, channels),
            activation(),
            conv(channels, channels),
            activation(),
            conv(channels, channels),
        )
        self.act = activation()

    def forward(self, x, past):
        return self.act(self.conv(torch.cat((x, past), dim=1)) + x)


class TemporalPool(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, x):
        _, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TemporalGrow(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x):
        _, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


def _run_video_layers(layers, x, patch_size, decode):
    batch = x.shape[0]
    memory = [None] * len(layers)
    queue = deque((frame, 0) for frame in x.unbind(1))
    output = []
    while queue:
        frame, index = queue.popleft()
        if index == 0 and not decode:
            frame = F.pixel_unshuffle(frame, patch_size)
        if index == len(layers):
            output.append(F.pixel_shuffle(frame, patch_size) if decode else frame)
            continue
        layer = layers[index]
        if isinstance(layer, MemBlock):
            past = torch.zeros_like(frame) if memory[index] is None else memory[index]
            result = layer(frame, past)
            memory[index] = frame.detach().clone()
            queue.appendleft((result, index + 1))
        elif isinstance(layer, TemporalPool):
            if memory[index] is None:
                memory[index] = []
            memory[index].append(frame)
            if len(memory[index]) == layer.stride:
                packed = torch.stack(memory[index], dim=1).flatten(0, 1)
                queue.appendleft((layer(packed), index + 1))
                memory[index] = []
        elif isinstance(layer, TemporalGrow):
            grown = layer(frame).reshape(batch, layer.stride, *frame.shape[1:])
            for result in reversed(grown.unbind(1)):
                queue.appendleft((result, index + 1))
        else:
            queue.appendleft((layer(frame), index + 1))
    return torch.stack(output, dim=1)


class TinyVideoAutoencoder(nn.Module):
    """TAEHV family with request-local temporal memories."""

    def __init__(self, latent_channels=16, load_encoder=True, load_decoder=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = {24: 2, 32: 2, 48: 2, 128: 4}.get(latent_channels, 1)
        self.temporal_ratio = 8 if latent_channels == 128 else 4
        self.is_h3 = latent_channels == 24
        leaky = latent_channels == 32

        def activation():
            return nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

        encoder = [conv(3 * self.patch_size**2, 64), activation()]
        for stride in (2, 2, 2) if latent_channels == 128 else (2, 2, 1):
            encoder.extend(
                [
                    TemporalPool(64, stride),
                    conv(64, 64, stride=2, bias=False),
                    MemBlock(64, leaky),
                    MemBlock(64, leaky),
                    MemBlock(64, leaky),
                ]
            )
        encoder.append(conv(64, latent_channels))
        decoder = [Clamp(), conv(latent_channels, 256), activation()]
        strides = (2, 2, 2) if latent_channels == 128 else (1, 2, 2)
        for channels, next_channels, stride in zip(
            (256, 128, 64), (128, 64, 64), strides
        ):
            decoder.extend(
                [
                    MemBlock(channels, leaky),
                    MemBlock(channels, leaky),
                    MemBlock(channels, leaky),
                    nn.Upsample(scale_factor=2),
                    TemporalGrow(channels, stride),
                    conv(channels, next_channels, bias=False),
                ]
            )
        decoder.extend([activation(), conv(64, 3 * self.patch_size**2)])
        self.encoder = nn.Sequential(*encoder) if load_encoder else None
        self.decoder = nn.Sequential(*decoder) if load_decoder else None

    def encode(self, pixels):
        x = pixels.transpose(1, 2)
        b, time = x.shape[:2]
        if self.is_h3:
            x = torch.cat((x, x[:, -1:].expand(-1, -time % 17, -1, -1, -1)), dim=1)
            x = F.pad(x.reshape(b, -1, 17, *x.shape[2:]), (0, 0, 0, 0, 0, 0, 3, 0))
            x = torch.cat(
                [
                    _run_video_layers(self.encoder, chunk, self.patch_size, False)
                    for chunk in x.unbind(1)
                ],
                dim=1,
            )
            x = x[:, :1] if time == 1 else x[:, :-3]
        else:
            x = torch.cat(
                (x, x[:, -1:].expand(-1, -time % self.temporal_ratio, -1, -1, -1)),
                dim=1,
            )
            x = _run_video_layers(self.encoder, x, self.patch_size, False)
        return x.transpose(1, 2)

    def decode(self, latents):
        if (
            latents.ndim != 5
            or latents.shape[1] != self.latent_channels
            or latents.shape[2] < 1
        ):
            raise ValueError(
                f"Tiny video latents must be [B,{self.latent_channels},T,H,W] with T >= 1"
            )
        x = _run_video_layers(
            self.decoder, latents.transpose(1, 2), self.patch_size, True
        )
        trim = self.temporal_ratio - 1
        if self.is_h3 and latents.shape[2] != 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % 20))
            x = x.unflatten(1, (-1, 20))[:, :, trim:].flatten(1, 2)
            x = x[:, :-12]
        else:
            x = x[:, trim:]
        return x.transpose(1, 2).clamp(0, 1)
