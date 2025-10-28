# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert patch_size to 2-tuple
        if isinstance(patch_size, list | tuple):
            if len(patch_size) == 1:
                patch_size = (patch_size[0], patch_size[0])
        else:
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer="silu",
        frequency_embedding_size=256,
        max_period=10000,
        dtype=None,
        freq_dtype=torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            act_type=act_layer,
            dtype=dtype,
        )
        self.freq_dtype = freq_dtype

    def forward(
        self, t: torch.Tensor, timestep_seq_len: int | None = None
    ) -> torch.Tensor:
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period, dtype=self.freq_dtype
        ).to(self.mlp.fc_in.weight.dtype)
        if timestep_seq_len is not None:
            t_freq = t_freq.unflatten(0, (1, timestep_seq_len))
        # t_freq = t_freq.to(self.mlp.fc_in.weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: Tensor of shape [B] with timesteps
        dim: Embedding dimension
        max_period: Controls the minimum frequency of the embeddings

    Returns:
        Tensor of shape [B, dim] with embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=dtype, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ModulateProjection(nn.Module):
    """Modulation layer for DiT blocks."""

    def __init__(
        self,
        hidden_size: int,
        factor: int = 2,
        act_layer: str = "silu",
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.factor = factor
        self.hidden_size = hidden_size
        self.linear = ReplicatedLinear(
            hidden_size, hidden_size * factor, bias=True, params_dtype=dtype
        )
        self.act = get_act_fn(act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        x, _ = self.linear(x)
        return x


def unpatchify(x, t, h, w, patch_size, channels) -> torch.Tensor:
    """
    Convert patched representation back to image space.

    Args:
        x: Tensor of shape [B, T*H*W, C*P_t*P_h*P_w]
        t, h, w: Temporal and spatial dimensions

    Returns:
        Unpatchified tensor of shape [B, C, T*P_t, H*P_h, W*P_w]
    """
    assert x.ndim == 3, f"x.ndim: {x.ndim}"
    assert len(patch_size) == 3, f"patch_size: {patch_size}"
    assert t * h * w == x.shape[1], f"t * h * w: {t * h * w}, x.shape[1]: {x.shape[1]}"
    c = channels
    pt, ph, pw = patch_size

    x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
    x = torch.einsum("nthwcopq->nctohpwq", x)
    imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

    return imgs
