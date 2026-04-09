# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings as _CombinedTimestepGuidanceTextProjEmbeddings,
)
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings as _CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
)
from diffusers.models.embeddings import Timesteps as _Timesteps
from diffusers.models.embeddings import (
    get_timestep_embedding as timestep_embedding_diffusers,
)

from sglang.jit_kernel.timestep_embedding import (
    timestep_embedding as timestep_embedding_cuda,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.platforms import current_platform

_is_cuda = current_platform.is_cuda()


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
        if isinstance(patch_size, list | tuple):
            if len(patch_size) == 1:
                patch_size = (1, patch_size[0], patch_size[0])
            elif len(patch_size) == 2:
                patch_size = (1, patch_size[0], patch_size[1])
        else:
            patch_size = (1, patch_size, patch_size)

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
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            pt, ph, pw = self.patch_size

            if T % pt == 0 and H % ph == 0 and W % pw == 0:
                T_ = T // pt
                H_ = H // ph
                W_ = W // pw

                x = x.reshape(B, C, T_, pt, H_, ph, W_, pw)
                x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                x = x.reshape(B, T_ * H_ * W_, C * pt * ph * pw)

                w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
                x = F.linear(x, w, self.proj.bias)  # [B, T'*H'*W', embed_dim]

                if not self.flatten:
                    x = x.reshape(B, T_, H_, W_, -1).permute(0, 4, 1, 2, 3).contiguous()

                x = self.norm(x)
                return x

        # Fallback to Conv3d for non-5D input or indivisible spatial dims.
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Timesteps(_Timesteps):
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if _is_cuda:
            return timestep_embedding_cuda(
                timesteps,
                self.num_channels,
                flip_sin_to_cos=self.flip_sin_to_cos,
                downscale_freq_shift=self.downscale_freq_shift,
                scale=self.scale,
            )
        else:
            return timestep_embedding_diffusers(
                timesteps,
                self.num_channels,
                flip_sin_to_cos=self.flip_sin_to_cos,
                downscale_freq_shift=self.downscale_freq_shift,
                scale=self.scale,
            )


class CombinedTimestepGuidanceTextProjEmbeddings(
    _CombinedTimestepGuidanceTextProjEmbeddings
):
    def __init__(self, embedding_dim, pooled_projection_dim):
        nn.Module.__init__(self)

        # use sgld op
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        # use diffusers op
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )


class CombinedTimestepTextProjEmbeddings(_CombinedTimestepTextProjEmbeddings):
    def __init__(self, embedding_dim, pooled_projection_dim):
        nn.Module.__init__(self)

        # use sgld op
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        # use diffusers op
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )


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
            assert (
                t_freq.shape[0] % timestep_seq_len == 0
            ), "timestep length is not divisible by timestep_seq_len"
            batch_size = t_freq.shape[0] // timestep_seq_len
            t_freq = t_freq.unflatten(0, (batch_size, timestep_seq_len))
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
        self.linear = ColumnParallelLinear(
            hidden_size,
            hidden_size * factor,
            bias=True,
            gather_output=True,
            params_dtype=dtype,
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
