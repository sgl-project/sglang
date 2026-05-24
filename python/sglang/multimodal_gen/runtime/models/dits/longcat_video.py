# Copied and adapted from: ../LongCat-Video/longcat_video/modules/longcat_video_dit.py
# Auxiliary blocks are copied from:
#   ../LongCat-Video/longcat_video/modules/blocks.py
#   ../LongCat-Video/longcat_video/modules/attention.py
#   ../LongCat-Video/longcat_video/modules/rope_3d.py

# SPDX-License-Identifier: Apache-2.0
import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from sglang.multimodal_gen.configs.models.dits import LongCatVideoConfig
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm, RMSNorm
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _autocast_float32(device: torch.device):
    if device.type == "cuda":
        return amp.autocast(device_type="cuda", dtype=torch.float32)
    return nullcontext()


def _assert_no_sp() -> None:
    try:
        sp_world_size = get_sp_world_size()
    except AssertionError:
        sp_world_size = 1
    if sp_world_size > 1:
        raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.size()
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if d % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            d, h, w = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, h, w)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def modulate_fp32(norm_func, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    assert (
        shift.dtype == torch.float32 and scale.dtype == torch.float32
    ), f"modulate_fp32 requires float32 inputs; got shift={shift.dtype}, scale={scale.dtype}"
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    return x.to(dtype)


class FinalLayer_FP32(nn.Module):
    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim
        self.norm_final = FP32LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, latent_shape):
        assert t.dtype == torch.float32
        bsz, n_tokens, channels = x.shape
        n_frames, _, _ = latent_shape
        with _autocast_float32(x.device):
            shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1)
            x = modulate_fp32(
                self.norm_final,
                x.view(bsz, n_frames, -1, channels),
                shift,
                scale,
            ).view(bsz, n_tokens, channels)
            x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor, dtype: torch.dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        return self.mlp(t_freq)


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        caption = self.y_proj(caption)
        return caption


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, "Dim must be a multiply of 8 for 3D RoPE."
        self.base = 10000
        self.freqs_dict = {}

    def register_grid_size(self, grid_size):
        if grid_size not in self.freqs_dict:
            self.freqs_dict.update({grid_size: self.precompute_freqs_cis_3d(grid_size)})

    def precompute_freqs_cis_3d(self, grid_size):
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (
            self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
        )
        freqs_h = 1.0 / (
            self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
        )
        freqs_w = 1.0 / (
            self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
        )
        grid_t = torch.from_numpy(
            np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
        ).float()
        grid_h = torch.from_numpy(
            np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        ).float()
        grid_w = torch.from_numpy(
            np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        ).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat(
            (
                freqs_t[:, None, None, :],
                freqs_h[None, :, None, :],
                freqs_w[None, None, :, :],
            ),
            dim=-1,
        )
        return rearrange(freqs, "T H W D -> (T H W) D")

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_size):
        if grid_size not in self.freqs_dict:
            self.register_grid_size(grid_size)

        freqs_cis = self.freqs_dict[grid_size]
        if freqs_cis.device != q.device:
            freqs_cis = freqs_cis.to(q.device)
            self.freqs_dict[grid_size] = (
                freqs_cis  # Cache GPU tensor to avoid repeated CPU→GPU copies
            )
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float()
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)
        return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict | None = None,
        cp_split_hw=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if enable_bsa:
            raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")
        if cp_split_hw is not None:
            raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.rope_3d = RotaryPositionalEmbedding(self.head_dim)

    def _process_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape):
        if self.enable_flashattn3:
            from flash_attn_interface import flash_attn_func

            q = rearrange(q, "B H S D -> B S H D").contiguous()
            k = rearrange(k, "B H S D -> B S H D").contiguous()
            v = rearrange(v, "B H S D -> B S H D").contiguous()
            x, *_ = flash_attn_func(q, k, v, softmax_scale=self.scale)
            return rearrange(x, "B S H D -> B H S D")
        if self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func

                q = rearrange(q, "B H S D -> B S H D")
                k = rearrange(k, "B H S D -> B S H D")
                v = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.scale)
                return rearrange(x, "B S H D -> B H S D")
            except (ImportError, Exception) as e:
                if not getattr(self, "_fa2_warned", False):
                    logger.warning(
                        "FlashAttention2 unavailable (%s), falling back to SDPA. "
                        "Install flash-attn for faster attention.",
                        type(e).__name__,
                    )
                    self._fa2_warned = True
        if self.enable_xformers:
            import xformers.ops

            q = rearrange(q, "B H M K -> B M H K")
            k = rearrange(k, "B H M K -> B M H K")
            v = rearrange(v, "B H M K -> B M H K")
            x = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=None
            )
            return rearrange(x, "B M H K -> B H M K")

        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def forward(
        self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False
    ):
        bsz, n_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, n_tokens, 3, self.num_heads, self.head_dim).permute(
            (2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape)

        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (n_tokens // shape[0])
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v, shape)

        x = x.transpose(1, 2).reshape(bsz, n_tokens, channels)
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        return x

    def forward_with_kv_cache(
        self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None
    ):
        bsz, n_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, n_tokens, 3, self.num_heads, self.head_dim).permute(
            (2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        n_frames, height, width = shape
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, bsz]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(bsz, 1, 1, 1)
            v_cache = v_cache.repeat(bsz, 1, 1, 1)

        k_full = torch.cat([k_cache, k], dim=2).contiguous()
        v_full = torch.cat([v_cache, v], dim=2).contiguous()
        # RoPE is only applied when num_cond_latents > 0 (conditioning latents present).
        # When num_cond_latents is None or 0, the cached keys/values already have RoPE
        # baked in from the return_kv pass; re-applying here would double-rotate them.
        # This matches the prototype behavior (LongCat-Video/longcat_video/modules/attention.py).
        if num_cond_latents is not None and num_cond_latents > 0:
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=2).contiguous()
            q_padding, k_full = self.rope_3d(
                q_padding, k_full, (n_frames + num_cond_latents, height, width)
            )
            q = q_padding[:, :, -n_tokens:].contiguous()

        x = self._process_attn(q, k_full, v_full, shape)
        x = x.transpose(1, 2).reshape(bsz, n_tokens, channels)
        return self.proj(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        enable_flashattn3=False,
        enable_flashattn2=False,
        enable_xformers=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, x: torch.Tensor, cond: torch.Tensor, kv_seqlen):
        bsz, n_tokens, channels = x.shape
        assert channels == self.dim and cond.shape[2] == self.dim
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.enable_flashattn3:
            from flash_attn_interface import flash_attn_varlen_func

            x = flash_attn_varlen_func(
                q=q[0],
                k=k[0],
                v=v[0],
                cu_seqlens_q=torch.tensor([0] + [n_tokens] * bsz, device=q.device)
                .cumsum(0)
                .to(torch.int32),
                cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device)
                .cumsum(0)
                .to(torch.int32),
                max_seqlen_q=n_tokens,
                max_seqlen_k=max(kv_seqlen),
            )[0]
        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_varlen_func

                x = flash_attn_varlen_func(
                    q=q[0],
                    k=k[0],
                    v=v[0],
                    cu_seqlens_q=torch.tensor([0] + [n_tokens] * bsz, device=q.device)
                    .cumsum(0)
                    .to(torch.int32),
                    cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device)
                    .cumsum(0)
                    .to(torch.int32),
                    max_seqlen_q=n_tokens,
                    max_seqlen_k=max(kv_seqlen),
                )
            except (ImportError, Exception) as e:
                if not getattr(self, "_fa2_warned", False):
                    logger.warning(
                        "FlashAttention2 unavailable (%s), falling back to SDPA for cross-attention. "
                        "Install flash-attn for faster attention.",
                        type(e).__name__,
                    )
                    self._fa2_warned = True
                outputs = []
                offset = 0
                q = rearrange(q, "one s h d -> one h s d")
                k = rearrange(k, "one s h d -> one h s d")
                v = rearrange(v, "one s h d -> one h s d")
                for i, seqlen in enumerate(kv_seqlen):
                    q_i = q[:, :, i * n_tokens : (i + 1) * n_tokens]
                    k_i = k[:, :, offset : offset + seqlen]
                    v_i = v[:, :, offset : offset + seqlen]
                    out = F.scaled_dot_product_attention(q_i, k_i, v_i)
                    outputs.append(rearrange(out[0], "h s d -> s h d"))
                    offset += seqlen
                x = torch.cat(outputs, dim=0)
        elif self.enable_xformers:
            import xformers.ops

            attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [n_tokens] * bsz, kv_seqlen
            )
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        else:
            outputs = []
            offset = 0
            q = rearrange(q, "one s h d -> one h s d")
            k = rearrange(k, "one s h d -> one h s d")
            v = rearrange(v, "one s h d -> one h s d")
            for i, seqlen in enumerate(kv_seqlen):
                q_i = q[:, :, i * n_tokens : (i + 1) * n_tokens]
                k_i = k[:, :, offset : offset + seqlen]
                v_i = v[:, :, offset : offset + seqlen]
                out = F.scaled_dot_product_attention(q_i, k_i, v_i)
                outputs.append(rearrange(out[0], "h s d -> s h d"))
                offset += seqlen
            x = torch.cat(outputs, dim=0)

        x = x.view(bsz, -1, channels)
        return self.proj(x)

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)

        bsz, n_tokens, channels = x.shape
        assert shape is not None, "SHOULD pass in the shape"
        num_cond_latents_thw = num_cond_latents * (n_tokens // shape[0])
        x_noise = x[:, num_cond_latents_thw:]
        output_noise = self._process_cross_attn(x_noise, cond, kv_seqlen)
        output = torch.cat(
            [
                torch.zeros(
                    (bsz, num_cond_latents_thw, channels),
                    dtype=output_noise.dtype,
                    device=output_noise.device,
                ),
                output_noise,
            ],
            dim=1,
        ).contiguous()
        return output


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        self.mod_norm_attn = FP32LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.mod_norm_ffn = FP32LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.pre_crs_attn_norm = FP32LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=True
        )
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(
            dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio)
        )

    def forward(
        self,
        x,
        y,
        t,
        y_seqlen,
        latent_shape,
        num_cond_latents=None,
        return_kv=False,
        kv_cache=None,
        skip_crs_attn=False,
    ):
        x_dtype = x.dtype
        bsz, n_tokens, channels = x.shape
        n_frames, _, _ = latent_shape

        with _autocast_float32(x.device):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
            )

        x_m = modulate_fp32(
            self.mod_norm_attn,
            x.view(bsz, n_frames, -1, channels),
            shift_msa,
            scale_msa,
        ).view(bsz, n_tokens, channels)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                kv_cache=kv_cache,
            )
        else:
            attn_outputs = self.attn(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
            )

        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with _autocast_float32(x.device):
            x = x + (gate_msa * x_s.view(bsz, -1, n_tokens // n_frames, channels)).view(
                bsz, -1, channels
            )
        x = x.to(x_dtype)

        if not skip_crs_attn:
            if kv_cache is not None:
                # Behavioral difference between normal and kv-cache paths:
                # - Normal path: cross_attn zeros the output for cond latents
                #   (MultiHeadCrossAttention inserts a zero block for the first
                #   num_cond_latents_thw tokens when num_cond_latents > 0).
                # - kv-cache path: num_cond_latents is set to None here so cond
                #   latents receive full cross-attention. During cached inference,
                #   only noise tokens are forwarded (cond latents are not in
                #   hidden_states), so the zeroing logic is not needed.
                num_cond_latents = None
            x = x + self.cross_attn(
                self.pre_crs_attn_norm(x),
                y,
                y_seqlen,
                num_cond_latents=num_cond_latents,
                shape=latent_shape,
            )

        x_m = modulate_fp32(
            self.mod_norm_ffn,
            x.view(bsz, -1, n_tokens // n_frames, channels),
            shift_mlp,
            scale_mlp,
        ).view(bsz, -1, channels)
        x_s = self.ffn(x_m)
        with _autocast_float32(x.device):
            x = x + (gate_mlp * x_s.view(bsz, -1, n_tokens // n_frames, channels)).view(
                bsz, -1, channels
            )
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        return x


class LongCatVideoTransformer3DModel(CachableDiT):
    _fsdp_shard_conditions = LongCatVideoConfig()._fsdp_shard_conditions
    _compile_conditions = LongCatVideoConfig()._compile_conditions
    _supported_attention_backends = LongCatVideoConfig()._supported_attention_backends
    param_names_mapping = LongCatVideoConfig().param_names_mapping
    reverse_param_names_mapping = LongCatVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LongCatVideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: LongCatVideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        if config.cp_split_hw is not None:
            raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")
        if config.enable_bsa:
            raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")
        super().__init__(config=config, hf_config=hf_config)

        self.patch_size = tuple(config.patch_size)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.cp_split_hw = config.cp_split_hw

        self.x_embedder = PatchEmbed3D(
            self.patch_size, self.in_channels, self.hidden_size
        )
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=config.adaln_tembed_dim,
            frequency_embedding_size=config.frequency_embedding_size,
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    adaln_tembed_dim=config.adaln_tembed_dim,
                    enable_flashattn3=config.enable_flashattn3,
                    enable_flashattn2=config.enable_flashattn2,
                    enable_xformers=config.enable_xformers,
                    enable_bsa=config.enable_bsa,
                    bsa_params=config.bsa_params,
                    cp_split_hw=config.cp_split_hw,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = FinalLayer_FP32(
            self.hidden_size,
            np.prod(self.patch_size),
            self.out_channels,
            config.adaln_tembed_dim,
        )
        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = config.text_tokens_zero_pad
        self.lora_dict = {}  # TODO: LoRA not yet implemented for LongCat-Video
        self.active_loras = []  # TODO: LoRA not yet implemented for LongCat-Video

    def post_load_weights(self) -> None:
        # Pre-warm RoPE cache for common generation grid sizes to avoid
        # CPU→GPU copies on the first denoising step.
        # Grid size = (latent_T, latent_H, latent_W) where:
        #   latent_T = (num_frames - 1) // 4 + 1  (VAE temporal stride = 4)
        #   latent_H = height // 16               (VAE spatial stride 8 × patch 2)
        #   latent_W = width  // 16
        default_grid_sizes = [
            (23, 30, 52),  # 93 frames, 480×832
            (7, 30, 52),  # 25 frames,  480×832
            (23, 52, 30),  # 93 frames, 832×480
            (7, 52, 30),  # 25 frames,  832×480
        ]
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return
        for block in self.blocks:
            rope = block.attn.rope_3d
            for grid_size in default_grid_sizes:
                if grid_size not in rope.freqs_dict:
                    rope.register_grid_size(grid_size)
                freqs = rope.freqs_dict[grid_size]
                if freqs.device != device:
                    rope.freqs_dict[grid_size] = freqs.to(device)

    def enable_bsa(self):
        raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")

    def disable_bsa(self):
        for block in self.blocks:
            block.attn.enable_bsa = False

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        return_kv=False,
        kv_cache_dict=None,
        skip_crs_attn=False,
        offload_kv_cache=False,
    ):
        _assert_no_sp()
        assert not (return_kv and kv_cache_dict), (
            "return_kv=True and kv_cache_dict are mutually exclusive: "
            "when kv_cache_dict is provided the model reads from cache and "
            "does not produce new KV tensors."
        )
        if self.cp_split_hw is not None:
            raise NotImplementedError("LongCat T2V MVP only supports no CP/SP.")

        if kv_cache_dict is None:
            kv_cache_dict = {}

        bsz, _, n_frames_raw, height_raw, width_raw = hidden_states.shape
        n_frames = n_frames_raw // self.patch_size[0]
        height = height_raw // self.patch_size[1]
        width = width_raw // self.patch_size[2]

        assert (
            self.patch_size[0] == 1
        ), "Currently, 3D x_embedder should not compress the temporal dimension."

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, n_frames)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)

        with _autocast_float32(hidden_states.device):
            t = self.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(bsz, n_frames, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(
                encoder_attention_mask.dtype
            )

        if encoder_attention_mask is not None:
            # encoder_attention_mask is [batch, seq] from longcat_text_postprocess;
            # the squeeze calls are no-ops on a 2D tensor but retained for shape-safety
            # in case the mask arrives with extra singleton dims from other code paths.
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[
                0
            ]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            block_outputs = block(
                hidden_states,
                encoder_hidden_states,
                t,
                y_seqlens,
                (n_frames, height, width),
                num_cond_latents,
                return_kv,
                kv_cache_dict.get(i, None),
                skip_crs_attn,
            )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (
                        kv_cache[0].contiguous(),
                        kv_cache[1].contiguous(),
                    )
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (n_frames, height, width))
        hidden_states = self.unpatchify(hidden_states, n_frames, height, width)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        return hidden_states

    def unpatchify(self, x, n_frames, height, width):
        t_patch, h_patch, w_patch = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=n_frames,
            N_h=height,
            N_w=width,
            T_p=t_patch,
            H_p=h_patch,
            W_p=w_patch,
            C_out=self.out_channels,
        )
        return x


EntryClass = LongCatVideoTransformer3DModel
