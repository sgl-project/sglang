# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.visual_embedding import Timesteps
from sglang.multimodal_gen.runtime.managers.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        if hidden_dtype is not None:
            timesteps_proj = timesteps_proj.to(dtype=hidden_dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class SanaAdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.emb = SanaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(self, timestep, hidden_dtype=None):
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        out = self.linear(self.silu(embedded_timestep))
        return out, embedded_timestep


class SanaModulatedNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x, temb, scale_shift_table):
        x = self.norm(x)
        shift, scale = (scale_shift_table[None] + temb[:, None]).chunk(2, dim=1)
        x = x * (1 + scale) + shift
        return x


class GLUMBConv(nn.Module):
    """Gated Linear Unit with Multi-Branch Convolution."""

    def __init__(self, in_channels, out_channels, expand_ratio=2.5):
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            1,
            1,
            groups=hidden_channels * 2,
        )
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)
        return hidden_states


class SanaLinearAttention(nn.Module):
    """Linear attention with O(N*D^2) complexity instead of O(N^2*D)."""

    def __init__(self, query_dim, num_heads, head_dim, bias=False):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )

    def forward(self, hidden_states):
        B, S, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        query = F.relu(query)
        key = F.relu(key)

        kv = torch.matmul(key.transpose(-2, -1), value)  # (B, H, D, D)
        qkv = torch.matmul(query, kv)  # (B, H, S, D)

        key_sum = key.sum(dim=-2, keepdim=True)  # (B, H, 1, D)
        normalizer = torch.matmul(query, key_sum.transpose(-2, -1)).clamp(min=1e-6)
        hidden_states = qkv / normalizer

        hidden_states = hidden_states.transpose(1, 2).reshape(B, S, -1)
        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


class SanaCrossAttention(nn.Module):
    def __init__(self, query_dim, cross_attention_dim, num_heads, head_dim, bias=False):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )

    def forward(
        self, hidden_states, encoder_hidden_states, encoder_attention_mask=None
    ):
        B, S, _ = hidden_states.shape
        T = encoder_hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if encoder_attention_mask is not None:
            attn_mask = encoder_attention_mask.bool()
            attn_mask = attn_mask[:, None, None, :].expand(B, self.num_heads, S, T)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(B, S, -1)
        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


class SanaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        num_cross_attention_heads,
        cross_attention_head_dim,
        cross_attention_dim,
        mlp_ratio,
        norm_eps,
        attention_bias=False,
    ):
        super().__init__()

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaLinearAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn2 = SanaCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
            bias=True,
        )

        self.ff = GLUMBConv(in_channels=dim, out_channels=dim, expand_ratio=mlp_ratio)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        height,
        width,
        encoder_attention_mask=None,
    ):
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden)
        hidden_states = hidden_states + gate_msa * attn_output

        attn_output = self.attn2(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = hidden_states + attn_output

        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp) + shift_mlp
        norm_hidden = norm_hidden.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class SanaTransformer2DModel(CachableDiT, OffloadableDiTMixin):

    _fsdp_shard_conditions = [
        lambda n, m: isinstance(m, SanaTransformerBlock),
    ]
    _compile_conditions = [
        lambda n, m: isinstance(m, SanaTransformerBlock),
    ]
    param_names_mapping = SanaConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(self, config: SanaConfig, hf_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config or {}, **kwargs)

        arch = config.arch_config
        self.out_channels = arch.out_channels
        self.patch_size = arch.patch_size
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim

        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents

        self.patch_embed = nn.ModuleDict(
            {
                "proj": nn.Conv2d(
                    arch.in_channels,
                    self.inner_dim,
                    kernel_size=arch.patch_size,
                    stride=arch.patch_size,
                    bias=True,
                ),
            }
        )
        self.time_embed = SanaAdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=arch.caption_channels,
            hidden_size=self.inner_dim,
        )

        self.caption_norm = RMSNorm(self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=arch.num_attention_heads,
                    attention_head_dim=arch.attention_head_dim,
                    num_cross_attention_heads=arch.num_cross_attention_heads,
                    cross_attention_head_dim=arch.cross_attention_head_dim,
                    cross_attention_dim=arch.cross_attention_dim,
                    mlp_ratio=arch.mlp_ratio,
                    norm_eps=arch.norm_eps,
                    attention_bias=False,
                )
                for _ in range(arch.num_layers)
            ]
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )

        self.norm_out = SanaModulatedNorm(self.inner_dim, eps=arch.norm_eps)

        self.proj_out = nn.Linear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
        )

        self.layer_names = ["transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        # Input validation - fail fast
        if encoder_hidden_states is None:
            raise ValueError("SANA forward pass requires encoder_hidden_states")

        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.patch_embed["proj"](hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        timestep_emb, embedded_timestep = self.time_embed(
            timestep, hidden_dtype=hidden_states.dtype
        )

        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        if encoder_hidden_states.shape[0] != batch_size:
            encoder_hidden_states = encoder_hidden_states.expand(
                batch_size, -1, -1
            ).contiguous()
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        if (
            encoder_attention_mask is not None
            and encoder_attention_mask.shape[0] != batch_size
        ):
            encoder_attention_mask = encoder_attention_mask.expand(
                batch_size, -1
            ).contiguous()

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_emb,
                post_patch_height,
                post_patch_width,
                encoder_attention_mask=encoder_attention_mask,
            )
        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, p, p, self.out_channels
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(
            batch_size, self.out_channels, height, width
        )

        return hidden_states


EntryClass = SanaTransformer2DModel
