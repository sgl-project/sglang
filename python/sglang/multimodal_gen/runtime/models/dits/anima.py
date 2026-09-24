# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Cosmos Predict2 image transformer used by Anima."""

import math

import torch
from diffusers.models.embeddings import Timesteps
from torch import nn

from sglang.multimodal_gen.configs.models.dits.anima import AnimaDiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT


def _is_anima_block(name, module):
    return is_module_list_entry_in(name, ("transformer_blocks",))


class AnimaPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        channels = config.in_channels + int(config.concat_padding_mask)
        self.proj = ReplicatedLinear(
            channels * math.prod(config.patch_size), config.hidden_size, bias=False
        )

    def forward(self, x):
        b, c, t, h, w = x.shape
        pt, ph, pw = self.patch_size
        x = x.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4).flatten(1, 3)
        return self.proj(x)[0]


class AnimaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.attention_head_dim
        self.dims = (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))
        self.scales = config.rope_scale
        self.patch_size = config.patch_size

    def forward(self, x):
        shape = [size // patch for size, patch in zip(x.shape[-3:], self.patch_size)]
        freqs = []
        for axis, (dim, scale) in enumerate(zip(self.dims, self.scales)):
            theta = 10000.0 * scale ** (dim / (dim - 2))
            inv = 1.0 / (
                theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
            )
            angles = torch.outer(
                torch.arange(shape[axis], device=x.device).float(), inv
            )
            view = [1, 1, 1, dim // 2]
            view[axis] = shape[axis]
            freqs.append(angles.view(view).expand(*shape, dim // 2))
        angles = torch.cat(freqs * 2, dim=-1).flatten(0, 2)
        return angles.cos(), angles.sin()


class AnimaTimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_proj = Timesteps(
            hidden_size, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.t_embedder = nn.Module()
        self.t_embedder.linear_1 = ReplicatedLinear(
            hidden_size, hidden_size, bias=False
        )
        self.t_embedder.linear_2 = ReplicatedLinear(
            hidden_size, 3 * hidden_size, bias=False
        )
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, timestep, dtype):
        x = self.time_proj(timestep).to(dtype)
        emb = self.t_embedder.linear_1(x)[0]
        emb = self.t_embedder.linear_2(torch.nn.functional.silu(emb))[0]
        return emb, self.norm(x)


class AnimaAdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, rank, gated=True):
        super().__init__()
        self.gated = gated
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_1 = ReplicatedLinear(hidden_size, rank, bias=False)
        self.linear_2 = ReplicatedLinear(
            rank, (3 if gated else 2) * hidden_size, bias=False
        )

    def forward(self, x, embedded_timestep, temb):
        modulation = self.linear_1(torch.nn.functional.silu(embedded_timestep))[0]
        modulation = self.linear_2(modulation)[0]
        modulation = modulation + temb[..., : modulation.shape[-1]]
        values = modulation.unsqueeze(1).chunk(3 if self.gated else 2, dim=-1)
        out = self.norm(x) * (1 + values[1]) + values[0]
        return (out, values[2]) if self.gated else out


class AnimaAttention(nn.Module):
    def __init__(self, config, cross_attention, quant_config, prefix):
        super().__init__()
        self.heads = config.num_attention_heads // get_tp_world_size()
        self.head_dim = config.attention_head_dim
        size = config.hidden_size
        context_size = config.text_embed_dim if cross_attention else size
        self.to_q = ColumnParallelLinear(
            size, size, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            context_size,
            size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            context_size,
            size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    size,
                    size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-6)
        self.attn = USPAttention(
            self.heads,
            self.head_dim,
            is_cross_attention=cross_attention,
            skip_sequence_parallel=cross_attention,
            prefix=prefix,
        )

    def forward(self, x, context=None, rope=None, attn_mask_meta=None):
        context = x if context is None else context
        q = self.to_q(x)[0].unflatten(-1, (self.heads, self.head_dim))
        k = self.to_k(context)[0].unflatten(-1, (self.heads, self.head_dim))
        v = self.to_v(context)[0].unflatten(-1, (self.heads, self.head_dim))
        q, k = self.norm_q(q), self.norm_k(k)
        if rope is not None:
            cos, sin = (r[None, :, None, :] for r in rope)
            # Cosmos uses split-half RoPE in fp32, then rounds once
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            q = (q.float() * cos + torch.cat((-q2, q1), -1).float() * sin).to(q.dtype)
            k = (k.float() * cos + torch.cat((-k2, k1), -1).float() * sin).to(k.dtype)
        out = self.attn(q, k, v, attn_mask_meta=attn_mask_meta).flatten(2)
        return self.to_out[0](out)[0]


class AnimaFeedForward(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        size = config.hidden_size
        inner = int(size * config.mlp_ratio)
        projection = nn.Module()
        projection.proj = ColumnParallelLinear(
            size,
            inner,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.net.0.proj",
        )
        self.net = nn.ModuleList(
            [
                projection,
                nn.Identity(),
                RowParallelLinear(
                    inner,
                    size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.net.2",
                ),
            ]
        )

    def forward(self, x):
        x = torch.nn.functional.gelu(self.net[0].proj(x)[0])
        return self.net[2](x)[0]


class AnimaTransformerBlock(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        size, rank = config.hidden_size, config.adaln_lora_dim
        self.norm1 = AnimaAdaLayerNorm(size, rank)
        self.attn1 = AnimaAttention(config, False, quant_config, f"{prefix}.attn1")
        self.norm2 = AnimaAdaLayerNorm(size, rank)
        self.attn2 = AnimaAttention(config, True, quant_config, f"{prefix}.attn2")
        self.norm3 = AnimaAdaLayerNorm(size, rank)
        self.ff = AnimaFeedForward(config, quant_config, f"{prefix}.ff")

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        embedded_timestep,
        temb,
        rope,
        attn_mask_meta=None,
    ):
        x, gate = self.norm1(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn1(
            x, rope=rope, attn_mask_meta=attn_mask_meta
        )
        x, gate = self.norm2(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn2(
            x, context=encoder_hidden_states
        )
        x, gate = self.norm3(hidden_states, embedded_timestep, temb)
        return hidden_states + gate * self.ff(x)


class AnimaTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _aliases = ["CosmosTransformer3DModel"]
    _fsdp_shard_conditions = [_is_anima_block]
    _compile_conditions = [_is_anima_block]
    layer_names = ["transformer_blocks"]
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    def __init__(self, config: AnimaDiTConfig, hf_config, quant_config=None, **kwargs):
        super().__init__(config=config, hf_config=hf_config)
        arch = self.config
        if (
            arch.extra_pos_embed_type
            or arch.use_crossattn_projection
            or arch.img_context_dim_in
        ):
            raise ValueError(
                "Only Anima's Cosmos image transformer architecture is supported"
            )
        if arch.num_attention_heads % get_tp_world_size():
            raise ValueError("Anima attention heads must be divisible by TP size")
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels
        self.patch_embed = AnimaPatchEmbed(arch)
        self.rope = AnimaRotaryEmbedding(arch)
        self.time_embed = AnimaTimeEmbedding(arch.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                AnimaTransformerBlock(arch, quant_config, f"transformer_blocks.{i}")
                for i in range(arch.num_layers)
            ]
        )
        self.norm_out = AnimaAdaLayerNorm(
            arch.hidden_size, arch.adaln_lora_dim, gated=False
        )
        self.proj_out = ReplicatedLinear(
            arch.hidden_size, math.prod(arch.patch_size) * arch.out_channels, bias=False
        )
        self.__post_init__()

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        b, _, t, h, w = hidden_states.shape
        pt, ph, pw = self.config.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(
                "Anima latent dimensions must be divisible by the patch size"
            )
        rope = self.rope(hidden_states)
        if self.config.concat_padding_mask:
            hidden_states = torch.cat(
                [hidden_states, hidden_states.new_zeros(b, 1, t, h, w)], dim=1
            )
        hidden_states = self.patch_embed(hidden_states)
        hidden_states, shard = shard_seq(hidden_states)
        rope = tuple(shard_like(r, shard, dim=0) for r in rope)
        attn_mask_meta = tail_attn_meta(shard, b, hidden_states.device)
        temb, embedded_timestep = self.time_embed(
            timestep.to(hidden_states.dtype) / 1000, hidden_states.dtype
        )
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb,
                rope,
                attn_mask_meta,
            )
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)[0]
        hidden_states = gather_seq(hidden_states, shard.orig_len)
        # checkpoint output ordering is (ph, pw, pt, c), unlike the input patches
        hidden_states = hidden_states.reshape(
            b, t // pt, h // ph, w // pw, ph, pw, pt, -1
        )
        return hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5).reshape(
            b, self.config.out_channels, t, h, w
        )


EntryClass = AnimaTransformer3DModel
