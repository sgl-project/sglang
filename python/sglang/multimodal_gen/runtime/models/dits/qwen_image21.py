# Copyright 2026 Qwen-Image Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class QwenImage21Layout:
    text_indices: torch.Tensor
    image_indices: torch.Tensor
    prefix_rope: torch.Tensor
    target_rope: torch.Tensor
    segments: tuple[tuple[int, int, bool], ...]


def build_layout(image_slots, image_shapes, axes_dims, device):
    """Expand each condition-image slot to its complete latent grid before denoising."""
    indices, image_indices, positions, segments = [], [], [], []
    cursor = position = image_index = 0
    for text_index, is_image in enumerate(image_slots):
        if not is_image:
            indices.append(text_index)
            positions.append((position, position, position))
            position += 1
            continue
        start = len(indices)
        if start > cursor:
            segments.append((cursor, start, False))
        _, height, width = image_shapes[image_index]
        for h in range(-(height - height // 2), height // 2):
            for w in range(-(width - width // 2), width // 2):
                indices.append(text_index)
                image_indices.append(len(indices) - 1)
                positions.append((position, h, w))
        segments.append((start, len(indices), True))
        cursor = len(indices)
        position += max(height, width)
        image_index += 1
    if image_index != len(image_shapes) - 1:
        raise ValueError("condition-image slots do not match image_shapes")
    if len(indices) > cursor:
        segments.append((cursor, len(indices), False))
    prefix_len = len(indices)
    _, height, width = image_shapes[-1]
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    pos = torch.tensor(positions, device=device, dtype=torch.float32)
    angles = torch.cat(
        [
            pos[:, axis : axis + 1]
            * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))
            for axis, dim in enumerate(axes_dims)
        ],
        dim=-1,
    )
    rope = torch.polar(torch.ones_like(angles), angles)
    return QwenImage21Layout(
        torch.tensor(indices, device=device, dtype=torch.long),
        torch.tensor(image_indices, device=device, dtype=torch.long),
        rope[:prefix_len],
        rope[prefix_len:],
        tuple(segments),
    )


def apply_rope(x, rope):
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim, eps, zero_centered=False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(dim) if zero_centered else torch.ones(dim)
        )
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, x):
        scale = self.weight.float() + int(self.zero_centered)
        value = x.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
            * scale
        ).to(x.dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = QwenImage21RMSNorm(context_dim, eps, zero_centered=True)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(
            nn.functional.gelu(self.in_layer(self.text_norm(x)), approximate="tanh")
        )


class QwenImage21TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(256, dim, bias=False)
        self.timestep_embedder.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t, dtype):
        freq = torch.exp(
            -math.log(10000) * torch.arange(128, device=t.device).float() / 128
        )
        angles = t.float()[:, None] * 1000 * freq
        x = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        return self.timestep_embedder.linear_2(
            nn.functional.silu(self.timestep_embedder.linear_1(x))
        )


class QwenImage21FeedForward(nn.Module):
    def __init__(self, dim, ratio, quant_config, prefix):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.gate_layer = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_layer",
        )
        self.out = RowParallelLinear(
            dim * ratio,
            dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

    def forward(self, x):
        return self.out(nn.functional.silu(self.gate_layer(x)[0]) * self.proj(x)[0])[0]


class QwenImage21Attention(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        dim = ac.hidden_size
        self.heads = ac.num_attention_heads // get_tp_world_size()
        self.head_dim = ac.attention_head_dim
        self.to_q = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_v"
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.norm_q = QwenImage21RMSNorm(self.head_dim, ac.eps)
        self.norm_k = QwenImage21RMSNorm(self.head_dim, ac.eps)
        backends = QwenImage21Transformer2DModel._supported_attention_backends
        self.local_attn = LocalAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )
        self.target_attn = USPAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )

    def qkv(self, x, rope):
        q = self.to_q(x)[0].unflatten(-1, (self.heads, self.head_dim))
        k = self.to_k(x)[0].unflatten(-1, (self.heads, self.head_dim))
        v = self.to_v(x)[0].unflatten(-1, (self.heads, self.head_dim))
        return apply_rope(self.norm_q(q), rope), apply_rope(self.norm_k(k), rope), v

    def forward(self, x, rope, prefix, prefix_rope, segments, cache):
        if cache:
            kp, vp = cache["key"], cache["value"]
            prefix_output = None
        else:
            qp, kp, vp = self.qkv(prefix, prefix_rope)
            outputs = []
            # text runs are causal; image blocks see the entire preceding sequence and themselves
            for start, end, is_image in segments:
                mask = None
                if not is_image:
                    mask = (
                        torch.arange(end, device=x.device)[None, :]
                        <= torch.arange(start, end, device=x.device)[:, None]
                    )
                    mask = mask[None, None]
                outputs.append(
                    self.local_attn(
                        qp[:, start:end], kp[:, :end], vp[:, :end], attn_mask=mask
                    )
                )
            prefix_output = self.to_out[0](torch.cat(outputs, dim=1).flatten(2))[0]
            if cache is not None:
                cache.update(key=kp, value=vp)
        q, k, v = self.qkv(x, rope)
        out = self.target_attn.forward_with_replicated_kv_prefix(q, kp, vp, k, v)
        return self.to_out[0](out.flatten(2))[0], prefix_output


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.img_norm2 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.attn = QwenImage21Attention(ac, quant_config, f"{prefix}.attn")
        self.img_mlp = QwenImage21FeedForward(
            ac.hidden_size, ac.mlp_ratio, quant_config, f"{prefix}.img_mlp"
        )

    def forward(
        self, hidden_states, modulation, prefix, prefix_modulation, layout, rope, cache
    ):
        scale1, gate1, scale2, gate2 = modulation[:, None].chunk(4, dim=-1)
        ps1, pg1, ps2, pg2 = prefix_modulation[:, None].chunk(4, dim=-1)
        p = None if cache else self.img_norm1(prefix) * (1 + ps1)
        attention, prefix_attention = self.attn(
            self.img_norm1(hidden_states) * (1 + scale1),
            rope,
            p,
            layout.prefix_rope,
            layout.segments,
            cache,
        )
        hidden_states = hidden_states + gate1.tanh() * attention
        hidden_states = hidden_states + gate2.tanh() * self.img_mlp(
            self.img_norm2(hidden_states) * (1 + scale2)
        )
        if prefix_attention is not None:
            prefix = prefix + pg1.tanh() * prefix_attention
            prefix = prefix + pg2.tanh() * self.img_mlp(
                self.img_norm2(prefix) * (1 + ps2)
            )
        return hidden_states, prefix


class QwenImage21OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x, temb):
        return self.norm(x) * (1 + self.linear(nn.functional.silu(temb))[:, None])


class QwenImage21Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.TORCH_SDPA,
    }
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, QwenImage21TransformerBlock)
    ]
    _compile_conditions = _fsdp_shard_conditions
    layer_names = ["transformer_blocks"]
    param_names_mapping = {}

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config, **kwargs)
        ac = self.config
        if ac.patch_size != 1 or not ac.causal_condition or not ac.causal_block:
            raise ValueError(
                "Qwen-Image 2.1 requires patch_size=1, causal_condition=True and causal_block=True"
            )
        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.in_channels
        self.img_in = nn.Linear(ac.in_channels, ac.hidden_size, bias=False)
        self.txt_in = QwenImage21TextProjection(
            ac.context_in_dim, ac.hidden_size, ac.eps
        )
        self.time_text_embed = QwenImage21TimeEmbedding(ac.hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(ac.hidden_size, ac.hidden_size * 4, bias=False)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(ac, quant_config, f"transformer_blocks.{i}")
                for i in range(ac.num_layers)
            ]
        )
        self.norm_out = QwenImage21OutputNorm(ac.hidden_size, ac.eps)
        self.proj_out = nn.Linear(ac.hidden_size, ac.out_channels, bias=False)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        layouts,
        condition_latents=None,
        prefix_caches=None,
        **kwargs,
    ):
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        sp = get_sp_world_size()
        target_len = hidden_states.shape[1]
        if target_len % sp:
            raise ValueError(
                f"target token count {target_len} must be divisible by SP degree {sp}"
            )
        local_len = target_len // sp
        rank = get_sp_parallel_rank()
        start, end = rank * local_len, (rank + 1) * local_len
        images = self.img_in(hidden_states[:, start:end])
        temb = self.time_text_embed((timestep.to(images.dtype) / 1000), images.dtype)
        zero_temb = self.time_text_embed(
            timestep.new_zeros(1).to(images.dtype), images.dtype
        )
        modulation, prefix_modulation = self.modulation(temb), self.modulation(
            zero_temb
        )
        outputs = []
        for sample, layout in enumerate(layouts):
            caches = (
                prefix_caches[sample]
                if prefix_caches is not None
                else [None] * len(self.transformer_blocks)
            )
            prefix = None
            if not caches[0]:
                prefix = self.txt_in(
                    encoder_hidden_states[sample : sample + 1]
                ).index_select(1, layout.text_indices)
                if condition_latents is not None:
                    prefix[:, layout.image_indices] = self.img_in(
                        condition_latents[sample : sample + 1]
                    )
            x = images[sample : sample + 1]
            for i, block in enumerate(self.transformer_blocks):
                x, prefix = block(
                    x,
                    modulation[sample : sample + 1],
                    prefix,
                    prefix_modulation,
                    layout,
                    layout.target_rope[start:end],
                    caches[i],
                )
            outputs.append(self.proj_out(self.norm_out(x, temb[sample : sample + 1])))
        output = torch.cat(outputs)
        if sp > 1:
            output = sequence_model_parallel_all_gather(output, dim=1)
        return output


EntryClass = QwenImage21Transformer2DModel
