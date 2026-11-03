# SPDX-License-Identifier: Apache-2.0
# DM05 action expert and sampling math adapted from dexmal/opendm,
# revision fe1ed79a22d81cc1bfbeb2b4418bbb79fc75fc50 (Apache-2.0).

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma3Config, Gemma3TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3ForConditionalGeneration,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)

HISTORY_PAD_TOKEN_ID = 7


def linear_fp32(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(x.float(), layer.weight.float(), bias)


def time_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    fraction = torch.linspace(
        0.0, 1.0, dim // 2, device=time.device, dtype=torch.float32
    )
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = (1.0 / period * 2 * math.pi)[None, :] * time[:, None].float()
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def validate_action_config(
    action_config: Gemma3TextConfig, language_config: Gemma3TextConfig
) -> None:
    for name in (
        "num_hidden_layers",
        "layer_types",
        "max_position_embeddings",
        "rope_parameters",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        if getattr(action_config, name) != getattr(language_config, name):
            raise ValueError(
                f"DM05 action and language configurations differ at {name}"
            )


class DM05ActionExpert(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        attention_backend: str,
        precision_policy: str,
    ):
        super().__init__()
        if attention_backend not in ("eager", "sdpa"):
            raise ValueError("DM05 action attention must be 'eager' or 'sdpa'")
        self.config = config
        self.attention_backend = attention_backend
        self.precision_policy = precision_policy
        self.layers = nn.ModuleList(
            Gemma3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        hidden = config.hidden_size
        self.input_time_modulators = nn.ModuleList(
            nn.Linear(hidden, 3 * hidden) for _ in self.layers
        )
        self.mlp_time_modulators = nn.ModuleList(
            nn.Linear(hidden, 3 * hidden) for _ in self.layers
        )
        self.final_time_modulator = nn.Linear(hidden, 3 * hidden)

    def _adaptive_norm(self, norm, x, condition, modulator):
        variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        eps = getattr(norm, "eps", None)
        if eps is None:
            eps = norm.variance_epsilon
        normalized = x * torch.rsqrt(variance + eps)
        if self.precision_policy == "fp32_mixed":
            modulation = linear_fp32(condition, modulator)
        else:
            modulation = modulator(condition.to(dtype=modulator.weight.dtype))
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normalized = normalized * (1.0 + scale.float()) + shift.float()
        return normalized.to(x.dtype), gate.to(x.dtype)

    def forward(
        self,
        suffix_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_keys: tuple[torch.Tensor, ...],
        prefix_values: tuple[torch.Tensor, ...],
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = suffix_embeds
        for index, layer in enumerate(self.layers):
            prenorm, attention_gate = self._adaptive_norm(
                layer.input_layernorm,
                hidden,
                condition,
                self.input_time_modulators[index],
            )
            batch, length, _ = prenorm.shape
            shape = (batch, length, -1, layer.self_attn.head_dim)
            query = layer.self_attn.q_proj(prenorm).view(shape).transpose(1, 2)
            key = layer.self_attn.k_proj(prenorm).view(shape).transpose(1, 2)
            value = layer.self_attn.v_proj(prenorm).view(shape).transpose(1, 2)
            query = layer.self_attn.q_norm(query)
            key = layer.self_attn.k_norm(key)
            cos, sin = self.rotary_emb(
                query, position_ids, layer_type=self.config.layer_types[index]
            )
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = torch.cat((prefix_keys[index], key), dim=2)
            value = torch.cat((prefix_values[index], value), dim=2)
            # DynamicCache retains a shorter K/V tail on sliding-attention layers.
            mask = attention_mask[..., -key.shape[-2] :]
            if self.attention_backend == "sdpa":
                groups = layer.self_attn.num_key_value_groups
                attended = (
                    F.scaled_dot_product_attention(
                        query,
                        repeat_kv(key, groups),
                        repeat_kv(value, groups),
                        attn_mask=mask == 0,
                        scale=layer.self_attn.scaling,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                attended, _ = eager_attention_forward(
                    layer.self_attn,
                    query,
                    key,
                    value,
                    mask,
                    scaling=layer.self_attn.scaling,
                )
            attended = layer.self_attn.o_proj(attended.reshape(batch, length, -1))
            attended = layer.post_attention_layernorm(attended)
            residual = hidden + attended * attention_gate
            mlp_input, mlp_gate = self._adaptive_norm(
                layer.pre_feedforward_layernorm,
                residual,
                condition,
                self.mlp_time_modulators[index],
            )
            mlp_output = layer.post_feedforward_layernorm(layer.mlp(mlp_input))
            hidden = residual + mlp_output * mlp_gate
        return self._adaptive_norm(
            self.norm, hidden, condition, self.final_time_modulator
        )[0]


class DM05CoreModel(nn.Module):
    def __init__(
        self,
        checkpoint_config: dict[str, Any],
        *,
        precision_policy: str = "bf16_mixed",
        language_attention: str = "eager",
        vision_attention: str = "sdpa",
        action_attention: str = "eager",
    ):
        super().__init__()
        if precision_policy not in ("bf16_mixed", "fp32_mixed"):
            raise ValueError(f"Unsupported DM05 precision policy: {precision_policy}")
        vlm_config = Gemma3Config(**checkpoint_config["vlm_config"])
        action_config = Gemma3TextConfig(**checkpoint_config["action_config"])
        if precision_policy == "fp32_mixed":
            for config in (
                vlm_config,
                vlm_config.text_config,
                vlm_config.vision_config,
                action_config,
            ):
                config.dtype = torch.float32
        validate_action_config(action_config, vlm_config.text_config)
        vlm_config._attn_implementation = {
            "": language_attention,
            "text_config": language_attention,
            "vision_config": vision_attention,
        }
        vlm_config.text_config._attn_implementation = language_attention
        vlm_config.vision_config._attn_implementation = vision_attention
        action_config._attn_implementation = action_attention
        self.precision_policy = precision_policy
        self.vlm = Gemma3ForConditionalGeneration(vlm_config)
        self.action_expert = DM05ActionExpert(
            action_config,
            attention_backend=action_attention,
            precision_policy=precision_policy,
        )
        hidden = action_config.hidden_size
        action_dim = int(checkpoint_config["action_dim"])
        self.action_in_proj = nn.Linear(action_dim, hidden)
        self.action_out_proj = nn.Linear(hidden, action_dim)
        self.time_mlp_in = nn.Linear(hidden, hidden)
        self.time_mlp_out = nn.Linear(hidden, hidden)

    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> PrefixContext:
        model = self.vlm.model
        inputs_embeds = None
        has_history = (
            history_pixel_values is not None
            and history_mask is not None
            and history_pixel_values.shape[0] > 0
            and bool(history_mask.any().item())
        )
        # This prefix-only decision matches the reference, outside CUDA capture.
        if has_history or bool((input_ids == HISTORY_PAD_TOKEN_ID).any().item()):
            inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds.masked_fill(
                (input_ids == HISTORY_PAD_TOKEN_ID).unsqueeze(-1), 0
            )
            if has_history:
                features = model.get_image_features(
                    history_pixel_values, return_dict=True
                ).pooler_output
                spatial = math.isqrt(features.shape[1])
                grid = features.view(-1, spatial, spatial, features.shape[-1])
                grid = F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2), (4, 4))
                features = grid.permute(0, 2, 3, 1).reshape(-1, 16, features.shape[-1])
                inputs_embeds = inputs_embeds.masked_scatter(
                    history_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    features.to(inputs_embeds.dtype),
                )
        prefix_attention_mask = attention_mask.masked_fill(
            input_ids == HISTORY_PAD_TOKEN_ID, 0
        )
        positions = (prefix_attention_mask.cumsum(-1) - 1).clamp_min_(0)
        cache = DynamicCache(config=model.language_model.config)
        model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=prefix_attention_mask,
            position_ids=positions,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            past_key_values=cache,
            use_cache=True,
        )
        suffix_visible = (input_ids != model.language_model.padding_idx) & (
            input_ids != HISTORY_PAD_TOKEN_ID
        )
        return PrefixContext(
            past_key_values=VLADensePrefixCache(
                (layer.keys, layer.values, None) for layer in cache.layers
            ),
            prefix_pad_masks=suffix_visible,
            prefix_len=input_ids.shape[1],
            layout={"model_family": "dm05", "prefix_full_attention": False},
        )

    def velocity(
        self, prefix: PrefixContext, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        hidden = linear_fp32(x_t, self.action_in_proj).to(
            self.action_in_proj.weight.dtype
        )
        if self.precision_policy == "fp32_mixed":
            condition = linear_fp32(
                time_embedding(timestep, hidden.shape[-1]), self.time_mlp_in
            )
            condition = linear_fp32(F.silu(condition), self.time_mlp_out)
        else:
            condition = self.time_mlp_in(
                time_embedding(timestep, hidden.shape[-1]).to(
                    self.time_mlp_in.weight.dtype
                )
            )
            condition = self.time_mlp_out(F.silu(condition))
        condition = F.silu(condition)
        batch, horizon, _ = hidden.shape
        prefix_mask = torch.zeros(
            (batch, horizon, prefix.prefix_len),
            device=hidden.device,
            dtype=hidden.dtype,
        ).masked_fill(~prefix.prefix_pad_masks[:, None, :], -2.3819763e38)
        suffix_mask = torch.zeros(
            (batch, horizon, horizon), device=hidden.device, dtype=hidden.dtype
        )
        attention_mask = torch.cat((prefix_mask, suffix_mask), dim=-1).unsqueeze(1)
        positions = (
            prefix.prefix_pad_masks.sum(-1, keepdim=True)
            + torch.arange(horizon, device=hidden.device)[None, :]
        )
        keys = tuple(layer[0] for layer in prefix.past_key_values)
        values = tuple(layer[1] for layer in prefix.past_key_values)
        output = self.action_expert(
            hidden, positions, attention_mask, keys, values, condition
        )
        return linear_fp32(output, self.action_out_proj)
