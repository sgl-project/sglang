# SPDX-License-Identifier: Apache-2.0
# Adapted from LeRobot's PiGemma modules.

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import (
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
)

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.layers.rotary_embedding import (
    apply_rotary_pos_emb as native_apply_rotary_pos_emb,
)


def config_compute_dtype(config: GemmaConfig) -> torch.dtype | None:
    dtype = getattr(config, "dtype", None)
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    return None


def gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(
    layernorm: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is not None:
        return layernorm(x, cond=cond)
    return layernorm(x)


def _use_ulysses_action_attention(num_heads: int) -> bool:
    if not model_parallel_is_initialized():
        return False
    try:
        sp_world_size = get_sequence_parallel_world_size()
        ulysses_world_size = get_ulysses_parallel_world_size()
        ring_world_size = get_ring_parallel_world_size()
    except AssertionError:
        return False
    return (
        sp_world_size > 1
        and ulysses_world_size > 1
        and ring_world_size == 1
        and num_heads % sp_world_size == 0
    )


class Pi05SiglipAttention(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.scale = getattr(attention, "scale", self.head_dim**-0.5)
        self.dropout = getattr(attention, "dropout", 0.0)
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.out_proj = attention.out_proj
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=self.scale,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.FA2,
                AttentionBackendEnum.TORCH_SDPA,
            },
            compute_dtype=self.q_proj.weight.dtype,
            allow_cudnn_sdp=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).view(
            *input_shape,
            self.num_heads,
            self.head_dim,
        )
        key_states = self.k_proj(hidden_states).view(
            *input_shape,
            self.num_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).view(
            *input_shape,
            self.num_heads,
            self.head_dim,
        )
        attn_output = self.attn(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )
        attn_output = attn_output.reshape(*input_shape, self.embed_dim).contiguous()
        return self.out_proj(attn_output), None


def patch_siglip_vision_attention_to_native(vision_model: nn.Module) -> None:
    for layer in vision_model.encoder.layers:
        if isinstance(layer.self_attn, Pi05SiglipAttention):
            continue
        layer.self_attn = Pi05SiglipAttention(layer.self_attn)


class PiGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None
        else:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None

        modulation = self.dense(cond.to(dtype=self.dense.weight.dtype))
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class PiGemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn = LocalAttention(
            num_heads=config.num_attention_heads,
            head_size=self.head_dim,
            num_kv_heads=config.num_key_value_heads,
            softmax_scale=self.scaling,
            causal=self.is_causal,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.FA2,
                AttentionBackendEnum.TORCH_SDPA,
            },
            compute_dtype=config_compute_dtype(config),
            allow_cudnn_sdp=True,
        )
        self.sp_attn = (
            USPAttention(
                num_heads=config.num_attention_heads,
                head_size=self.head_dim,
                num_kv_heads=config.num_attention_heads,
                softmax_scale=self.scaling,
                causal=self.is_causal,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.FA2,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                allow_cudnn_sdp=True,
            )
            if _use_ulysses_action_attention(config.num_attention_heads)
            else None
        )

    def _repeat_kv_for_sequence_parallel(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return states
        return states.repeat_interleave(self.num_key_value_groups, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = native_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

        if past_key_values is not None:
            if (
                self.sp_attn is not None
                and hasattr(past_key_values, "get_prefix")
                and attention_mask is None
            ):
                prefix_key_states, prefix_value_states = past_key_values.get_prefix(
                    self.layer_idx
                )
                attn_output = self.sp_attn.forward_with_replicated_kv_prefix(
                    query_states.transpose(1, 2),
                    self._repeat_kv_for_sequence_parallel(
                        prefix_key_states.transpose(1, 2)
                    ),
                    self._repeat_kv_for_sequence_parallel(
                        prefix_value_states.transpose(1, 2)
                    ),
                    self._repeat_kv_for_sequence_parallel(key_states.transpose(1, 2)),
                    self._repeat_kv_for_sequence_parallel(value_states.transpose(1, 2)),
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), None

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attn_output = self.attn(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_mask=attention_mask,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


class PiGemmaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PiGemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        cond_dim = (
            getattr(config, "adarms_cond_dim", None)
            if getattr(config, "use_adarms", False)
            else None
        )
        self.input_layernorm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.post_attention_layernorm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(
            hidden_states, cond=adarms_cond
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = gated_residual(residual, hidden_states, gate)
        return hidden_states


class PiGemmaModel(GemmaModel):
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        del self.layers
        del self.norm
        self.layerwise_cpu_offload_enabled = False
        self.layerwise_cpu_offload_device: torch.device | None = None
        self.layerwise_cpu_offload_empty_cache = True
        cond_dim = getattr(config, "adarms_cond_dim", None)
        self.layers = nn.ModuleList(
            [
                PiGemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )

    def configure_layerwise_cpu_offload(
        self,
        *,
        compute_device: torch.device,
        empty_cache: bool = True,
    ) -> None:
        self.layerwise_cpu_offload_enabled = True
        self.layerwise_cpu_offload_device = torch.device(compute_device)
        self.layerwise_cpu_offload_empty_cache = empty_cache

    def _maybe_move_inputs_for_layerwise_offload(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        adarms_cond: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.LongTensor | None,
        torch.LongTensor | None,
        torch.Tensor | None,
    ]:
        if not self.layerwise_cpu_offload_enabled:
            return (
                inputs_embeds,
                attention_mask,
                position_ids,
                cache_position,
                adarms_cond,
            )
        if self.layerwise_cpu_offload_device is None:
            raise RuntimeError("PiGemma layerwise CPU offload compute device is unset")

        device = self.layerwise_cpu_offload_device
        if inputs_embeds.device != device:
            inputs_embeds = inputs_embeds.to(device=device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device=device)
        if position_ids is not None and position_ids.device != device:
            position_ids = position_ids.to(device=device)
        if cache_position is not None and cache_position.device != device:
            cache_position = cache_position.to(device=device)
        if adarms_cond is not None and adarms_cond.device != device:
            adarms_cond = adarms_cond.to(device=device)
        return inputs_embeds, attention_mask, position_ids, cache_position, adarms_cond

    def _layer_to_compute_device(self, decoder_layer: nn.Module) -> None:
        if not self.layerwise_cpu_offload_enabled:
            return
        decoder_layer.to(self.layerwise_cpu_offload_device)

    def _layer_to_cpu_after_compute(self, decoder_layer: nn.Module) -> None:
        if not self.layerwise_cpu_offload_enabled:
            return
        decoder_layer.to("cpu")
        device = self.layerwise_cpu_offload_device
        if (
            self.layerwise_cpu_offload_empty_cache
            and device is not None
            and device.type == "cuda"
        ):
            torch.cuda.empty_cache()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        (
            inputs_embeds,
            attention_mask,
            position_ids,
            cache_position,
            adarms_cond,
        ) = self._maybe_move_inputs_for_layerwise_offload(
            inputs_embeds,
            attention_mask,
            position_ids,
            cache_position,
            adarms_cond,
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None and not getattr(self.config, "is_causal", True):
            causal_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        if (
            len(self.layers) > 0
            and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            self._layer_to_compute_device(decoder_layer)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )
            hidden_states = layer_outputs
            self._layer_to_cpu_after_compute(decoder_layer)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        del self.model
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    def __init__(self, config):
        super().__init__(config)
        del self.language_model
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        del self.model
        self.model = PaliGemmaModelWithPiGemma(config)

    @property
    def language_model(self):
        return self.model.language_model
