# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenPI and LeRobot Pi0.5 PyTorch inference semantics.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLADensePrefixCache
from sglang.srt.layers.activation import GeluAndMul
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


@dataclass
class PiGemmaModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: object | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


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


def linear_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = module(x)
    return output[0] if isinstance(output, tuple) else output


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
        if cond is None:
            if self.dense is not None:
                return normed.type_as(x), None
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None

        modulation = self.dense(cond.to(dtype=self.dense.weight.dtype))
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class PiGemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig, *, tensor_parallel: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tensor_parallel = tensor_parallel
        if tensor_parallel:
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.intermediate_size] * 2,
                bias=False,
            )
            self.down_proj = RowParallelLinear(
                input_size=self.intermediate_size,
                output_size=self.hidden_size,
                bias=False,
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )
            self.gate_up_proj = None
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported PiGemma activation: {config.hidden_act}")
        self.act_fn = GeluAndMul(approximate="tanh")

    @property
    def projection_dtype(self) -> torch.dtype:
        if self.tensor_parallel:
            return self.gate_up_proj.weight.dtype
        return self.up_proj.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel:
            gate_up = linear_forward(self.gate_up_proj, x)
        else:
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return linear_forward(self.down_proj, self.act_fn(gate_up))


class PiGemmaRotaryEmbedding(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_parameters = config.rope_parameters
        if rope_parameters["rope_type"] != "default":
            raise ValueError(
                f"Unsupported PiGemma rope type: {rope_parameters['rope_type']}"
            )
        self.rope = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=self.max_seq_len_cached,
            base=rope_parameters["rope_theta"],
            is_neox_style=True,
            dtype=torch.float32,
        )

    def _ensure_cache(self, x: torch.Tensor) -> None:
        cache = self.rope.cos_sin_cache
        if cache.device == x.device and cache.dtype == torch.float:
            return
        if cache.dtype != torch.float:
            cache = self.rope._compute_cos_sin_cache()
        self.rope.cos_sin_cache = cache.to(device=x.device, dtype=torch.float)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache(x)
        flat_positions = position_ids.reshape(-1)
        cos_sin = self.rope.cos_sin_cache.index_select(0, flat_positions)
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        output_shape = (*position_ids.shape, self.head_dim)
        return cos.reshape(output_shape).to(x.dtype), sin.reshape(output_shape).to(
            x.dtype
        )


class PiGemmaAttention(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: int,
        *,
        tensor_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tensor_parallel = tensor_parallel
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        if tensor_parallel:
            self.total_num_heads = config.num_attention_heads
            self.total_num_key_value_heads = config.num_key_value_heads
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_key_value_heads,
                bias=config.attention_bias,
            )
            self.num_heads = self.qkv_proj.num_heads
            self.num_key_value_heads = self.qkv_proj.num_kv_heads
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_key_value_heads * self.head_dim
            self.o_proj = RowParallelLinear(
                input_size=self.total_num_heads * self.head_dim,
                output_size=config.hidden_size,
                bias=config.attention_bias,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
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
            self.qkv_proj = None
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_key_value_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_key_value_heads,
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
            and not tensor_parallel
            else None
        )

    @property
    def projection_dtype(self) -> torch.dtype:
        if self.tensor_parallel:
            return self.qkv_proj.weight.dtype
        return self.q_proj.weight.dtype

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        if self.tensor_parallel:
            qkv = linear_forward(self.qkv_proj, hidden_states)
            query_states, key_states, value_states = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        return (
            query_states.view(query_shape).transpose(1, 2),
            key_states.view(kv_shape).transpose(1, 2),
            value_states.view(kv_shape).transpose(1, 2),
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
        query_states, key_states, value_states = self.project_qkv(hidden_states)

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
                and getattr(past_key_values, "read_only", False)
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
                return linear_forward(self.o_proj, attn_output), None

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
        return linear_forward(self.o_proj, attn_output), None


class PiGemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: int,
        *,
        tensor_parallel: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PiGemmaAttention(
            config=config,
            layer_idx=layer_idx,
            tensor_parallel=tensor_parallel,
        )
        self.mlp = PiGemmaMLP(config, tensor_parallel=tensor_parallel)
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


class PiGemmaModel(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        *,
        tensor_parallel: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.tensor_parallel = tensor_parallel
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layerwise_cpu_offload_enabled = False
        self.layerwise_cpu_offload_device: torch.device | None = None
        self.layerwise_cpu_offload_empty_cache = True
        cond_dim = getattr(config, "adarms_cond_dim", None)
        self.layers = nn.ModuleList(
            [
                PiGemmaDecoderLayer(
                    config,
                    layer_idx,
                    tensor_parallel=tensor_parallel,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.rotary_emb = PiGemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

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
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> PiGemmaModelOutput:
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
            past_key_values = VLADensePrefixCache()

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

        causal_mask = attention_mask

        hidden_states = inputs_embeds
        if (
            len(self.layers) > 0
            and self.layers[0].self_attn.projection_dtype == torch.bfloat16
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

        return PiGemmaModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        *,
        tensor_parallel: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.model = PiGemmaModel(config, tensor_parallel=tensor_parallel)
        self.lm_head = None


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    def __init__(self, config, *, tensor_parallel: bool = False):
        super().__init__(config)
        del self.language_model
        self.language_model = PiGemmaModel(
            config.text_config,
            tensor_parallel=tensor_parallel,
        )


class PaliGemmaForConditionalGenerationWithPiGemma(nn.Module):
    def __init__(self, config, *, tensor_parallel: bool = False):
        super().__init__()
        self.config = config
        self.model = PaliGemmaModelWithPiGemma(
            config,
            tensor_parallel=tensor_parallel,
        )
        self.lm_head = None

    @property
    def language_model(self):
        return self.model.language_model


OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclass(frozen=True)
class GemmaVariantConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_variant_config(variant: str) -> GemmaVariantConfig:
    if variant == "gemma_300m":
        return GemmaVariantConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaVariantConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown Pi05 Gemma variant: {variant}")


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    fraction = torch.linspace(
        0.0,
        1.0,
        dimension // 2,
        dtype=torch.float64,
        device=time.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError("pad_masks and att_masks must be [batch, seq]")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def trim_trailing_padding_tokens(
    tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_len = int(token_masks.sum(dim=1).max().item())
    if token_len <= 0 or token_len >= tokens.shape[1]:
        return tokens, token_masks
    return tokens[:, :token_len], token_masks[:, :token_len]


def prepare_optional_full_attention_mask(
    att_2d_masks: torch.Tensor,
    *,
    full_attention: bool | None = None,
) -> torch.Tensor | None:
    if full_attention is None:
        full_attention = bool(att_2d_masks.all().item())
    if full_attention:
        return None
    masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


def siglip_vision_forward_with_openpi_dtype(
    self,
    pixel_values,
    interpolate_pos_encoding: bool | None = False,
    **kwargs,
) -> BaseModelOutputWithPooling:
    hidden_states = self.embeddings(
        pixel_values,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )
    if (
        len(self.encoder.layers) > 0
        and self.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    ):
        hidden_states = hidden_states.to(torch.bfloat16)

    encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
    last_hidden_state = encoder_outputs.last_hidden_state
    last_hidden_state = self.post_layernorm(last_hidden_state)
    pooler_output = self.head(last_hidden_state) if self.use_head else None
    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def compute_layer_complete(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    *,
    layers,
    rotary_emb,
):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(
            layer.input_layernorm, hidden_states, adarms_cond[i]
        )
        gates.append(gate)
        query_state, key_state, value_state = layer.self_attn.project_qkv(hidden_states)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = native_apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    paligemma_layer = layers[0]
    att_output = paligemma_layer.self_attn.attn(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        attn_mask=attention_mask,
    )
    batch_size = query_states.shape[0]
    head_dim = paligemma_layer.self_attn.head_dim
    hidden_size = paligemma_layer.self_attn.num_heads * head_dim
    att_output = att_output.reshape(batch_size, -1, hidden_size)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.projection_dtype:
            att_output = att_output.to(layer.self_attn.projection_dtype)
        out_emb = linear_forward(
            layer.self_attn.o_proj,
            att_output[:, start_pos:end_pos],
        )
        out_emb = gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(
            layer.post_attention_layernorm, out_emb, adarms_cond[i]
        )
        if layer.mlp.projection_dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config: GemmaVariantConfig,
        action_expert_config: GemmaVariantConfig,
        *,
        use_adarms: list[bool],
        precision: Literal["bfloat16", "float32"],
        image_size: int,
        runtime_role: Literal["all", "prefix", "action", "idle"] = "all",
        prefix_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.paligemma = None
        self.gemma_expert = None
        self.prefix_output_device: torch.device | None = None

        if runtime_role in ("all", "prefix"):
            vlm_config_hf = CONFIG_MAPPING["paligemma"]()
            vlm_config_hf._vocab_size = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.image_token_index = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.text_config.hidden_size = vlm_config.width
            vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
            vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
            vlm_config_hf.text_config.head_dim = vlm_config.head_dim
            vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
            vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
            vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
            vlm_config_hf.text_config.dtype = precision
            vlm_config_hf.text_config.vocab_size = PALIGEMMA_VOCAB_SIZE
            vlm_config_hf.text_config.use_adarms = use_adarms[0]
            vlm_config_hf.text_config.is_causal = False
            vlm_config_hf.text_config.use_bidirectional_attention = True
            vlm_config_hf.text_config.adarms_cond_dim = (
                vlm_config.width if use_adarms[0] else None
            )
            vlm_config_hf.vision_config.image_size = image_size
            vlm_config_hf.vision_config.intermediate_size = 4304
            vlm_config_hf.vision_config.projection_dim = 2048
            vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
            vlm_config_hf.vision_config.dtype = "float32"
            self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(
                config=vlm_config_hf,
                tensor_parallel=prefix_tensor_parallel,
            )
            vision_tower = self.paligemma.model.vision_tower
            vision_model = getattr(vision_tower, "vision_model", vision_tower)
            vision_model.forward = siglip_vision_forward_with_openpi_dtype.__get__(
                vision_model,
                type(vision_model),
            )
            self.paligemma.lm_head = None

        if runtime_role in ("all", "action"):
            action_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=action_expert_config.head_dim,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=PALIGEMMA_VOCAB_SIZE,
                hidden_activation="gelu_pytorch_tanh",
                dtype=precision,
                use_adarms=use_adarms[1],
                is_causal=False,
                use_bidirectional_attention=True,
                adarms_cond_dim=(action_expert_config.width if use_adarms[1] else None),
            )
            self.gemma_expert = PiGemmaForCausalLM(
                config=action_config_hf,
                tensor_parallel=False,
            )
            self.gemma_expert.lm_head = None
            self.gemma_expert.model.embed_tokens = None
        self.to_selected_dtype(precision)
        self.patch_native_attention_after_dtype_finalize()

    def patch_native_attention_after_dtype_finalize(self) -> None:
        if self.paligemma is None:
            return
        vision_tower = self.paligemma.model.vision_tower
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        patch_siglip_vision_attention_to_native(vision_model)

    def to_selected_dtype(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid Pi05 precision: {precision}")
        self.to(dtype=torch.bfloat16)
        keep_fp32 = [
            "vision_tower.embeddings.patch_embedding.weight",
            "vision_tower.embeddings.patch_embedding.bias",
            "vision_tower.embeddings.position_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def set_prefix_output_device(self, device: torch.device) -> None:
        self.prefix_output_device = torch.device(device)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        return next(module.parameters()).device

    def _prefix_transformer_device(self) -> torch.device:
        if self.prefix_output_device is not None:
            return self.prefix_output_device
        language_model = self.paligemma.model.language_model
        return self._module_device(language_model.layers[0])

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        out_dtype = image.dtype
        vision_device = self._module_device(self.paligemma.model.vision_tower)
        output_device = self._prefix_transformer_device()
        if image.device != vision_device or image.dtype != torch.float32:
            image = image.to(device=vision_device, dtype=torch.float32)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.device != output_device or features.dtype != out_dtype:
            features = features.to(device=output_device, dtype=out_dtype)
        return features

    def embed_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(images) == 1:
            return [self.embed_image(images[0])]
        out_dtype = images[0].dtype
        vision_device = self._module_device(self.paligemma.model.vision_tower)
        output_device = self._prefix_transformer_device()
        batch_sizes = [image.shape[0] for image in images]
        batched_images = torch.cat(
            [
                (
                    image.to(device=vision_device, dtype=torch.float32)
                    if image.device != vision_device or image.dtype != torch.float32
                    else image
                )
                for image in images
            ],
            dim=0,
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            image_outputs = self.paligemma.model.get_image_features(batched_images)
        features = image_outputs.pooler_output
        if features.device != output_device or features.dtype != out_dtype:
            features = features.to(device=output_device, dtype=out_dtype)
        return list(features.split(batch_sizes, dim=0))

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.paligemma.model.language_model.get_input_embeddings()
        embedding_device = self._module_device(embedding)
        output_device = self._prefix_transformer_device()
        if tokens.device != embedding_device:
            tokens = tokens.to(device=embedding_device)
        embeds = embedding(tokens)
        if embeds.device != output_device:
            embeds = embeds.to(device=output_device)
        return embeds

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        inputs_embeds: list[torch.FloatTensor | None],
        use_cache: bool | None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [
                prefix_output.last_hidden_state,
                None,
            ], prefix_output.past_key_values

        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            return [None, suffix_output.last_hidden_state], None

        paligemma_layers = self.paligemma.model.language_model.layers
        expert_layers = self.gemma_expert.model.layers
        rotary_emb = self.paligemma.model.language_model.rotary_emb
        for layers in zip(paligemma_layers, expert_layers, strict=True):
            inputs_embeds = compute_layer_complete(
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                layers=layers,
                rotary_emb=rotary_emb,
            )

        final_norms = (
            self.paligemma.model.language_model.norm,
            self.gemma_expert.model.norm,
        )
        outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(
                final_norms[i], hidden_states, adarms_cond[i]
            )
            outputs.append(out_emb)
        return outputs, None


class Pi05CoreModel(nn.Module):
    def __init__(
        self,
        config: Pi05PipelineConfig,
        runtime_role: Literal["all", "prefix", "action", "idle"] = "all",
        *,
        prefix_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        vlm_config = get_gemma_variant_config(config.paligemma_variant)
        action_config = get_gemma_variant_config(config.action_expert_variant)
        precision = (
            "bfloat16"
            if config.materialize_dtype in ("bf16", "bfloat16")
            else "float32"
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            vlm_config,
            action_config,
            use_adarms=[False, True],
            precision=precision,
            image_size=config.image_size[0],
            runtime_role=runtime_role,
            prefix_tensor_parallel=prefix_tensor_parallel,
        )
        if runtime_role in ("all", "action"):
            self.action_in_proj = nn.Linear(config.action_dim, action_config.width)
            self.action_out_proj = nn.Linear(action_config.width, config.action_dim)
            self.time_mlp_in = nn.Linear(action_config.width, action_config.width)
            self.time_mlp_out = nn.Linear(action_config.width, action_config.width)
            if precision == "bfloat16":
                for module in (
                    self.action_in_proj,
                    self.action_out_proj,
                    self.time_mlp_in,
                    self.time_mlp_out,
                ):
                    module.to(dtype=torch.float32)
        else:
            self.action_in_proj = None
            self.action_out_proj = None
            self.time_mlp_in = None
            self.time_mlp_out = None

    def retain_runtime_components(
        self,
        role: Literal["all", "prefix", "action", "idle"],
    ) -> None:
        if role == "all":
            return
        if role in ("prefix", "idle"):
            self.paligemma_with_expert.gemma_expert = None
            self.action_in_proj = None
            self.action_out_proj = None
            self.time_mlp_in = None
            self.time_mlp_out = None
        if role in ("action", "idle"):
            self.paligemma_with_expert.paligemma = None

    def prepare_attention_masks_4d(
        self,
        att_2d_masks: torch.Tensor,
        *,
        full_attention: bool | None = None,
    ) -> torch.Tensor | None:
        return prepare_optional_full_attention_mask(
            att_2d_masks,
            full_attention=full_attention,
        )

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        image_embs = self.paligemma_with_expert.embed_images(images)
        for image_emb, image_mask in zip(image_embs, image_masks, strict=True):
            batch_size, num_image_embs = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(batch_size, num_image_embs))
            att_masks += [0] * num_image_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(lang_emb)
        pad_masks.append(token_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_t = att_masks_t[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks_t

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.time_embedding_min_period,
            max_period=self.config.time_embedding_max_period,
        )
        action_emb = self.action_in_proj(
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        )
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = F.silu(time_emb)

        batch_size, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(
            batch_size,
            action_len,
            dtype=torch.bool,
            device=noisy_actions.device,
        )
        att_masks_t = torch.zeros(
            batch_size,
            action_len,
            dtype=action_emb.dtype,
            device=noisy_actions.device,
        )
        att_masks_t[:, 0] = 1
        return action_emb, pad_masks, att_masks_t, adarms_cond

    def _move_prefix_image_encoder_to_device(self, device: torch.device) -> None:
        paligemma = self.paligemma_with_expert.paligemma
        paligemma.model.vision_tower.to(device)
        paligemma.model.multi_modal_projector.to(device)

    def _prefix_language_phase_offload_layer_count(self) -> int:
        language_model = self.paligemma_with_expert.paligemma.model.language_model
        if self.config.offload_prefix_language_layers_after_prefix:
            return len(language_model.layers)
        return min(
            max(self.config.offload_prefix_language_layer_count_after_prefix, 0),
            len(language_model.layers),
        )

    def _move_prefix_language_layers_to_device(self, device: torch.device) -> None:
        language_model = self.paligemma_with_expert.paligemma.model.language_model
        layer_count = self._prefix_language_phase_offload_layer_count()
        for layer in language_model.layers[:layer_count]:
            layer.to(device)

    def _prepare_prefix_image_encoder_for_embed(self) -> None:
        if not self.config.offload_prefix_image_encoder_after_embed:
            return
        self._move_prefix_image_encoder_to_device(
            self.paligemma_with_expert._prefix_transformer_device()
        )

    def _offload_prefix_image_encoder_after_embed(self) -> None:
        if not self.config.offload_prefix_image_encoder_after_embed:
            return
        self._move_prefix_image_encoder_to_device(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_prefix_language_layers_for_forward(self) -> None:
        if (
            not self._prefix_language_phase_offload_layer_count()
            or self.config.offload_prefix_language_layers
        ):
            return
        self._move_prefix_language_layers_to_device(
            self.paligemma_with_expert._prefix_transformer_device()
        )

    def _offload_prefix_language_layers_after_prefix(self) -> None:
        if (
            not self._prefix_language_phase_offload_layer_count()
            or self.config.offload_prefix_language_layers
        ):
            return
        self._move_prefix_language_layers_to_device(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_masks: torch.Tensor,
        prefix_full_attention_hint: bool | None = None,
        tokens_trimmed: bool = False,
    ):
        if not tokens_trimmed:
            tokens, token_masks = trim_trailing_padding_tokens(tokens, token_masks)
        self._prepare_prefix_image_encoder_for_embed()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, image_masks, tokens, token_masks
        )
        self._offload_prefix_image_encoder_after_embed()
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_full_attention = bool(prefix_full_attention_hint)
        if prefix_full_attention:
            attention_mask = None
        else:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_full_attention = bool(prefix_att_2d_masks.all().item())
            attention_mask = self.prepare_attention_masks_4d(
                prefix_att_2d_masks,
                full_attention=prefix_full_attention,
            )
        self._prepare_prefix_language_layers_for_forward()
        with set_forward_context(current_timestep=0, attn_metadata=None):
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        self._offload_prefix_language_layers_after_prefix()
        return (
            past_key_values,
            prefix_pad_masks,
            prefix_position_ids,
            prefix_full_attention,
        )

    @torch.no_grad()
    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_full_attention: bool = False,
        *,
        action_position_offset: int = 0,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(x_t, timestep)
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        if prefix_full_attention:
            attention_mask = None
        else:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks],
                dim=2,
            )
            attention_mask = self.prepare_attention_masks_4d(full_att_2d_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets
            + action_position_offset
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=VLADensePrefixCache(
                    past_key_values,
                    read_only=True,
                ),
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        suffix_out = outputs_embeds[1][:, -x_t.shape[1] :]
        return self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        ).to(dtype=torch.float32)
