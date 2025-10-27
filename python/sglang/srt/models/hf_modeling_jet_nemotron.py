# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .configuration_jet_nemotron import JetNemotronConfig
from .jet_block import JetBlock
from .kv_cache import JetNemotronCache

logger = logging.get_logger(__name__)


class JetNemotronMLP(nn.Module):
    def __init__(self, config: JetNemotronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class JetNemotronAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: JetNemotronConfig,
        layer_idx: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def _get_target_length(
        self,
        sequence_length: int,
        past_key_values: JetNemotronCache,
    ):
        past_seen_tokens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        target_length = sequence_length + min(past_seen_tokens, self.sliding_window - 1)
        return target_length

    def _update_causal_mask_for_sliding_window(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: JetNemotronCache,
    ) -> torch.Tensor:

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = self._get_target_length(sequence_length, past_key_values)

        past_seen_tokens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + sequence_length,
            device=input_tensor.device,
        )

        if attention_mask is not None:
            # left padding
            assert attention_mask.dim() == 4, "Attention mask must be 4D"
            diagonal_attend_mask = attention_mask < -1
            diagonal_attend_mask = diagonal_attend_mask[:, :, :, -target_length:]
        else:
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            diagonal_attend_mask = diagonal_attend_mask[None, None, :, :]

        if past_key_values is None or target_length > self.sliding_window:
            # training mode or prefill mode when dealing with long prefix)
            sliding_attend_mask = torch.arange(
                past_seen_tokens + sequence_length, device=device
            )[-target_length:] <= (
                cache_position.reshape(-1, 1) - self.sliding_window
            )  # bs, sequence_length, target_length
            sliding_attend_mask = sliding_attend_mask[None, None, :, :]

            diagonal_attend_mask = diagonal_attend_mask | sliding_attend_mask

        # training
        causal_mask = torch.full(
            (1, 1, sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        causal_mask = causal_mask * diagonal_attend_mask
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[JetNemotronCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Fill zero if position embeddings not provided
        if position_embeddings is None:
            position_embeddings = (
                torch.zeros(
                    hidden_states.size(0),
                    self.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                torch.zeros(
                    hidden_states.size(0),
                    self.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
            )

        if (
            self.sliding_window is not None
            and self.config._attn_implementation != "flash_attention_2"
        ):
            attention_mask = self._update_causal_mask_for_sliding_window(
                attention_mask, hidden_states, past_key_value
            )

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            state = past_key_value.update(
                attn_state=(key_states, value_states),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
                cache_kwargs={"window_size": self.sliding_window},
            )
            key_states, value_states = state["attn_state"]

        fa2_sliding_window = None
        if self.sliding_window is not None:
            fa2_sliding_window = self.sliding_window - 1

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa":
                past_seen_tokens = (
                    past_key_value.get_seq_length() if past_key_value is not None else 0
                )
                if self.sliding_window is None:
                    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
                    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
                    # to infer the attention mask.
                    if AttentionMaskConverter._ignore_causal_mask_sdpa(
                        attention_mask,
                        inputs_embeds=hidden_states,
                        past_key_values_length=past_seen_tokens,
                        sliding_window=self.sliding_window,
                        is_training=self.training,
                    ):
                        attention_mask = None

            elif self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None:
                    assert len(attention_mask.shape) == 2, "Attention mask must be 2D"
                    attention_mask = attention_mask[:, -key_states.shape[2] :]
            else:
                raise ValueError(
                    f"Unsupported attention implementation: {self.config._attn_implementation}. "
                    "Supported implementations are: eager, sdpa, flash_attention_2."
                )

            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=fa2_sliding_window,  # main diff with Llama
            **kwargs,
        )

        if self.sliding_window is not None and past_key_value is not None:
            past_key_value.trim_attn_state(self.layer_idx, self.sliding_window)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class JetNemotronRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JetNemotronRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


EFFICIENT_ATTENTION_CLASSES = {
    "jet": JetBlock,
}


class JetNemotronDecoderLayer(nn.Module):
    def __init__(self, config: JetNemotronConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.layer_types[layer_idx] == "attn":
            self.self_attn = JetNemotronAttention(config, layer_idx)
        elif config.layer_types[layer_idx] == "swa":
            assert (
                config.efficient_attention_config is not None
            ), "Efficient attention config must be provided in JetNemotronConfig."
            assert (
                "swa" in config.efficient_attention_config
            ), "Sliding Window Attention is enabled but no `swa` configuration found in `efficient_attention_config`."
            self.self_attn = JetNemotronAttention(
                config,
                layer_idx,
                sliding_window=config.efficient_attention_config["swa"]["window_size"],
            )
        else:
            assert config.layer_types[layer_idx] in EFFICIENT_ATTENTION_CLASSES, (
                f"Layer type {config.layer_types[layer_idx]} not supported. Supported types are: "
                f"{['attn', 'swa'] + list(EFFICIENT_ATTENTION_CLASSES.keys())}"
            )
            self.self_attn = EFFICIENT_ATTENTION_CLASSES[config.layer_types[layer_idx]](
                config, config.layer_types[layer_idx], layer_idx
            )

        self.mlp = JetNemotronMLP(config)
        self.input_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class JetNemotronRotaryEmbedding(nn.Module):
    def __init__(self, config: JetNemotronConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
