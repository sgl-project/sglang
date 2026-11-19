# coding=utf-8
# Copyright 2023-2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dory (NVIDIA looped normalized-GPT): nGPT residual stream with a looped recurrent block."""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.dory import DoryConfig
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.auto_loader import LLAMA_STACKED_MAPPING
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel

_EPS = 1e-6


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """nGPT normalize: x * rsqrt(sum(x^2) + eps), accumulated in fp32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + _EPS)
    return y.to(x.dtype)


class DoryAttention(nn.Module):
    """GQA with post-RoPE nGPT qk-norm, sigmoid output gate and one KV slot per loop."""

    def __init__(
        self,
        config: DoryConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.layer_id = layer_id
        std = config.initializer_range

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("g_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self._sqk_factor = self.head_dim**0.5 / std
        gate_init = config.attention_output_gate_logit_scale_init
        if gate_init is None:
            gate_init = config.hidden_size**0.5
        self._s_gate_factor = gate_init**0.5 / std

        layer_type = config.block_types[layer_id]
        rope = config.rope_parameters[layer_type]
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * rope.get("partial_rotary_factor", 1.0)),
            max_position=config.max_position_embeddings,
            base=int(rope.get("rope_theta", 10000)),
            is_neox_style=True,
        )
        window = config.sliding_window if layer_type == "sliding_attention" else -1
        self.attn = nn.ModuleList(
            [
                RadixAttention(
                    self.num_heads,
                    self.head_dim,
                    1.0 / math.sqrt(self.head_dim),
                    num_kv_heads=self.num_kv_heads,
                    layer_id=config.kv_slot_id(layer_id, loop_idx),
                    sliding_window_size=window,
                    quant_config=quant_config,
                    prefix=add_prefix("attn", prefix),
                )
                for loop_idx in range(config.kv_loops(layer_id))
            ]
        )

    def _norm_q(self, q: torch.Tensor) -> torch.Tensor:
        rows = q.shape[0]
        heads = q.reshape(rows, self.num_heads, self.head_dim)
        sqk = (self.sqk.float() * self._sqk_factor).view(
            1, self.num_heads, self.head_dim
        )
        normed = sqk.square() * justnorm(heads).float()
        return normed.to(q.dtype).reshape(rows, self.q_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int = 0,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        gate, _ = self.g_proj(hidden_states)

        k = justnorm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.kv_size
        )
        out = self.attn[loop_idx](self._norm_q(q), k, v, forward_batch)

        gate_scale = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * gate_scale)).to(out.dtype)
        out, _ = self.o_proj(justnorm(out))
        return out


class DoryMLP(nn.Module):
    """SwiGLU with nGPT suv scaling and a justnorm before down_proj."""

    def __init__(
        self,
        config: DoryConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.suv_gate = nn.Parameter(torch.empty(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.empty(config.intermediate_size))
        self._suv_factor = config.hidden_size**0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = (gate.float() * (self.suv_gate.float() * self._suv_factor)).to(
            gate_up.dtype
        )
        up = (up.float() * (self.suv_up.float() * self._suv_factor)).to(gate_up.dtype)
        act = (F.silu(gate.float()) * up.float()).to(gate_up.dtype)
        out, _ = self.down_proj(justnorm(act))
        return out


class DoryDecoderLayer(nn.Module):
    """One physical layer: a single sublayer (attention or MLP) plus a slerp residual."""

    def __init__(
        self,
        config: DoryConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_attention = config.block_types[layer_id] != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        mixer_prefix = add_prefix("mixer", prefix)
        if self.is_attention:
            self.mixer = DoryAttention(
                config, layer_id, quant_config=quant_config, prefix=mixer_prefix
            )
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config, quant_config=quant_config, prefix=mixer_prefix)
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    def _residual(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        alpha = self.attn_alpha if self.is_attention else self.mlp_alpha
        base = justnorm(residual).float()
        target = justnorm(update).float()
        step = (alpha.float() * self._alpha_scale).abs()
        return justnorm(base + step * (target - base)).to(residual.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_attention:
            update = self.mixer(positions, hidden_states, forward_batch, loop_idx)
        else:
            update = self.mixer(hidden_states)
        return self._residual(hidden_states, update)


class DoryModel(nn.Module):
    """Input block, then n_recurrent_loops passes over the shared recurrent block, then output block."""

    def __init__(
        self,
        config: DoryConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embeddings", prefix),
        )
        self.layers = nn.ModuleList(
            [
                DoryDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    @property
    def start_layer(self) -> int:
        return 0

    @property
    def end_layer(self) -> int:
        # KV slots, not physical layers: recurrent attention layers own one slot per loop.
        # CUDA graph replay pads this list back to num_hidden_layers, so graph capture is off
        # for Dory until the slot count and the layer count can be reconciled.
        return self.config.num_kv_slots()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.embeddings(input_ids) if input_embeds is None else input_embeds
        config = self.config
        n_input = config.n_input_layers
        n_recurrent = config.n_recurrent_layers

        for i in range(n_input):
            hidden = self.layers[i](positions, hidden, forward_batch)
        loop_input = hidden
        state = hidden
        for loop_idx in range(config.n_recurrent_loops):
            hidden = justnorm(state + loop_input)
            for i in range(n_input, n_input + n_recurrent):
                hidden = self.layers[i](
                    positions, hidden, forward_batch, loop_idx=loop_idx
                )
            state = justnorm(hidden)

        for i in range(n_input + n_recurrent, config.num_hidden_layers):
            state = self.layers[i](positions, state, forward_batch)
        return state


class DoryForCausalLM(nn.Module):
    def __init__(
        self,
        config: DoryConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        if parallel.tp_size != 1:
            raise ValueError("Dory currently supports tensor_parallel_size=1")
        if parallel.pp_group.world_size != 1:
            raise ValueError("Dory currently supports pipeline_parallel_size=1")
        self.config = config
        self.quant_config = quant_config
        self.model = DoryModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        # Per-vocabulary logit scale folded in fp32 at the logits, as in the reference.
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embeddings(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        res: LogitsProcessorOutput = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        if res.next_token_logits is not None:
            res.next_token_logits = res.next_token_logits.float() * (
                self.lm_head.logit_scale / self.config.initializer_range
            )
        return res

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            # Checkpoint namespace is backbone.*; the runtime module is .model for graph discovery.
            name = name.replace("backbone.", "model.", 1)
            target = LLAMA_STACKED_MAPPING.try_load(name, tensor, params_dict)
            if target is not None:
                loaded.add(target)
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(name)
        return loaded


EntryClass = DoryForCausalLM
