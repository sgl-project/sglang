# SPDX-License-Identifier: Apache-2.0
#
# Gemma2 2B text encoder for SANA.
#
# This is a decoder-only language model used as a text encoder: we feed
# in tokenized text and extract the final hidden states (not logits) as
# the conditioning signal for SANA's cross-attention layers.
#
# Architecture follows google/gemma-2-2b-it:
#   - 26 layers, alternating global / sliding-window attention
#   - GQA with 8 query heads, 4 KV heads, head_dim=256
#   - Pre/post attention + pre/post feedforward LayerNorm (Gemma2-style)
#   - GeGLU activation (gelu_pytorch_tanh)
#
# Adapted from the Gemma3 text model implementation in this codebase.

import logging
from typing import Any, Iterable

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import GeluAndMul
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma2 uses `gelu_pytorch_tanh` as the hidden activation. "
                f"Got: {hidden_act}"
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Gemma2Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma2Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        arch = config.arch_config
        self.head_dim = arch.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = arch.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=arch.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=arch.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Gemma2 interleaves global (even layers) and sliding-window (odd layers)
        # attention. This pattern reduces memory for long sequences while
        # maintaining global context every other layer.
        self.is_sliding = (layer_id % 2) == 1
        if self.is_sliding:
            self.sliding_window = arch.sliding_window
        else:
            self.sliding_window = None

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=arch.max_position_embeddings,
            base=arch.rope_theta,
            is_neox_style=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        attn_mask = torch.zeros(
            (seq_len, seq_len), device=hidden_states.device, dtype=torch.float32
        )
        causal = torch.triu(
            torch.ones(
                (seq_len, seq_len), device=hidden_states.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        attn_mask = attn_mask.masked_fill(causal, float("-inf"))
        if self.is_sliding and self.sliding_window is not None:
            idx = torch.arange(seq_len, device=hidden_states.device)
            dist = idx[None, :] - idx[:, None]
            too_far = dist > self.sliding_window
            attn_mask = attn_mask.masked_fill(too_far, float("-inf"))

        if attention_mask is not None:
            key_pad = ~attention_mask.to(torch.bool)
            attn_mask = attn_mask[None, None, :, :].expand(
                batch_size, 1, seq_len, seq_len
            )
            attn_mask = attn_mask.masked_fill(
                key_pad[:, None, None, :].expand(batch_size, 1, seq_len, seq_len),
                float("-inf"),
            )

        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": 0.0,
            "is_causal": False,
            "scale": self.scaling,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, **attn_kwargs
        )

        # NOTE: Gemma2 specifies attn_logit_softcapping (tanh(logits/cap)*cap) but
        # PyTorch's scaled_dot_product_attention does not support it natively.
        # For short text-encoder sequences (~300 tokens), the quality impact is
        # negligible. A custom attention kernel would be needed for full fidelity.

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.self_attn = Gemma2Attention(
            layer_id=layer_id,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=arch.num_attention_heads,
            num_kv_heads=arch.num_key_value_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=arch.intermediate_size,
            hidden_act=arch.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Gemma2RMSNorm(self.hidden_size, eps=arch.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(
            self.hidden_size, eps=arch.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            self.hidden_size, eps=arch.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            self.hidden_size, eps=arch.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2Model(nn.Module):
    """Gemma2 text encoder model for SANA pipeline."""

    _fsdp_shard_conditions = []

    def __init__(self, config: Gemma2Config, **kwargs):
        super().__init__()
        self.config = config
        arch = config.arch_config
        self.quant_config = None

        self.vocab_size = arch.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            arch.hidden_size,
            org_num_embeddings=arch.vocab_size,
            quant_config=self.quant_config,
        )
        self.embed_scale = arch.hidden_size**0.5

        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayer(
                    layer_id=i,
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"model.layers.{i}",
                )
                for i in range(arch.num_hidden_layers)
            ]
        )

        self.norm = Gemma2RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config.arch_config, "output_hidden_states", False)
        )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(position_ids, hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        stacked_params_mapping = getattr(
            self.config.arch_config, "stacked_params_mapping", None
        )
        if stacked_params_mapping is None:
            stacked_params_mapping = [
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
                (".gate_up_proj", ".gate_proj", "0"),
                (".gate_up_proj", ".up_proj", "1"),
            ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # HF Gemma2Model stores weights as model.layers.X... / model.embed_tokens...
            # Strip "model." prefix if present to match our naming
            if name.startswith("model."):
                name = name[len("model.") :]

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                self._load_with_shard_id(weight_loader, param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params

    @staticmethod
    def _load_with_shard_id(weight_loader, param, loaded_weight, shard_id):
        try:
            weight_loader(param, loaded_weight, shard_id)
            return
        except (AssertionError, TypeError):
            pass

        if isinstance(shard_id, str):
            mapping = {"q": 0, "k": 1, "v": 2}
            if shard_id in mapping:
                weight_loader(param, loaded_weight, mapping[shard_id])
                return
            if shard_id.isdigit():
                weight_loader(param, loaded_weight, int(shard_id))
                return
        elif isinstance(shard_id, int):
            mapping = {0: "q", 1: "k", 2: "v"}
            if shard_id in mapping:
                weight_loader(param, loaded_weight, mapping[shard_id])
                return

        raise TypeError(
            f"Unsupported shard_id={shard_id!r} for weight_loader={weight_loader}"
        )


EntryClass = Gemma2Model
