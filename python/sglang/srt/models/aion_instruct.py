# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from modeling_aion_instruct.py in
# https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B
"""Inference-only Aion-1.0-Instruct model compatible with HuggingFace weights."""

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm_rope, can_use_jit_qk_norm_rope
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


def get_attention_sliding_window_size(config: PretrainedConfig) -> int:
    # HF's window counts the current token; SGLang's counts only earlier tokens.
    return config.sliding_window - 1


class AionInstructAttention(nn.Module):
    """GQA with per-head Q/K RMSNorm, over full context or a sliding window."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_theta: float,
        rope_scaling: Dict[str, Any],
        max_position_embeddings: int,
        rms_norm_eps: float,
        attention_bias: bool,
        sliding_window_size: int = -1,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        attn_tp_size = get_parallel().attn_tp_size
        assert num_heads % attn_tp_size == 0
        if num_kv_heads >= attn_tp_size:
            assert num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % num_kv_heads == 0
        self.num_heads = num_heads // attn_tp_size
        self.num_kv_heads = max(1, num_kv_heads // attn_tp_size)

        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            # The decoder layer's stage boundary completes the sum.
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream
        self.use_jit_qk_norm_rope = can_use_jit_qk_norm_rope(
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            head_dim=self.head_dim,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states)[0].split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q, k = apply_qk_norm_rope(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            positions=positions,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
            use_jit=self.use_jit_qk_norm_rope,
        )
        return self.o_proj(self.attn(q, k, v, forward_batch))[0]


class AionInstructMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}; only silu is.")
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            # The decoder layer's stage boundary completes the sum.
            reduce_results=False,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


class AionInstructDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.self_attn = AionInstructAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_id=layer_id,
            rope_theta=config.rope_parameters[config.layer_types[layer_id]][
                "rope_theta"
            ],
            rope_scaling=config.rope_parameters[config.layer_types[layer_id]],
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            sliding_window_size=(
                get_attention_sliding_window_size(config)
                if config.layer_types[layer_id] == "sliding_attention"
                else -1
            ),
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = AionInstructMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn, ffn = self.stage_facts(config=config, layer_id=layer_id)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: PretrainedConfig, layer_id: int):
        """The stages an Aion layer declares, from the config alone; make_layers
        reads the same function for a layer it does not build."""
        return declare_attn(), declare_ffn()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)

        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class AionInstructModel(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: AionInstructDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=get_stream("alt") if _is_cuda else None,
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: AionInstructDecoderLayer.stage_facts(
                config=config, layer_id=idx
            ),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        return residual_batch.final_norm(hidden_states, forward_batch, self.norm)


class AionInstructForCausalLM(nn.Module):
    # HF checkpoint names folded into SGLang's fused projections.
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = AionInstructModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_attention_sliding_window_size(self) -> int:
        return get_attention_sliding_window_size(self.config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        return self.logits_processor(
            input_ids,
            self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            ),
            self.lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            for param_name, shard_name, shard_id in self.stacked_params_mapping:
                if shard_name not in name:
                    continue
                param = params_dict[name.replace(shard_name, param_name)]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )


EntryClass = AionInstructForCausalLM
