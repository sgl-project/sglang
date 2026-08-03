# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

# Copyright 2025 SGLang Team
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

# Adapted from sglang/srt/models/granitemoe.py and the HuggingFace Transformers
# `granitemoe_swa` implementation.
"""Inference-only GraniteMoeSWA model.

GraniteMoeSWA is GraniteMoe (mixture-of-experts) combined with the sliding-window
attention and learnable per-head attention sinks of GraniteSWA, plus optional
shared experts (disabled by default, `shared_intermediate_size == 0`). It reuses
`GraniteSWAAttention` from `granite_swa.py` (sink + per-layer sliding window) and
`GraniteMoeMoE` from `granitemoe.py`; the rest mirrors `granitemoe.py`.
"""

from typing import Iterable, Optional

import torch
from torch import nn
from transformers.models.granitemoe_swa import GraniteMoeSWAConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models import mixtral
from sglang.srt.models.granite_swa import GraniteSWAAttention
from sglang.srt.models.granitemoe import GraniteMoeMoE
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class GraniteMoeSWASharedMLP(nn.Module):
    """Shared-expert MLP (only built when `shared_intermediate_size > 0`).

    Uses the HuggingFace `input_linear`/`output_linear` names so checkpoint
    weights load without remapping. There is no shared-expert reference in the
    SGLang Granite family, so this is implemented directly (it mirrors the
    routed expert's gated SiLU MLP, sized by `shared_intermediate_size`).
    """

    def __init__(
        self,
        config: GraniteMoeSWAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.input_linear = MergedColumnParallelLinear(
            self.input_size,
            [self.hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("input_linear", prefix),
        )
        self.output_linear = RowParallelLinear(
            self.hidden_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("output_linear", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.input_linear(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.output_linear(hidden_states)
        return hidden_states


class GraniteMoeSWADecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteMoeSWAConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Reuse GraniteSWA's attention (learnable sink + per-layer sliding window).
        self.self_attn = GraniteSWAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.block_sparse_moe = GraniteMoeMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
        )
        self.shared_mlp = (
            None
            if getattr(config, "shared_intermediate_size", 0) == 0
            else GraniteMoeSWASharedMLP(
                config,
                quant_config=quant_config,
                prefix=add_prefix("shared_mlp", prefix),
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            # Compute the shared expert BEFORE the routed experts: `block_sparse_moe`
            # (FusedMoE) consumes/mutates `hidden_states` in-place, so the shared MLP
            # must read the pristine post-attention-layernorm input first (matches the
            # qwen2_moe shared-expert ordering in SGLang).
            shared_output = self.shared_mlp(hidden_states)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = hidden_states + shared_output
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


class GraniteMoeSWAModel(nn.Module):
    def __init__(
        self,
        config: GraniteMoeSWAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.embedding_multiplier = config.embedding_multiplier

        self.layers = nn.ModuleList(
            [
                GraniteMoeSWADecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        hidden_states *= self.embedding_multiplier

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                forward_batch,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GraniteMoeSWAForCausalLM(nn.Module):
    def __init__(
        self,
        config: GraniteMoeSWAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = GraniteMoeSWAModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # Granite logit scaling factors are applied via division, but
        # LogitsProcessor expects a multiplicative factor.
        if hasattr(config, "logits_scaling"):
            logit_scale = 1.0 / config.logits_scaling
        else:
            logit_scale = None
        self.logits_processor = LogitsProcessor(config, logit_scale=logit_scale)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if not get_embedding:
            logits_processor_output: LogitsProcessorOutput = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
            return logits_processor_output
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parallel = get_parallel()
        tp_size = parallel.tp_size
        new_weights = {}
        for n, p in weights:
            if n.endswith(".block_sparse_moe.experts.gate_up_proj"):
                for e in range(p.size(0)):
                    w1_name = n.replace(
                        ".block_sparse_moe.experts.gate_up_proj",
                        f".block_sparse_moe.experts.{e}.w1.weight",
                    )
                    w3_name = n.replace(
                        ".block_sparse_moe.experts.gate_up_proj",
                        f".block_sparse_moe.experts.{e}.w3.weight",
                    )
                    w1_param, w3_param = p[e].chunk(2, dim=0)
                    assert w1_name not in new_weights
                    assert w3_name not in new_weights
                    new_weights[w1_name] = w1_param
                    new_weights[w3_name] = w3_param
            elif n.endswith(".block_sparse_moe.experts.down_proj"):
                for e in range(p.size(0)):
                    w2_name = n.replace(
                        ".block_sparse_moe.experts.down_proj",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    w2_param = p[e]
                    assert w2_name not in new_weights
                    new_weights[w2_name] = w2_param
            elif n.endswith(".block_sparse_moe.router.weight"):
                gate_name = n.replace(
                    ".block_sparse_moe.router.weight",
                    ".block_sparse_moe.gate.weight",
                )
                assert gate_name not in new_weights
                new_weights[gate_name] = p
            elif n.endswith(".sinks"):
                # Attention sinks: one scalar per head, sharded across TP ranks.
                # Pre-slice to the local head count so the default loader copies
                # a correctly-shaped tensor.
                shard = p.numel() // tp_size
                start = parallel.attn_tp_rank * shard
                new_weights[n] = p.narrow(0, start, shard)
            else:
                new_weights[n] = p
        mixtral.MixtralForCausalLM.load_weights(self, new_weights.items())


EntryClass = [GraniteMoeSWAForCausalLM]
