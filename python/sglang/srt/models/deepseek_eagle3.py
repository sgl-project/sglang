"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Inference-only Eagle3 model with DeepSeek MLA attention for speculative decoding."""

import copy
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
)
from sglang.srt.utils import add_prefix


class DeepseekEagle3DecoderLayer(nn.Module):
    """Single decoder layer for Eagle3 with DeepSeek MLA attention.

    For the first layer (layer_id=0), the fused QKV-A projection is overridden
    to accept 2*hidden_size input (concatenation of embeddings and hidden states).
    Always uses dense MLP (no MoE).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        q_lora_rank = getattr(config, "q_lora_rank", None)

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=True,
            prefix=add_prefix("self_attn", prefix),
        )

        # Override the fused QKV-A projection for the first layer to accept
        # 2*hidden_size input (concatenated embeddings + hidden states)
        if q_lora_rank is not None:
            fused_output_size = (
                q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
            )
            self.self_attn.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                2 * config.hidden_size,
                fused_output_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.fused_qkv_a_proj_with_mqa", prefix),
            )
        else:
            # Fallback for models without q_lora_rank: override q_proj and kv_a_proj
            from sglang.srt.layers.linear import ColumnParallelLinear

            self.self_attn.q_proj = ColumnParallelLinear(
                2 * config.hidden_size,
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.q_proj", prefix),
            )
            self.self_attn.kv_a_proj_with_mqa = ReplicatedLinear(
                2 * config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.kv_a_proj_with_mqa", prefix),
            )

        self.mlp = DeepseekV2MLP(
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
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # Normalize embeddings and hidden states separately, then concatenate
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention (MLA)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Post-attention norm + residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Dense MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class DeepseekEagle3Model(nn.Module):
    """Eagle3 backbone model with MLA attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Target hidden size may differ from draft hidden size
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)

        # FC layer to project concatenated aux hidden states (3 layers) to hidden_size
        self.fc = torch.nn.Linear(
            target_hidden_size * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Single decoder layer
        self.midlayer = DeepseekEagle3DecoderLayer(
            config, layer_id=0, quant_config=quant_config, prefix=prefix
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                assert embeds is not None
                embeds = torch.cat(
                    [embeds[:-1], self.embed_tokens(input_ids[-1].unsqueeze(0))]
                )
            if embeds is None:
                embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # Idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class Eagle3DeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    """Eagle3 speculative decoding model with DeepSeek MLA attention.

    Inherits from DeepseekV2ForCausalLM for weight loading compatibility
    but uses a custom __init__ that builds a simplified Eagle3 backbone.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Skip DeepseekV2ForCausalLM.__init__ and call nn.Module.__init__ directly
        # Eagle3 doesn't need MoE, pipeline parallelism, shared expert fusion, etc.
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if config.num_hidden_layers != 1:
            raise ValueError("Eagle3 currently only supports 1 decoder layer")

        # Fuse q_a_proj and kv_a_proj_with_mqa for weight loading
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        self.packed_modules_mapping = {}
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        # No MoE / shared expert fusion in Eagle3
        self.num_fused_shared_experts = 0

        self.model = DeepseekEagle3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # LM head with draft vocab size
        self.load_lm_head_from_target = False
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = getattr(config, "draft_vocab_size", None)
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = getattr(config_, "draft_vocab_size", config_.vocab_size)
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Cache for fusing q_a_proj + kv_a_proj_with_mqa
        cached_a_proj = {} if self.fuse_qkv_a_proj else None

        for name, loaded_weight in weights:
            # Handle d2t (draft-to-target) token mapping
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0]
                )
                continue

            if "t2d" in name:
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked params (gate_up_proj)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                full_name = name if name in params_dict else f"model.{name}"
                if full_name in params_dict:
                    param = params_dict[full_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break

            if is_stacked:
                continue

            # Handle fused q_a_proj + kv_a_proj_with_mqa
            if self.fuse_qkv_a_proj and (
                "q_a_proj" in name or "kv_a_proj_with_mqa" in name
            ):
                cached_a_proj[name] = loaded_weight
                q_a_proj_name = (
                    name
                    if "q_a_proj" in name
                    else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                )
                kv_a_proj_name = (
                    name
                    if "kv_a_proj_with_mqa" in name
                    else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                )

                # When both parts are cached, concatenate and load
                if q_a_proj_name in cached_a_proj and kv_a_proj_name in cached_a_proj:
                    q_a_proj_weight = cached_a_proj[q_a_proj_name]
                    kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                    fused_weight = torch.cat(
                        [q_a_proj_weight, kv_a_proj_weight], dim=0
                    )

                    fused_name = (
                        q_a_proj_name.replace(
                            "q_a_proj", "fused_qkv_a_proj_with_mqa"
                        )
                    )
                    full_name = (
                        fused_name
                        if fused_name in params_dict
                        else f"model.{fused_name}"
                    )
                    if full_name in params_dict:
                        param = params_dict[full_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused_weight)
                    cached_a_proj.pop(q_a_proj_name)
                    cached_a_proj.pop(kv_a_proj_name)
                continue

            # Handle regular parameters
            full_name = name if name in params_dict else f"model.{name}"
            if full_name in params_dict:
                param = params_dict[full_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


class Eagle3DeepseekV3ForCausalLM(Eagle3DeepseekV2ForCausalLM):
    """Alias for DeepSeek V3 compatibility."""

    pass


EntryClass = [Eagle3DeepseekV2ForCausalLM, Eagle3DeepseekV3ForCausalLM]
