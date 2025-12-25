# Copyright 2023-2024 SGLang Team
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

"""Inference-only DeepSeek V3 Eagle3 Speculative Decoding with MLA attention.

This follows the Llama Eagle3 pattern: inherit from base classes and only
replace the projections that need to accept 2*hidden_size input.
"""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_pp_group
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2ForCausalLM
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda

_is_cuda = is_cuda()


class DeepseekV2DecoderLayerEagle3(DeepseekV2DecoderLayer):
    """DeepSeek V2 decoder layer modified for Eagle3.

    Following the Llama Eagle3 pattern: inherit from base layer and only
    replace the QKV projection to accept 2*hidden_size input.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Initialize parent - this creates self.self_attn with standard MLA attention
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)

        # Eagle3 modification: input is 2*hidden_size (embed + hidden states concatenated)
        eagle3_input_size = 2 * config.hidden_size

        # Replace the QKV projection to accept 2*hidden_size input (like Llama Eagle3)
        if self.self_attn.q_lora_rank is not None:
            self.self_attn.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                eagle3_input_size,  # 2*hidden_size for Eagle3
                self.self_attn.q_lora_rank
                + self.self_attn.kv_lora_rank
                + self.self_attn.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.fused_qkv_a_proj_with_mqa", prefix),
            )
        else:
            # For models without q_lora_rank
            self.self_attn.q_proj = ReplicatedLinear(
                eagle3_input_size,
                self.self_attn.num_heads * self.self_attn.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.q_proj", prefix),
            )
            self.self_attn.kv_a_proj_with_mqa = ReplicatedLinear(
                eagle3_input_size,
                self.self_attn.kv_lora_rank + self.self_attn.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("self_attn.kv_a_proj_with_mqa", prefix),
            )

        # Add extra norm for Eagle3 (for hidden states, input_layernorm is for embeds)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eagle3 forward: normalize separately, concat, then use base attention."""
        residual = hidden_states

        # Eagle3 preprocessing: apply separate norms then concat
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        concat_hidden = torch.cat([embeds, hidden_states], dim=-1)

        # Create zero allocator for attention
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=concat_hidden.device,
        )

        # Compute QKV latent and set up attn_inputs so base attention's fetch_qkv_latent() works
        if self.self_attn.q_lora_rank is not None:
            qkv_latent = self.self_attn.fused_qkv_a_proj_with_mqa(concat_hidden)[0]
        else:
            # For models without fused projection
            q = self.self_attn.q_proj(concat_hidden)[0]
            kv = self.self_attn.kv_a_proj_with_mqa(concat_hidden)[0]
            qkv_latent = torch.cat([q, kv], dim=-1)

        # Set up attn_inputs so the base attention can use fetch_qkv_latent()
        attn_inputs = AttentionInputs(
            concat_hidden, forward_batch, lambda x, fb: qkv_latent
        )
        get_attn_tp_context().set_attn_inputs(attn_inputs)

        # Now call the base attention - it will use fetch_qkv_latent() which works!
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=concat_hidden,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        # All-reduce attention output across TP ranks
        # (base attention has reduce_results=False, so we need to do it manually)
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        # Post-attention processing
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states, forward_batch)

        return hidden_states, residual


class DeepseekModelEagle3(nn.Module):
    """DeepSeek model for Eagle3 speculative decoding."""

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
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Get target hidden size for projection
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        # Eagle3 fusion layer: projects 3*hidden_size to hidden_size
        self.fc = nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Single decoder layer for Eagle3
        self.midlayer = DeepseekV2DecoderLayerEagle3(
            config, 0, quant_config, prefix=add_prefix("midlayer", prefix)
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, list]:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states

        # Project hidden states if dimension mismatch
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # Handle idle batch
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

        # Return hidden states for logits and auxiliary hidden states for next draft
        return hidden_states_to_logits, [hidden_states_to_aux]


class Eagle3DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    """Eagle3 speculative decoding model with DeepSeek V3 MLA attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_nsa(config))

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = DeepseekModelEagle3(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Handle tied embeddings and lm_head
        self.load_lm_head_from_target = False
        if getattr(self.config, "tie_word_embeddings", False):
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
    ) -> torch.Tensor:
        hidden_states, aux_hidden_states = self.model(
            input_ids, positions, forward_batch
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        # Parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Handle fused QKV projections for MLA
        a_proj_params = {}

        for name, loaded_weight in weights:
            # Handle d2t token mapping
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            # Handle stacked parameters (gate_up_proj)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name_full = f"model.{name}" if name not in params_dict else name
                if param_name_full in params_dict:
                    param = params_dict[param_name_full]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break

            if is_stacked:
                continue

            # Handle fused a_proj for MLA
            if getattr(self.config, "q_lora_rank", None) is not None and (
                "q_a_proj" in name or "kv_a_proj_with_mqa" in name
            ):
                # Cache these to fuse later
                layer_prefix = name.rsplit(".", 2)[0]
                if layer_prefix not in a_proj_params:
                    a_proj_params[layer_prefix] = {}
                if "q_a_proj" in name:
                    a_proj_params[layer_prefix]["q_a_proj"] = loaded_weight
                else:
                    a_proj_params[layer_prefix]["kv_a_proj"] = loaded_weight

                # Check if we can fuse now
                if (
                    "q_a_proj" in a_proj_params[layer_prefix]
                    and "kv_a_proj" in a_proj_params[layer_prefix]
                ):
                    q_weight = a_proj_params[layer_prefix]["q_a_proj"]
                    kv_weight = a_proj_params[layer_prefix]["kv_a_proj"]
                    fused_weight = torch.cat([q_weight, kv_weight], dim=0)

                    # Build the fused parameter name from the layer prefix
                    # (Don't use string replace on the result - it causes double replacement)
                    fused_name = f"{layer_prefix}.fused_qkv_a_proj_with_mqa.weight"
                    param_name_full = (
                        f"model.{fused_name}"
                        if fused_name not in params_dict
                        else fused_name
                    )
                    if param_name_full in params_dict:
                        param = params_dict[param_name_full]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused_weight)
                continue

            # Handle regular parameters
            param_name_full = name if name in params_dict else f"model.{name}"
            if param_name_full in params_dict:
                param = params_dict[param_name_full]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # Post-process weights to set up w_kc and w_vc for MLA attention
        self.post_load_weights()

    def post_load_weights(self):
        """Process kv_b_proj weights into w_kc and w_vc for MLA attention."""
        self_attn = self.model.midlayer.self_attn

        # Get the kv_b_proj weight
        w = self_attn.kv_b_proj.weight

        # Unflatten and split into w_kc and w_vc
        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)

        # Set w_kc and w_vc (transposed for bmm)
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)

        # Handle weight scale if present
        if hasattr(self_attn.kv_b_proj, "weight_scale") and self_attn.w_scale is None:
            self_attn.w_scale = self_attn.kv_b_proj.weight_scale

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [Eagle3DeepseekV3ForCausalLM]
