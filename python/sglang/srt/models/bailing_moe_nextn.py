# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""SGLang BailingMoENextN model."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

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
from sglang.srt.models.bailing_moe import BailingMoEBlock, BailingMoEForCausalLM
from sglang.srt.models.bailing_moe_linear import (
    BailingMoELinearDecoderLayer,
    BailingMoeV2_5ForCausalLM,
)
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoELinearDecoderLayer as BailingMoeV3DecoderLayer,
)
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator, add_prefix

logger = logging.getLogger(__name__)


def _is_bailing_moe_v3_config(config: PretrainedConfig) -> bool:
    """Ling-V3 (KDA + gated MLA) vs the V2.5 lightning checkpoint.

    ``use_kda`` is set by BailingHybridConfig from the presence of a short
    conv, which is exactly what distinguishes the two.
    """
    return config.model_type == "bailing_hybrid" and config.use_kda


class BailingMoEModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_fused_shared_experts: int = 0,
    ) -> None:
        super().__init__()
        self.layer_group_size = 1
        self.start_layer = 0
        self.end_layer = 1
        self.total_num_layers = 1
        self.vocab_size = config.vocab_size
        config.for_nextn_model = True

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.debug(
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
            )
            quant_config = None

        self.vocab_size = config.vocab_size

        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("word_embeddings", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(f"layers.{config.num_hidden_layers}.eh_proj", prefix),
        )

        self.is_hybrid = (
            hasattr(config, "model_type") and config.model_type == "bailing_hybrid"
        )
        if self.is_hybrid:
            config.attention_type = 1
            decoder_layer_cls = BailingMoELinearDecoderLayer
            decoder_kwargs = {
                "quant_config": quant_config,
                "layer_id": 0,
                "is_nextn": True,
                "prefix": add_prefix(f"layers.{config.num_hidden_layers}", prefix),
            }
            if _is_bailing_moe_v3_config(config):
                decoder_layer_cls = BailingMoeV3DecoderLayer
                decoder_kwargs["num_fused_shared_experts"] = num_fused_shared_experts
            self.decoder = decoder_layer_cls(config, **decoder_kwargs)
        else:
            self.decoder = BailingMoEBlock(
                config,
                0,
                quant_config=quant_config,
                prefix=add_prefix("decoder", prefix),
            )

        self.shared_head = nn.Module()
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        if input_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            hidden_states, _ = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(
                            forward_batch.spec_info.hidden_states.to(
                                self.hnorm.weight.dtype
                            )
                        ),
                    ),
                    dim=-1,
                )
            )

        residual = None
        if self.is_hybrid:
            device = input_ids.device
            zero_allocator = BumpAllocator(
                buffer_size=self.total_num_layers
                * 2
                * (2 if forward_batch.can_run_tbo else 1),
                dtype=torch.float32,
                device=device,
            )
            hidden_states, residual = self.decoder(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
            )
        else:
            hidden_states, residual = self.decoder(
                positions, hidden_states, forward_batch, residual
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.final_layernorm(hidden_states, residual)
            else:
                hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class BailingMoeForCausalLMNextN(nn.Module):
    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "attention.dense": "attention.o_proj",
        },
    )

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        if not _is_bailing_moe_v3_config(hf_config):
            return None
        return BailingMoeV3ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config,
            quant_config,
            expected_architecture="BailingMoeForCausalLMNextN",
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        is_bailing_moe_v3 = _is_bailing_moe_v3_config(config)
        if is_bailing_moe_v3:
            BailingMoeV3ForCausalLM.determine_num_fused_shared_experts(self)
        elif hasattr(self, "determine_num_fused_shared_experts"):
            # Asystem has determine_num_fused_shared_experts but theta does not.
            self.determine_num_fused_shared_experts("BailingMoeForCausalLMNextN")

        self.model = BailingMoEModelNextN(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            num_fused_shared_experts=self.num_fused_shared_experts,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        if is_bailing_moe_v3:
            self.base_load_weights_func = BailingMoeV3ForCausalLM.load_weights
            self.post_load_weights_func = BailingMoeV3ForCausalLM.post_load_weights
        elif config.model_type == "bailing_hybrid":
            self.base_load_weights_func = BailingMoeV2_5ForCausalLM.load_weights
            self.post_load_weights_func = BailingMoeV2_5ForCausalLM.post_load_weights
        else:
            self.base_load_weights_func = BailingMoEForCausalLM.load_weights
            # V1 BailingMoeAttention is standard QKV (no kv_b_proj), no fixup needed.
            self.post_load_weights_func = None

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor):
        # Defensive: V3's load_weights references `self.weight_direct_load` as the
        # default in `getattr(param, "weight_loader", self.weight_direct_load)`,
        # which is eagerly evaluated. Today the linear-attn branch that uses it is
        # never reached on NextN (attention_type is forced to softmax and
        # is_linear_layer(0, 1) is False), but keep this forward so a future change
        # that enables KDA-style layers on NextN doesn't hit AttributeError.
        BailingMoeV3ForCausalLM.weight_direct_load(param, loaded_weight)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.base_load_weights_func(self, weights, is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept
        # only because the underlying `load_weights` flow calls `self.post_load_weights`
        # with `is_nextn=...` as a kwarg.
        if self.post_load_weights_func is None:
            return
        self.post_load_weights_func(self, is_nextn=True, weight_names=weight_names)


EntryClass = [BailingMoeForCausalLMNextN]
