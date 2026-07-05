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

"""Inference-only GFusion model.

GFusion reuses the DeepSeekV3/DeepSeekV2 MLA implementation and weight loading
path, but switches the attention layers to bidirectional attention for dLLM
decoding and returns full logits for all positions.
"""

from typing import Optional

from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import LazyValue, add_prefix


class GFusionModel(DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._enable_bidirectional_attention()

    def _enable_bidirectional_attention(self) -> None:
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            layer.self_attn.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            layer.self_attn.attn_mha.attn_type = AttentionType.ENCODER_ONLY


class GFusionModelLM(DeepseekV2ForCausalLM):
    packed_modules_mapping = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.use_dsa = is_deepseek_dsa(config)
        self.model = GFusionModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )
        self.capture_aux_hidden_states = False

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = (
            is_prefill_context_parallel_enabled() and not is_deepseek_dsa(config)
        )
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = self.cp_size = None

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_dsa(config))

    def determine_num_fused_shared_experts(self, architecture: Optional[str] = None):
        if architecture is None:
            architectures = getattr(self.config, "architectures", None) or [
                "GFusionModelLM"
            ]
            architecture = architectures[0]
        return super().determine_num_fused_shared_experts(architecture=architecture)


EntryClass = GFusionModelLM
