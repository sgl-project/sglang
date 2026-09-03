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

import logging
from copy import copy

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
)
from sglang.srt.models.utils import WeightsMapper
from torch import nn
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class Glm5NextModelNextN(DeepseekModelNextN):
    def _build_decoder(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        moe_quant_config_override: QuantizationConfig | None,
        prefix: str,
        alt_stream: torch.cuda.Stream | None,
    ) -> nn.Module:
        # The published NextN block uses GLM's DSA/MoE layer but deliberately
        # omits the target model's mHC parameters. Build that layer with mHC
        # disabled instead of inheriting DeepSeek's decoder implementation.
        decoder_config = copy(config)
        decoder_config.mhc = False
        return Glm5NextDecoderLayer(
            config=decoder_config,
            layer_id=config.num_hidden_layers,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,
            prefix=prefix,
            alt_stream=alt_stream,
        )


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        return WeightsMapper(
            orig_to_new_substr={
                f"model.layers.{text_config.num_hidden_layers}": "model.decoder",
            },
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights."""
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        return super()._resolve_nextn_quant_config(config, quant_config)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )
        self.hot_token_id = None

    def _build_nextn_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> Glm5NextModelNextN:
        return Glm5NextModelNextN(config, quant_config, prefix=prefix)

    def set_embed(self, embed):
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
