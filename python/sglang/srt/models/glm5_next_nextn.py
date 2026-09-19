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

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    _NEXTN_SPEC_WEIGHT_NAMES = ("shared_head.norm", "eh_proj", "enorm", "hnorm")

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        layer_prefixes = (
            f"model.layers.{text_config.num_hidden_layers}",
            f"model.language_model.layers.{text_config.num_hidden_layers}",
        )
        special_mapping = {
            f"{layer_prefix}.{name}": f"model.{name}"
            for layer_prefix in layer_prefixes
            for name in cls._NEXTN_SPEC_WEIGHT_NAMES
        }
        decoder_mapping = {
            f"{layer_prefix}.": "model.decoder." for layer_prefix in layer_prefixes
        }
        return WeightsMapper(
            orig_to_new_substr=special_mapping,
            orig_to_new_prefix=decoder_mapping,
        )

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )

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
