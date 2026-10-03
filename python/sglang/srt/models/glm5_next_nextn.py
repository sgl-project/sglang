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

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multiplex.pdmux_context import get_pdmux_decode_alt_stream
from sglang.srt.runtime_context import get_disagg

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        n = text_config.num_hidden_layers
        # lookups arrive as checkpoint and normalized names, so every rule has both forms
        draft_rules: dict[str, str] = {}
        for ckpt_prefix in (f"model.language_model.layers.{n}", f"model.layers.{n}"):
            # eh_proj/enorm/hnorm sit beside the block under `model`, not in `decoder`
            draft_rules[f"{ckpt_prefix}.eh_proj"] = "model.eh_proj"
            draft_rules[f"{ckpt_prefix}.enorm"] = "model.enorm"
            draft_rules[f"{ckpt_prefix}.hnorm"] = "model.hnorm"
            draft_rules[ckpt_prefix] = "model.decoder"
        # target rules still normalize the non-draft layers and vision tower in `exclude`
        return Glm5NextForConditionalGeneration.hf_to_sglang_mapper | WeightsMapper(
            orig_to_new_substr=draft_rules,
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

    def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None):
        if get_disagg().enable_pdmux:
            # The inherited NextN block owns an ordinary helper stream. Never
            # let it escape the prefill or decode green-context SM partition.
            alt_stream = get_pdmux_decode_alt_stream(self.model.alt_stream)
            decoder = self.model.decoder
            decoder.self_attn.alt_stream = alt_stream
            if decoder.self_attn.indexer is not None:
                decoder.self_attn.indexer.alt_stream = alt_stream
            if decoder.is_layer_sparse:
                decoder.mlp.alt_stream = alt_stream
        return super().forward(
            input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
