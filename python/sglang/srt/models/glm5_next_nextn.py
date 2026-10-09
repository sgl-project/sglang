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

# [patch #12 df1cb1102c - remap NextN quant_config.ignore prefixes for INT8 MTP]
import copy
import logging

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    # [patch #9 9418627c11 - fix INT8 MTP draft accept via nextn weight mapper]
    # Draft checkpoints fuse q_a_proj/kv_a_proj_with_mqa. The loader only adds
    # this packing automatically for quark, so the INT8 path needs the class to
    # declare it or should_ignore_layer() cannot split the fused draft name.
    packed_modules_mapping = {
        **DeepseekV3ForCausalLMNextN.packed_modules_mapping,
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

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
            # [patch #9 9418627c11 - shared_head.norm also sits under `model`]
            draft_rules[f"{ckpt_prefix}.shared_head.norm"] = "model.shared_head.norm"
            draft_rules[ckpt_prefix] = "model.decoder"
        # target rules still normalize the non-draft layers and vision tower in `exclude`
        return Glm5NextForConditionalGeneration.hf_to_sglang_mapper | WeightsMapper(
            orig_to_new_substr=draft_rules,
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Return a quant_config whose ``ignore`` prefixes match NextN modules.

        The NextN (draft) block of an INT8 checkpoint is mixed precision: 6
        attention/indexer components are listed in ``quantization_config.ignore``
        (kept BF16) while the MoE experts are not (quantized to INT8). The
        checkpoint writes those entries with the HF prefix
        ``model.language_model.layers.<num_hidden_layers>.*``, whereas sglang
        builds the draft submodules under ``model.decoder.*`` / ``model.*``.

        ``model_loader.loader`` rewrites a quant config through
        ``quant_config.apply_weight_name_mapper(hf_to_sglang_mapper)``, and on
        0.5.21 ``W8A8Int8Config`` implements that hook, so a config that went
        through the model loader with this class's mapper arrives here already
        remapped: the mapper below changes nothing and the original object is
        returned untouched.

        This method is the fallback for paths that bypass that hook (e.g. a
        config built outside ``model_loader``): the HF-prefixed entries would
        otherwise never match ``should_ignore_layer()``, and the 6 BF16
        components would get INT8 parameters loaded from BF16 payloads, which
        collapses draft quality. A deepcopy is remapped with the same
        ``WeightsMapper`` that ``load_weights`` uses, so the target model's
        config stays intact.
        """
        # [patch #12 df1cb1102c - remap NextN quant_config.ignore prefixes for INT8 MTP] 旧代码：
        # """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        # inheriting global FP8 quantization would corrupt its QKV weights."""
        # raw_quant_config = getattr(config, "quantization_config", None) or {}
        # if hasattr(raw_quant_config, "to_dict"):
        #     raw_quant_config = raw_quant_config.to_dict()
        # ignored = (
        #     raw_quant_config.get("ignore", [])
        #     if isinstance(raw_quant_config, dict)
        #     else []
        # )
        # nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        # if nextn_layer_pattern in ignored:
        #     logger.warning(
        #         "GLM5 NextN layer %s is checkpoint-declared unquantized; "
        #         "using BF16 draft modules",
        #         nextn_layer_pattern,
        #     )
        #     return None
        # return super()._resolve_nextn_quant_config(config, quant_config)
        if quant_config is None:
            return None

        # Configs without an ``ignore`` list (e.g. quark) keep parent behavior.
        if not hasattr(quant_config, "ignore"):
            return super()._resolve_nextn_quant_config(config, quant_config)

        ignore = list(quant_config.ignore or [])
        if not ignore:
            return quant_config

        mapper = self.__class__.get_hf_to_sglang_mapper(config)
        remapped_ignore = mapper.apply_list(ignore)
        if remapped_ignore == ignore:
            # Already rewritten by apply_weight_name_mapper in the loader.
            return quant_config

        remapped = copy.deepcopy(quant_config)
        remapped.ignore = remapped_ignore

        nextn_prefixes = (
            "model.decoder.",
            "model.eh_proj",
            "model.enorm",
            "model.hnorm",
            "model.shared_head.norm",
        )
        nextn_hits = sum(1 for e in remapped_ignore if e.startswith(nextn_prefixes))
        logger.info(
            "GLM5 NextN precise quant fix: type=%s, "
            "orig_ignore=%d, remapped_ignore=%d, "
            "nextn_decoder_entries=%d (expect 6 for INT8 attn BF16), "
            "sample_remapped=%s",
            quant_config.get_name(),
            len(ignore),
            len(remapped_ignore),
            nextn_hits,
            [e for e in remapped_ignore if e.startswith("model.decoder.")][:6],
        )

        return remapped

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
