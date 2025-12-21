# Copyright 2023-2025 SGLang Team
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

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/modular_glmasr.py
"""Inference-only GLM-ASR-HF model compatible with HuggingFace weights."""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GlmasrEncoderConfig, GlmasrConfig
from transformers.models.glmasr.modeling_glmasr import (
    GlmasrEncoder,
    GlmasrMultiModalProjector,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama2 import LlamaForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class GlmasrMultimodalProcessor(BaseMultimodalProcessor):
    models = [GlmasrForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|begin_of_audio|><|pad|><|end_of_audio|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|begin_of_audio\|>(?:<\|pad\|>)+<\|end_of_audio\|>"
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        assert (
            "feature_attention_mask" in ret
        ), "feature_attention_mask not found in processor output"
        input_lengths = ret["feature_attention_mask"].sum(dim=-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1

        mm_items[0].audio_feature_lens = output_lengths

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }

class GlmasrForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: GlmasrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if getattr(self.config, "audio_config", None) is None:
            self.config.audio_config = GlmasrEncoderConfig(
                self.config._name_or_path
            )

        self.audio_tower = GlmasrEncoder(
            config.audio_config,
        )
        self.multi_modal_projector = GlmasrMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract audio features from input items
        input_features = torch.cat([item.feature for item in items], dim=0).type(
            self.audio_tower.dtype
        )

        audio_embeds = self.audio_tower(input_features).last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_embeds)

        audio_feature_lens = torch.cat([item.audio_feature_lens for item in items])
        new_embeds = []
        for i, d in zip(audio_feature_lens, audio_embeds):
            new_embeds.append(d[: i.item()])

        return torch.cat(new_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_tower" in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GlmasrForConditionalGeneration
