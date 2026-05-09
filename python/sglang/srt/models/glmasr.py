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
from transformers import GlmAsrConfig, GlmAsrEncoderConfig
from transformers.models.glmasr.modeling_glmasr import (
    GlmAsrEncoder,
    GlmAsrMultiModalProjector,
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
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class GlmAsrForConditionalGeneration(nn.Module):
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
        config: GlmAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if getattr(self.config, "audio_config", None) is None:
            self.config.audio_config = GlmAsrEncoderConfig(self.config._name_or_path)

        self.audio_tower = GlmAsrEncoder(
            config.audio_config,
        )
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)
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
        audio_embeds = audio_embeds.reshape(
            -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(audio_embeds)

        return audio_embeds

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        # The inner ``self.language_model`` (LlamaForCausalLM) owns its own
        # AutoWeightsLoader-based ``load_weights`` and handles tie_word_embeddings
        # / fused QKV / gate_up internally. The walker just routes
        # ``language_model.*`` checkpoints into it. ``audio_tower.*`` and
        # ``multi_modal_projector.*`` use HF nn.Linear params with the default
        # loader.
        return AutoWeightsLoader(self).load_weights(weights)


EntryClass = GlmAsrForConditionalGeneration
