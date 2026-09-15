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
from sglang.srt.model_loader.weight_utils import default_weight_loader
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
        # The processor emits mel features as a batch of fixed-size 30s windows
        # (``(num_chunks, num_mel_bins, 3000)``) plus an ``input_features_mask``
        # marking the valid frames per window; long clips span several windows.
        # The projector stacks ``merge_factor`` encoder frames into one
        # embedding, so we reshape per-window to
        # ``(num_chunks, -1, intermediate_size)`` and then keep only the valid
        # embeddings per window (derived from the mask via the encoder's
        # conv-subsampling + merge downsampling). This keeps the emitted count
        # aligned with the audio placeholder tokens the processor inserted,
        # mirroring the HF ``CohereAsr``/``GlmAsr`` ``get_audio_features`` path.
        audio_config = self.config.audio_config
        merge_factor = audio_config.intermediate_size // audio_config.hidden_size

        audio_embeds_list = []
        for item in items:
            input_features = item.feature.type(self.audio_tower.dtype)
            input_features_mask = item.input_features_mask
            if input_features_mask.dim() == 1:
                input_features_mask = input_features_mask.unsqueeze(0)
            input_features_mask = input_features_mask.to(input_features.device)

            hidden_states = self.audio_tower(input_features).last_hidden_state

            # The frame axis is not guaranteed to be a multiple of
            # ``merge_factor`` (e.g. a short unpadded clip yields an odd frame
            # count), so right-pad it before stacking ``merge_factor`` frames
            # per embedding. Padded rows are trimmed back per-window below via
            # ``post_lengths``, so the extra frames never reach the LM.
            num_frames = hidden_states.shape[1]
            pad = (-num_frames) % merge_factor
            if pad:
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad))
            hidden_states = hidden_states.reshape(
                input_features.shape[0], -1, audio_config.intermediate_size
            )
            embeds = self.multi_modal_projector(hidden_states)

            # Valid embeddings per window: mel frames -> conv subsampling
            # (stride 1 then stride 2) -> merge_factor downsampling.
            audio_lengths = input_features_mask.sum(-1)
            for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
                audio_lengths = (
                    audio_lengths + 2 * padding - (kernel_size - 1) - 1
                ) // stride + 1
            post_lengths = (audio_lengths - merge_factor) // merge_factor + 1

            valid = (
                torch.arange(embeds.shape[1], device=post_lengths.device)[None, :]
                < post_lengths[:, None]
            )
            audio_embeds_list.append(embeds[valid.to(embeds.device)])

        return torch.cat(audio_embeds_list, dim=0)

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


EntryClass = GlmAsrForConditionalGeneration
