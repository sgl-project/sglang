# Copyright 2025 SGLang Team
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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/paligemma.py

"""Inference-only PaliGemma model compatible with HuggingFace weights."""

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PaliGemmaConfig, PreTrainedModel

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class PaliGemmaMultiModalProjector(nn.Module):
    """Simple linear projector from vision hidden size to text hidden size."""

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(PreTrainedModel):
    config_class = PaliGemmaConfig

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        self.vision_tower = SiglipVisionModel(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projection_dim=config.vision_config.projection_dim,
        )

        self.vocab_size = config.text_config.vocab_size

        self.language_model = GemmaForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        if self.language_model.logits_processor.logit_scale:
            self.language_model.logits_processor.logit_scale *= logit_scale

        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: MultimodalInputs
    ) -> List[int]:
        """Pad input IDs by replacing image placeholder tokens with pad_values."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, image_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Process images through vision tower and projector.
        Returns projected image features scaled by hidden_size^(-0.5).
        """
        all_pixel_values = flatten_nested_list([item.feature for item in items])
        final_features_list = []

        for pixel_values_batch in all_pixel_values:
            # Handle precomputed embeddings
            if (
                pixel_values_batch.dim() == 3
                and pixel_values_batch.shape[-1] == self.config.text_config.hidden_size
            ):
                final_features_list.append(
                    pixel_values_batch.to(self.language_model.model.embed_tokens.weight.device)
                )
                continue

            # Normalize input shape to [batch_size, channels, height, width]
            if pixel_values_batch.dim() == 5:
                pixel_values_batch = pixel_values_batch.squeeze(0)
            elif pixel_values_batch.dim() == 3:
                pixel_values_batch = pixel_values_batch.unsqueeze(0)
            elif pixel_values_batch.dim() != 4:
                raise ValueError(
                    f"Unexpected pixel_values shape: {pixel_values_batch.shape}"
                )

            # Process each image through vision tower
            batch_vision_outputs = []
            batch_size = pixel_values_batch.shape[0]

            for i in range(batch_size):
                pixel_value = pixel_values_batch[i : i + 1]
                pixel_value = pixel_value.to(
                    device=self.vision_tower.device,
                    dtype=self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype,
                )
                vision_output = self.vision_tower(pixel_values=pixel_value)
                batch_vision_outputs.append(vision_output)

            if batch_vision_outputs:
                vision_outputs_cat = torch.cat(batch_vision_outputs, dim=0)
                projected_features = self.multi_modal_projector(vision_outputs_cat)
                # Scale by hidden_size^(-0.5) to compensate for the sqrt(hidden_size)
                # scaling applied later in GemmaModel.forward
                projected_features = projected_features * (self.config.hidden_size**-0.5)
                final_features_list.append(projected_features)

        if final_features_list:
            return torch.cat(final_features_list, dim=0)
        else:
            return torch.tensor(
                [], device=self.language_model.model.embed_tokens.weight.device
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> LogitsProcessor:
        # Replace OOV image token IDs with 0 to avoid index errors in embedding lookup.
        # The actual image embeddings are injected by general_mm_embed_routine.
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        hs = general_mm_embed_routine(
            input_ids=llm_input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

        return hs

    def tie_weights(self):
        if hasattr(self.language_model, "tie_weights"):
            return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "language_model" in name:
                # GemmaForCausalLM.load_weights doesn't return loaded_params
                GemmaForCausalLM.load_weights(self, [(name, loaded_weight)])
                loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_model" in name:
                    # Adapt to VisionAttention naming
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


EntryClass = PaliGemmaForConditionalGeneration
