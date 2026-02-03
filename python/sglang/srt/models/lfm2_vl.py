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
"""Inference-only LFM2-VL model compatible with HuggingFace weights."""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.auto.modeling_auto import AutoModel

from sglang.srt.configs.lfm2_vl import Lfm2VlConfig
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
from sglang.srt.models.lfm2 import Lfm2ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Lfm2VlMultiModalProjector(nn.Module):
    """Multimodal projector with pixel unshuffle downsampling."""

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.factor = config.downsample_factor
        self.use_layer_norm = config.projector_use_layernorm
        self.layer_norm = (
            nn.LayerNorm(in_channels) if config.projector_use_layernorm else None
        )
        self.linear_1 = nn.Linear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, image_features: torch.Tensor):
        image_features = self.pixel_unshuffle(image_features)
        if self.use_layer_norm:
            image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_unshuffle(self, hidden_states: torch.Tensor):
        batch_size, width, height, channels = hidden_states.size()
        hidden_states = hidden_states.reshape(
            batch_size, width, height // self.factor, channels * self.factor
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size,
            height // self.factor,
            width // self.factor,
            channels * self.factor**2,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states


class Lfm2VlForConditionalGeneration(PreTrainedModel):
    config_class = Lfm2VlConfig

    def __init__(
        self,
        config: Lfm2VlConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        # Vision tower: SigLip2 via HF AutoModel
        self.vision_tower = AutoModel.from_config(config=config.vision_config)

        # Multimodal projector
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)

        # Language model: reuse sglang's LFM2 implementation
        self.language_model = Lfm2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)
        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        result = pattern.pad_input_tokens(input_ids, mm_inputs)
        return result

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process images through vision tower and projector.

        Handles SigLip2's NaFlex variable-resolution output by unpadding
        features using the attention mask and reshaping per spatial_shapes.
        """
        all_pixel_values = flatten_nested_list([item.feature for item in items])
        all_pixel_attention_masks = flatten_nested_list(
            [item.pixel_attention_mask for item in items]
        )
        all_spatial_shapes = flatten_nested_list(
            [item.spatial_shapes for item in items]
        )

        image_features_list = []

        for pixel_values_batch, attn_mask_batch, shapes_batch in zip(
            all_pixel_values, all_pixel_attention_masks, all_spatial_shapes
        ):
            # Normalize shapes
            if pixel_values_batch.dim() == 2:
                pixel_values_batch = pixel_values_batch.unsqueeze(0)
            if attn_mask_batch.dim() == 1:
                attn_mask_batch = attn_mask_batch.unsqueeze(0)
            if shapes_batch.dim() == 1:
                shapes_batch = shapes_batch.unsqueeze(0)

            pixel_values_batch = pixel_values_batch.to(
                device=self.vision_tower.device,
                dtype=self.vision_tower.dtype,
            )
            attn_mask_batch = attn_mask_batch.to(device=self.vision_tower.device)
            shapes_batch = shapes_batch.to(device=self.vision_tower.device)

            # Forward through SigLip2 vision tower
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values_batch,
                spatial_shapes=shapes_batch,
                pixel_attention_mask=attn_mask_batch,
                return_dict=True,
            )
            last_hidden_state = vision_outputs.last_hidden_state

            # Unpad and project each image
            img_feature_lengths = attn_mask_batch.sum(dim=1)
            batch_size = last_hidden_state.size(0)

            for img_idx in range(batch_size):
                feature = last_hidden_state[img_idx]
                # Unpad: keep only non-padded tokens
                feat_len = img_feature_lengths[img_idx].item()
                feature = feature[:feat_len, :].unsqueeze(0)

                # Reshape to spatial dimensions (1, H, W, C)
                h, w = shapes_batch[img_idx].tolist()
                feature = feature.reshape(1, int(h), int(w), -1)

                # Project through multimodal projector
                img_embedding = self.multi_modal_projector(feature)

                # Flatten to (num_tokens, hidden_size)
                img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
                image_features_list.append(img_embedding)

        if image_features_list:
            return torch.cat(image_features_list, dim=0)

        return torch.tensor(
            [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Collect weights by destination
        vision_weights = []
        projector_weights = []
        lm_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model.vision_tower."):
                # model.vision_tower.* → vision_tower.*
                new_name = name.replace("model.vision_tower.", "vision_tower.", 1)
                vision_weights.append((new_name, loaded_weight))
            elif name.startswith("model.multi_modal_projector."):
                # model.multi_modal_projector.* → multi_modal_projector.*
                new_name = name.replace(
                    "model.multi_modal_projector.", "multi_modal_projector.", 1
                )
                projector_weights.append((new_name, loaded_weight))
            elif name.startswith("model.language_model."):
                # model.language_model.* → language_model.model.*
                new_name = name.replace(
                    "model.language_model.", "language_model.model.", 1
                )
                lm_weights.append((new_name, loaded_weight))
            elif name.startswith("lm_head."):
                # lm_head.* → language_model.lm_head.*
                new_name = name.replace("lm_head.", "language_model.lm_head.", 1)
                lm_weights.append((new_name, loaded_weight))
            else:
                # Try direct mapping
                lm_weights.append((name, loaded_weight))

        params_dict = dict(self.named_parameters())

        # Load vision tower weights
        for name, loaded_weight in vision_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # Load projector weights
        for name, loaded_weight in projector_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # Load language model weights via Lfm2ForCausalLM.load_weights
        # Strip the "language_model." prefix since Lfm2ForCausalLM expects
        # names like "model.layers.0..." and "lm_head.weight"
        lm_weights_stripped = []
        for name, loaded_weight in lm_weights:
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            lm_weights_stripped.append((name, loaded_weight))
        self.language_model.load_weights(lm_weights_stripped)


EntryClass = Lfm2VlForConditionalGeneration
