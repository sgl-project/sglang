# Copyright 2025 The RedNote HiLab team.
# Copyright 2025 The SGLang team.
#
# This code is based on the DeepseekVL2ForCausalLM and DotsVisionTransformer
# implementation in this library.
#
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
"""Inference-only Dots-VL model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.dots_vlm import DotsVLMConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

from .dots_vlm_vit import DotsVisionTransformer


class DotsVLMForCausalLM(nn.Module):
    """DotsVLM model for sglang inference"""

    def __init__(
        self, config: DotsVLMConfig, quant_config: Optional[QuantizationConfig] = None
    ) -> None:
        super().__init__()

        self.config = config
        self.image_token_id = config.im_span_id
        self.video_token_id = config.video_span_id

        self.language_model = DeepseekV2ForCausalLM(
            config.language_config, quant_config
        )

        # Initialize vision tower (matching transformers naming for weight compatibility)
        self.vision_tower = DotsVisionTransformer(config.vision_config)

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):
        """pad attn qkv weights for dummy heads"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads
        if num_dummy_heads == 0:
            return loaded_weight
        head_dim = self.config.vision_config.head_dim

        if "attn.qkv_proj" in name:
            wq, wk, wv = loaded_weight.chunk(3, dim=0)
            if name.endswith(".weight"):
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]
            elif name.endswith(".bias"):
                dummy_shape = [num_dummy_heads, head_dim]
            else:
                raise RuntimeError(f"Unsupported weight with name={name}")
            pad_func = lambda x: torch.cat(
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
            ).flatten(0, 1)
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
            loaded_weight = torch.cat([wq, wk, wv], dim=0)
        if "attn.proj.weight" in name:
            padded_weight = loaded_weight.new_zeros(
                loaded_weight.shape[0], head_dim * num_dummy_heads
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
        if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
        return loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""
        weights = list(weights)

        # Separate vision tower weights and language model weights
        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            if name.startswith("vision_tower."):
                vision_name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                vision_weights.append((vision_name, loaded_weight))
            else:
                # All other weights go to language model
                language_weights.append((name, loaded_weight))

        # Load vision tower weights
        vision_state_dict = dict(vision_weights)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in vision_state_dict.items():
            if name not in params_dict:
                raise ValueError(f"Weight {name} not found in params_dict")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)
            weight_loader(param, loaded_weight)

        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return DeepseekV2ForCausalLM.get_model_config_for_expert_location(config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input_ids with multimodal tokens"""
        # Get image token ID for padding pattern
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        padded_input_ids = pattern.pad_input_tokens(input_ids, mm_inputs)
        return padded_input_ids

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract pixel values and grid information (following reference pattern)
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_tower.dtype
        )
        image_grid_thw = torch.concat(
            [item.image_grid_thw for item in items], dim=0
        ).to(self.vision_tower.device)

        # Add dimension checks like in reference code
        assert pixel_values.dim() == 2, f"{pixel_values.dim()=}"
        assert image_grid_thw.dim() == 2, f"{image_grid_thw.dim()=}"

        # Process through vision tower
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)

        # Ensure consistent dtype for FlashInfer compatibility
        # Force bfloat16 to match model's expected dtype
        if image_embeds.dtype != torch.bfloat16 and hasattr(
            self.language_model.model, "embed_tokens"
        ):
            target_dtype = self.language_model.model.embed_tokens.weight.dtype
            image_embeds = image_embeds.to(target_dtype)

        return image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.language_model,
        )
        return hidden_states


EntryClass = [DotsVLMForCausalLM]
