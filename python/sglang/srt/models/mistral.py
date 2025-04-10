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
"""Inference-only Mistral model."""

from typing import List, Union

import torch
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.models.llama import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    pass


class Mistral3ForConditionalGeneration:
    MULTIMODAL_PROJECTOR_TYPE = Mistral3MultiModalProjector

    def __init__(self, **kwargs):
        # lazy load inner class
        # to bypass circular import
        from sglang.srt.models.llava import LlavaForConditionalGeneration

        # override config: mistral's projector adds patchmerger that doesn't require padding
        kwargs["config"].vision_config.pad_image_border = False

        self.inner = LlavaForConditionalGeneration(**kwargs)
        self.inner.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(
            kwargs["config"]
        )
        self.inner.encode_images = self.encode_images

    def encode_images(
        self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        encode images by vision tower and multimodal projector
        Args:
            pixel_values: torch.Tensor or List[torch.Tensor]: each tensor for an input image
        Returns:
            torch.Tensor: encoded image features from the input image; if multiple, flattened by seq_len axis
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.

        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ["default", "patch"]:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )
        image_sizes = [p.shape[-2:] for p in pixel_values]
        image_features = self.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes
        )
        return image_features

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __hasattr__(self, name):
        return hasattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


EntryClass = [MistralForCausalLM, Mistral3ForConditionalGeneration]
