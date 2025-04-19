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

from sglang.srt.managers.schedule_batch import MultimodalDataItem
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
        self.inner.get_image_feature = self.get_image_feature

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs.

        Args:
            items: List of MultimodalDataItem objects containing image data
                Note that an item can be either "image" or "multi-images"

        Returns:
            torch.Tensor: features from image inputs, concatenated
        """
        features = []
        for item in items:
            # in each item, we assume pixel_values is always batched
            pixel_values, image_sizes = item.pixel_values, item.image_sizes
            image_outputs = self.vision_tower(
                pixel_values, image_sizes, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[
                self.vision_feature_layer
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"
                )
            features.append(
                self.multi_modal_projector(
                    selected_image_feature.squeeze(0), image_sizes
                )
            )
        ret = torch.cat(features, dim=0)
        return ret

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __hasattr__(self, name):
        return hasattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


EntryClass = [MistralForCausalLM, Mistral3ForConditionalGeneration]
