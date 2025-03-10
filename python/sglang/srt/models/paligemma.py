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
"""PaliGemma - Google's Cutting-Edge Open Vision Language Model. """

import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    PaliGemmaConfig,
    SiglipVisionModel,
)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.utils import add_prefix



class PaliGemmaForConditionalGeneration(nn.module):
    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str ="",
    ) -> None:
        
        super().__init__()
        self.config = config
        self.languag_model = GemmaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)
    
    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        new_input_ids = []
        last_idx = 0
        image_idx = -1
        image_inputs.image_offsets = []

        # Get all special token IDs
        im_start_id = image_inputs.im_start_id
        im_end_id = image_inputs.im_end_id

        # Find all start and end positions for both types
        start_indices = [i for i, x in enumerate(input_ids) if x == im_start_id]
        end_indices = [i for i, x in enumerate(input_ids) if x == im_end_id]

        if len(start_indices) != len(end_indices):
            return input_ids
        # Process each region (both image and slice)
        for start_idx, end_idx in zip(start_indices, end_indices):
            # Add non-image tokens before this region
            new_input_ids.extend(input_ids[last_idx : start_idx + 1])

            is_image_start = input_ids[start_idx] == im_start_id

            if is_image_start:
                image_inputs.image_offsets += [start_idx]
                image_idx += 1

            num_tokens = end_idx - start_idx - 1  # exclude start and end tokens

            # Generate pad_ids
            pad_values = [image_inputs.pad_values[image_idx]]

            pad_ids = pad_values * ((num_tokens + len(pad_values)) // len(pad_values))
            pad_ids = pad_ids[:num_tokens]

            # Add pad_ids
            new_input_ids.extend(pad_ids)

            # Update last_idx to after end token
            last_idx = end_idx

        # Add remaining tokens after last region
        new_input_ids.extend(input_ids[last_idx:])
        assert len(input_ids) == len(new_input_ids)
        return new_input_ids
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs

        if forward_batch.forward_mode.is_extend():
            pass #TODO(xiao)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        vision_path = self.config.mm_vision_tower
    
        self.vision_tower = SiglipVisionModel.from_pretrained(
            vision_path, torch_dtype=torch.float16
            ).cuda()

        self.vision_tower.eval()

        #load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear",
            "model.vision_tower.vision_tower": "vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
        }

        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if "projector" in name or "vision_tower" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        #load language model
        self.language_model.load_weights(weights)

EntryClass = PaliGemmaForConditionalGeneration
    
