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
import logging

import numpy as np
import torch
from torch import nn
from transformers import PaliGemmaConfig, SiglipVisionModel
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaMultiModalProjector,
)


from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.utils import add_prefix
from sglang.srt.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)

logger = logging.getLogger(__name__)

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:

        super().__init__()
        self.config = config
        self.language_model = GemmaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)

        self.vision_tower = SiglipVisionModel(self.config.vision_config)
    
    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        if not isinstance(image_inputs.im_start_id, list) or not isinstance(
            image_inputs.im_end_id, list
        ):
            return input_ids

        new_input_ids = []
        last_idx = 0
        image_idx = -1
        image_inputs.image_offsets = []

        # Get all special token IDs
        im_start_id = (
            image_inputs.im_start_id[0].item()
            if isinstance(image_inputs.im_start_id[0], torch.Tensor)
            else image_inputs.im_start_id[0]
        )
        im_end_id = (
            image_inputs.im_end_id[0].item()
            if isinstance(image_inputs.im_end_id[0], torch.Tensor)
            else image_inputs.im_end_id[0]
        )
        slice_start_id = (
            image_inputs.slice_start_id[0].item()
            if isinstance(image_inputs.slice_start_id[0], torch.Tensor)
            else image_inputs.slice_start_id[0]
        )
        slice_end_id = (
            image_inputs.slice_end_id[0].item()
            if isinstance(image_inputs.slice_end_id[0], torch.Tensor)
            else image_inputs.slice_end_id[0]
        )

        # Find all start and end positions for both types
        start_indices = [
            i
            for i, x in enumerate(input_ids)
            if x == im_start_id or x == slice_start_id
        ]
        end_indices = [
            i for i, x in enumerate(input_ids) if x == im_end_id or x == slice_end_id
        ]

        if len(start_indices) != len(end_indices):
            return input_ids
        # Process each region (both image and slice)
        for start_idx, end_idx in zip(start_indices, end_indices):
            # Add non-image tokens before this region
            new_input_ids.extend(
                input_ids[last_idx : start_idx + 1]
            )  # include start token

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

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature =image_outputs #.last_hidden_state # Note: from transformers
        image_features = self.multi_model_projector(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs
        logger.info(f"1 paligemma::forward and inputs_ids.shape:{input_ids.shape}")
        if forward_batch.forward_mode.is_extend():
            bs = forward_batch.batch_size

            # Clamp input ids. See llava.py for more details
            input_ids.clamp_(0, self.config._vocab_size - 1)

            # Embed text inputs
            input_embeds = self.language_model.model.embed_tokens(input_ids)

 
                        # Got List[List[str]] extend it to List[str]
            # The length of the List should be equal to batch size

            pixel_values = [image_inputs[i].pixel_values  for i in range(bs) if image_inputs[i] is not None]
            logger.info(f"2 paligemma::forward and pixel_values:{len(pixel_values)}")
            pixel_values = torch.tensor(
                    np.array(pixel_values), device=self.vision_tower.device)
            logger.info(f"3 paligemma::forward and pixel_values.shape:{pixel_values.shape}")
            image_features = self.encode_images(pixel_values)
            logger.info(f"4 paligemma::forward and image_features.shape:{image_features.shape}")
            # Fill in the placeholder for the image
            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
            pt = 0
            for i in range(bs):
                if image_inputs[i] is None:
                    continue
                start_idx = extend_start_loc_cpu[i]
                seq_len = extend_seq_lens[i]
                prefix_len = prefix_lens_cpu[i]

                # Multiple images
                for image_idx, image_offset in enumerate(
                    image_inputs[i].image_offsets
                ):
                    if (
                        image_offset + image_inputs[i].image_pad_len[image_idx]
                        <= prefix_len
                    ):
                        continue
                    if image_offset >= prefix_len + seq_len:
                        break

                    tmp_image_feature = image_features[pt][image_idx]
                    pad_len = tmp_image_feature.shape[0]

                    input_offset = image_offset - prefix_len
                    left_idx = start_idx + input_offset
                    right_idx = left_idx + pad_len
                    assert right_idx > start_idx
                    if input_offset < 0:
                        left_idx = start_idx
                        tmp_image_feature = tmp_image_feature[-input_offset:]
                    if right_idx > start_idx + seq_len:
                        tmp_image_feature = tmp_image_feature[
                            : start_idx + seq_len - right_idx
                        ]
                        right_idx = start_idx + seq_len
                    try:
                        input_embeds[left_idx:right_idx] = tmp_image_feature
                    except RuntimeError as e:
                        print(f"RuntimeError in image encoding: {e}")
                        print(f"{input_embeds.shape=}, {tmp_image_feature.shape=}")
                        print(
                            f"{start_idx=}, {image_offset=}, {prefix_len=}, {pad_len=}"
                        )
                pt += 1
            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # load mm_projector
        projector_weights = {
            "multi_modal_projector.linear": "multi_model_projector.linear",
        }

        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if "projector" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif "language_model" in name:
                if "language_model." in name:  # load model weight from Paligemma
                    name = name.replace("language_model.", "")
                self.language_model.load_weights([(name, loaded_weight)])
            elif "vision_tower" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

EntryClass = PaliGemmaForConditionalGeneration
