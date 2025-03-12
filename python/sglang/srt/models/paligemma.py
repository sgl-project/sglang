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
    SiglipVisionModel
)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.utils import add_prefix


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str ="",
    ) -> None:
        
        super().__init__()
        self.config = config
        self.language_model = GemmaForCausalLM(
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

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.last_hidden_state #Note: from transformers     
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs

        if forward_batch.forward_mode.is_extend():
            bs = forward_batch.batch_size

            # Clamp input ids. See llava.py for more details
            input_ids.clamp__(0, self.config.vocab_size - 1)

            #Embed text inputs 
            input_embeds = self.language_model.model.embed_tokens(input_ids)
        
            #TODO(xiao)
            image_inputs = [ img for img in forward_batch.image_inputs if img is not None]

            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

            # Whether the requests need vision inputs
            max_image_offset = []
            for im in image_inputs:
                if im and im.image_offsets:
                    max_image_offset.append(max(im.image_offsets))
                else:
                    max_image_offset.append(-1)
            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)

            if need_vision.any():
                pixel_values = [
                    image_inputs[i].pixel_values for i in range(bs) if need_vision[i]
                ]
                image_offsets = [
                    image_inputs[i].image_offsets for i in range(bs) if need_vision[i]
                ]

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:
                    # : BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
                    np.concatenate(pixel_values, axis=0)
                    # ndim=4
                    concat_images = torch.tensor(
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    # image_features = self.encode_images(concat_images)
                    # split_sizes = [image.shape[0] for image in pixel_values]
                    # image_features = torch.split(image_features, split_sizes, dim=0)
                    image_features = self.encode_images(
                        concat_images
                    )  # , prompts)#, image_counts, long_video=long_video)
                    split_sizes = [image.shape[0] for image in pixel_values]
                    image_features = torch.split(image_features, split_sizes, dim=0)

                    # hd image_features: BS, num_patch, 576, 4096
                else:
                    # normal pixel: BS, C=3, H=336, W=336
                    pixel_values = torch.tensor(
                        np.array(pixel_values), device=self.vision_tower.device
                    )
                    image_features = self.encode_images(pixel_values)

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    new_image_features.append(image_feature.flatten(0, 1))
                image_features = new_image_features
            

                # Fill in the placeholder for the image
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    seq_len = extend_seq_lens[i]
                    prefix_len = prefix_lens_cpu[i]

                    # Multiple images
                    for image_offset in image_offsets[i]:
                        if image_offset < prefix_len:
                            continue

                        tmp_image_feature = image_features[pt]
                        pad_len = tmp_image_feature.shape[0]

                        input_offset  = image_offset - prefix_len
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
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.config._name_or_path, 
            torch_dtype=torch.float16
        ).to("cuda")

        self.vision_tower.eval()

        #load mm_projector
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
                self.language_model.load_weights([(name, loaded_weight)])

EntryClass = PaliGemmaForConditionalGeneration
    
