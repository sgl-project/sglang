"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Inference-only LLaVa video model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import CLIPVisionModel, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from python.sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata


class LlavaVidForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.mm_spatial_pool_stride = getattr(self.config, "mm_spatial_pool_stride", 2)
        self.resampler = nn.AvgPool2d(
            kernel_size=self.mm_spatial_pool_stride, stride=self.mm_spatial_pool_stride
        )
        self.language_model = LlamaForCausalLM(config, quant_config=quant_config)
        self.num_frames = getattr(self.config, "num_frames", 16)
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )

    def pad_input_ids(
        self,
        input_ids: List[int],
        pad_value: List[int],
        pixel_values: List,
        image_sizes: List[List[int]],
    ):
        new_image_feature_len = self.image_feature_len

        pad_ids = pad_value * (
            (new_image_feature_len + len(pad_value)) // len(pad_value)
        )
        offset = input_ids.index(self.config.image_token_index)
        # old_len + pad_len - 1, because we need to remove image_token_id
        new_input_ids = (
            input_ids[:offset]
            + pad_ids[:new_image_feature_len]
            + input_ids[offset + 1 :]
        )
        return new_input_ids, [offset]

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
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

        height = width = self.num_patches_per_side
        num_of_frames = selected_image_feature.shape[0]
        selected_image_feature = selected_image_feature.view(
            num_of_frames, height, width, -1
        )
        selected_image_feature = selected_image_feature.permute(0, 3, 1, 2).contiguous()
        selected_image_feature = (
            self.resampler(selected_image_feature)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        image_features = self.multi_modal_projector(selected_image_feature)

        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        pixel_values: Optional[List[Optional[np.array]]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if input_metadata.forward_mode == ForwardMode.EXTEND:
            bs = input_metadata.batch_size

            # Embed text inputs
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Whether the requests need vision inputs
            max_image_offset = np.array(
                [max(image_offsets[i]) if image_offsets[i] else -1 for i in range(bs)]
            )
            start_positions = positions[input_metadata.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= max_image_offset

            if need_vision.any():
                pixel_values = [pixel_values[i] for i in range(bs) if need_vision[i]]

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:
                    # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
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
                    # image_features: BS, 576, 4096

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    new_image_features.append(image_feature.flatten(0, 1))
                image_features = new_image_features

                # Fill in the placeholder for the image
                extend_start_loc_cpu = input_metadata.extend_start_loc.cpu().numpy()
                prefix_lens_cpu = input_metadata.extend_prefix_lens.cpu().numpy()
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    prefix_len = prefix_lens_cpu[i]

                    # Multiple images
                    for image_offset in image_offsets[i]:
                        if image_offset < prefix_len:
                            continue

                        tmp_image_feature = image_features[pt]
                        pad_len = tmp_image_feature.shape[0]

                        left_idx = start_idx + (image_offset - prefix_len)
                        right_idx = start_idx + (image_offset - prefix_len) + pad_len
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
                input_ids, positions, input_metadata, input_embeds=input_embeds
            )
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.language_model(input_ids, positions, input_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Load clip vision model by cfg['mm_vision_tower']:
        # huggingface_name or path_of_clip_relative_to_llava_model_dir
        # We put the initialization here instead of __init__ to allow it being reused by other subclasses.
        vision_path = self.config.mm_vision_tower
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_path, torch_dtype=torch.float16
        ).cuda()
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        print(f"target_frames: {self.num_frames}")
        self.image_feature_len = self.num_frames * int(
            (self.image_size / self.patch_size / self.mm_spatial_pool_stride) ** 2
        )
        if self.vision_feature_select_strategy == "patch":
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_resampler.mm_projector.0": "multi_modal_projector.linear_1",
            "model.vision_resampler.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
        }
        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            # FIXME: why projector weights read two times?
            if "projector" in name or "vision_tower" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                else:
                    print(f"Warning: {name} not found in the model")
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # load language model
        self.language_model.load_weights(weights)

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


EntryClass = LlavaVidForCausalLM
