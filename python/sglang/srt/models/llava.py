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

"""Inference-only LLaVa model compatible with HuggingFace weights."""

import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    LlavaConfig,
    MistralConfig,
    Qwen2Config,
    SiglipVisionModel,
)
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.mistral import MistralForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


class LlavaBaseForCausalLM(nn.Module):
    def pad_input_ids(
        self,
        input_ids: List[int],
        pad_value: List[int],
        pixel_values: List,
        image_sizes: List[List[int]],
    ):
        # hardcode for spatial_unpad + anyres
        image_aspect_ratio = "anyres" if len(image_sizes) == 1 else "pad"
        offset_list = []
        for image_s in image_sizes:
            if len(image_sizes) > 16:
                # 2x2 pooling with stride 2
                new_image_feature_len = (
                    math.ceil(self.image_size / self.patch_size / 2) ** 2
                )
            else:
                new_image_feature_len = self.image_feature_len  # multiimage

            height = width = self.num_patches_per_side
            if "anyres" in image_aspect_ratio:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_s,
                    self.image_grid_pinpoints,
                    self.vision_tower.config.image_size,
                )
                h = num_patch_height * height
                w = num_patch_width * width
                new_h, new_w = unpad_image_shape(h, w, image_s)

                if "anyres_max" in self.config.image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(
                        r"anyres_max_(\d+)", self.config.image_aspect_ratio
                    )
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                    # times = math.sqrt(h * w / (max_num_patches * unit**2))
                    times = math.sqrt(
                        new_h * new_w / (max_num_patches * self.image_feature_len)
                    )
                    if times > 1.1:
                        new_h = int(new_h // times)
                        new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)

            pad_ids = pad_value * (
                (new_image_feature_len + len(pad_value)) // len(pad_value)
            )
            # print("calculated new_image_feature_len: ", new_image_feature_len)
            try:
                offset = input_ids.index(self.config.image_token_index)
            except ValueError:
                offset = 0
            # old_len + pad_len - 1, because we need to remove image_token_id
            input_ids = (
                input_ids[:offset]
                + pad_ids[:new_image_feature_len]
                + input_ids[offset + 1 :]
            )
            offset_list.append(offset)
        return input_ids, offset_list

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
                image_sizes = [image_sizes[i] for i in range(bs) if need_vision[i]]

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:
                    # llava-hd: BS, num_patch, C=3, H=336, W=336, num_patch obtained from process_images
                    np.concatenate(pixel_values, axis=0)
                    # ndim=4
                    concat_images = torch.tensor(
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    image_features = self.encode_images(concat_images)
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

                if self.mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    height = width = self.num_patches_per_side
                    for image_idx, image_feature in enumerate(image_features):
                        if len(image_sizes[image_idx]) == 1:
                            image_aspect_ratio = (
                                self.config.image_aspect_ratio
                            )  # single image
                        else:
                            image_aspect_ratio = "pad"  # multi image
                        # image_aspect_ratio = (
                        #     "anyres" if len(image_sizes[image_idx]) == 1 else "pad"
                        # )
                        if (
                            image_feature.shape[0] > 1
                            and "anyres" in image_aspect_ratio
                        ):
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            assert height * width == base_image_feature.shape[0]

                            if "anyres_max" in image_aspect_ratio:
                                matched_anyres_max_num_patches = re.match(
                                    r"anyres_max_(\d+)", image_aspect_ratio
                                )
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(
                                        matched_anyres_max_num_patches.group(1)
                                    )

                            if (
                                image_aspect_ratio == "anyres"
                                or "anyres_max" in image_aspect_ratio
                            ):
                                vision_tower_image_size = self.image_size
                                try:
                                    num_patch_width, num_patch_height = (
                                        get_anyres_image_grid_shape(
                                            image_sizes[image_idx][0],
                                            self.config.image_grid_pinpoints,
                                            vision_tower_image_size,
                                        )
                                    )
                                except Exception as e:
                                    print(f"Error: {e}")
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(
                                    num_patch_height, num_patch_width, height, width, -1
                                )
                            else:
                                image_feature = image_feature.view(
                                    2, 2, height, width, -1
                                )

                            # (
                            #     num_patch_width,
                            #     num_patch_height,
                            # ) = get_anyres_image_grid_shape(
                            #     image_sizes[image_idx][0],
                            #     self.image_grid_pinpoints,
                            #     self.vision_tower.config.image_size,
                            # )

                            # image_feature = image_feature.view(
                            #     num_patch_height, num_patch_width, height, width, -1
                            # )

                            if "unpad" in self.mm_patch_merge_type:
                                unit = image_feature.shape[2]
                                image_feature = image_feature.permute(
                                    4, 0, 2, 1, 3
                                ).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(
                                    2, 3
                                )
                                image_feature = unpad_image(
                                    image_feature, image_sizes[image_idx][0]
                                )
                                if (
                                    "anyres_max" in image_aspect_ratio
                                    and matched_anyres_max_num_patches
                                ):
                                    c, h, w = image_feature.shape
                                    times = math.sqrt(
                                        h * w / (max_num_patches * unit**2)
                                    )
                                    if times > 1.1:
                                        image_feature = image_feature[None]
                                        image_feature = nn.functional.interpolate(
                                            image_feature,
                                            [int(h // times), int(w // times)],
                                            mode="bilinear",
                                        )[0]
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.language_model.model.image_newline[
                                            :, None, None
                                        ].expand(*image_feature.shape[:-1], 1),
                                    ),
                                    dim=-1,
                                )
                                image_feature = image_feature.flatten(1, 2).transpose(
                                    0, 1
                                )
                            else:
                                image_feature = image_feature.permute(
                                    0, 2, 1, 3, 4
                                ).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat(
                                (base_image_feature, image_feature), dim=0
                            )
                            image_feature = image_feature.unsqueeze(0)
                        else:
                            if image_feature.shape[0] > 16:  # video
                                # 2x2 pooling
                                num_of_frames = image_feature.shape[0]
                                image_feature = image_feature.view(
                                    num_of_frames, height, width, -1
                                )
                                image_feature = image_feature.permute(
                                    0, 3, 1, 2
                                ).contiguous()  # N, C, H, W
                                height, weight = image_feature.shape[2:]
                                scaled_shape = [
                                    math.ceil(height / 2),
                                    math.ceil(weight / 2),
                                ]
                                image_feature = nn.functional.interpolate(
                                    image_feature, size=scaled_shape, mode="bilinear"
                                )
                                image_feature = (
                                    image_feature.flatten(2)
                                    .transpose(1, 2)
                                    .contiguous()
                                )  # N, C, H*W

                        new_image_features.append(image_feature)
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
                    for j, image_offset in enumerate(image_offsets[i]):
                        if image_offset < prefix_len:
                            continue

                        tmp_image_feature = image_features[pt][j]
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
        if "clip" in vision_path:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
        elif "siglip" in vision_path:
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
            # Siglip needs all feature tokens
            self.config.mm_vision_select_feature = "full"
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
            "model.image_newline": "language_model.model.image_newline",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "projector" in name or "vision_tower" in name or "image_newline" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


class LlavaLlamaForCausalLM(LlavaBaseForCausalLM):
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
        self.language_model = LlamaForCausalLM(config, quant_config=quant_config)
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaQwenForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 151646

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(config, quant_config=quant_config)
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaMistralForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = MistralConfig(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 32000

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = MistralForCausalLM(config, quant_config=quant_config)
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


EntryClass = [LlavaLlamaForCausalLM, LlavaQwenForCausalLM, LlavaMistralForCausalLM]
