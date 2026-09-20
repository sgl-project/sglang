# Copyright 2023-2026 SGLang Team
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

import itertools
import json
import math
import os
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs, MultiModalData, ProcessorMixin
from transformers.utils import TensorType
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.utils.hub import cached_file

PROCESSOR_CONFIG_NAME = "processor_config.json"


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_tokens: int,
) -> tuple[int, int]:
    """Patch grid closest to the aspect ratio; returns (height, width) in pixels."""
    ideal_patches_height = height / patch_size
    ideal_patches_width = width / patch_size
    ratio = (
        ideal_patches_width / ideal_patches_height if ideal_patches_height > 0 else 1.0
    )
    if ideal_patches_height * ideal_patches_width > max_tokens:
        ideal_patches_height = (max_tokens / ratio) ** 0.5
        ideal_patches_width = ideal_patches_height * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(ideal_patches_height), math.ceil(ideal_patches_height)],
                [math.floor(ideal_patches_width), math.ceil(ideal_patches_width)],
            )
        )
    )
    candidates = [
        (patches_height, patches_width)
        for patches_height, patches_width in candidates
        if patches_height >= 1
        and patches_width >= 1
        and patches_height * patches_width <= max_tokens
    ]
    if not candidates:
        candidates = [
            (max(1, round(ideal_patches_height)), max(1, round(ideal_patches_width)))
        ]
    patches_height, patches_width = min(
        candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width)
    )
    return patches_height * patch_size, patches_width * patch_size


class MuseGlimmerImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_image_tokens: int


class MuseGlimmerImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.LANCZOS
    size = None
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_image_tokens = 4096
    valid_kwargs = MuseGlimmerImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        return_tensors: Optional[TensorType],
        patch_size: int,
        temporal_patch_size: int,
        max_image_tokens: int,
        merge_size: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if resample == PILImageResampling.LANCZOS:
            # BICUBIC stands in for LANCZOS, which is CPU-only.
            resample = PILImageResampling.BICUBIC

        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = get_aspect_ratio_preserving_size(
                    height=height,
                    width=width,
                    patch_size=patch_size * merge_size,
                    max_tokens=max_image_tokens,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                    antialias=True,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                temporal_patch_size * channel * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Patch rows a (height, width) image expands to."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        resized_height, resized_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_image_tokens,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)

    def _validate_preprocess_kwargs(self, **kwargs):
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)


class MuseGlimmerProcessor(ProcessorMixin):
    """Expands one ``<|patch|>`` placeholder into an image's patch-token run."""

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|patch|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        revision = kwargs.pop("revision", None)
        use_fast = kwargs.pop("use_fast", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_fast=use_fast,
        )
        image_processor = MuseGlimmerImageProcessor(
            **_load_image_processor_kwargs(pretrained_model_name_or_path, revision)
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = (
            image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        )
        return self.image_token * num_image_tokens

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Placeholder counts per image, without running the image processor."""
        vision_data = {}
        if image_sizes is not None:
            merge_size = self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, kwargs)
                for height, width in image_sizes
            ]
            vision_data.update(
                num_image_tokens=[
                    patches // merge_size**2 for patches in num_image_patches
                ],
                num_image_patches=num_image_patches,
            )
        return MultiModalData(**vision_data)


def _load_image_processor_kwargs(model_path: str, revision: Optional[str]) -> dict:
    """Read the image_processor block from processor_config.json."""
    local = os.path.join(model_path, PROCESSOR_CONFIG_NAME)
    config_file = (
        local
        if os.path.isfile(local)
        else cached_file(model_path, PROCESSOR_CONFIG_NAME, revision=revision)
    )
    with open(config_file) as f:
        config = json.load(f)
    image_processor_config = config.get("image_processor", {})
    return {
        key: value
        for key, value in image_processor_config.items()
        if key != "image_processor_type"
    }
