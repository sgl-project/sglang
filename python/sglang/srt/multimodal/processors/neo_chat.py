# SPDX-License-Identifier: Apache-2.0
"""Native-resolution multimodal processor for SenseNova U1."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from PIL import Image
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.neo_chat import NEOChatModel
from sglang.srt.models.neo_chat_vision import build_abs_positions_from_grid_hw
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute image aspect ratio must be smaller than 200")
    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, _floor_by_factor(height / scale, factor))
        resized_width = max(factor, _floor_by_factor(width / scale, factor))
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * scale, factor)
        resized_width = _ceil_by_factor(width * scale, factor)
    return resized_height, resized_width


def load_image_native(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: float,
    min_pixels: int,
    max_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    factor = int(patch_size // downsample_ratio)
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    pixel_values = transform(image).float()
    channels, height, width = pixel_values.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    pixel_values = (
        pixel_values.view(
            channels,
            grid_height,
            patch_size,
            grid_width,
            patch_size,
        )
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_height * grid_width, channels * patch_size**2)
    )
    return pixel_values, torch.tensor([[grid_height, grid_width]], dtype=torch.long)


def build_u1_mrope_positions(
    input_ids: torch.Tensor,
    *,
    img_start_token_id: int,
    img_context_token_id: int,
    grid_hw: torch.Tensor | None,
    downsample_ratio: float,
    future_decode_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = input_ids.flatten()
    image_start_shift = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            (input_ids == img_start_token_id).long(),
        ]
    )[:-1]
    not_image_token = (input_ids != img_context_token_id).long()
    t_indexes = (image_start_shift + not_image_token).cumsum(0) - 1
    h_indexes = torch.zeros_like(t_indexes)
    w_indexes = torch.zeros_like(t_indexes)

    if grid_hw is not None:
        merge_size = int(1 / downsample_ratio)
        abs_x, abs_y = build_abs_positions_from_grid_hw(grid_hw // merge_size)
        selected = input_ids == img_context_token_id
        if int(selected.sum().item()) != abs_x.numel():
            raise ValueError("image token count does not match grid geometry")
        h_indexes[selected] = abs_y
        w_indexes[selected] = abs_x

    positions = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
    delta = positions.max().reshape(1) + 1 - input_ids.numel()
    if future_decode_tokens > 0:
        future_t = torch.arange(
            int(t_indexes[-1].item()) + 1,
            int(t_indexes[-1].item()) + future_decode_tokens + 1,
            dtype=torch.long,
        )
        future_positions = torch.stack(
            [
                future_t,
                torch.zeros_like(future_t),
                torch.zeros_like(future_t),
            ],
            dim=0,
        )
        positions = torch.cat([positions, future_positions], dim=1)
    return positions, delta


class NEOChatMultimodalProcessor(BaseMultimodalProcessor):
    models: ClassVar[list[type[NEOChatModel]]] = [NEOChatModel]
    gpu_image_decode = False

    IMAGE_PLACEHOLDER = "<image>"
    IMAGE_START = "<img>"
    IMAGE_CONTEXT = "<IMG_CONTEXT>"
    IMAGE_END = "</img>"

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.tokenizer = self._tokenizer
        self.patch_size = hf_config.patch_size
        self.downsample_ratio = hf_config.downsample_ratio
        self.min_pixels = int(
            self.image_config.get("min_pixels", hf_config.vision_config.min_pixels)
        )
        self.max_pixels = int(
            self.image_config.get("max_pixels", hf_config.vision_config.max_pixels)
        )
        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_START)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMAGE_CONTEXT
        )
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_END)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_PLACEHOLDER,
            image_token_id=self.img_context_token_id,
        ).build(processor)

    @staticmethod
    def _ensure_image_placeholders(prompt: str, image_count: int) -> str:
        missing = image_count - prompt.count(
            NEOChatMultimodalProcessor.IMAGE_PLACEHOLDER
        )
        if missing <= 0:
            return prompt
        placeholders = "\n".join(
            [NEOChatMultimodalProcessor.IMAGE_PLACEHOLDER] * missing
        )
        assistant_marker = "<|im_start|>assistant"
        marker_index = prompt.rfind(assistant_marker)
        if marker_index == -1:
            return f"{placeholders}\n{prompt}"
        return f"{prompt[:marker_index]}{placeholders}\n{prompt[marker_index:]}"

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        del args, kwargs
        prompt = self._ensure_image_placeholders(
            input_text or "",
            len(image_data or []),
        )
        base_output = await self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        processed_images: list[tuple[torch.Tensor, torch.Tensor]] = []
        for image in base_output.images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL image, got {type(image).__name__}")
            processed_images.append(
                load_image_native(
                    image,
                    patch_size=self.patch_size,
                    downsample_ratio=self.downsample_ratio,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
            )

        expanded_prompt = base_output.input_text or prompt
        for _, grid_hw in processed_images:
            image_token_count = int(
                grid_hw[0, 0] * grid_hw[0, 1] * self.downsample_ratio**2
            )
            image_tokens = (
                self.IMAGE_START
                + self.IMAGE_CONTEXT * image_token_count
                + self.IMAGE_END
            )
            expanded_prompt = expanded_prompt.replace(
                self.IMAGE_PLACEHOLDER,
                image_tokens,
                1,
            )

        input_ids_tensor = self.tokenizer(
            expanded_prompt,
            return_tensors="pt",
        )["input_ids"].flatten()
        offsets = self.get_mm_items_offset(
            input_ids_tensor,
            self.img_context_token_id,
        )
        if len(offsets) != len(processed_images):
            raise ValueError("image offsets do not match processed image count")

        mm_items = []
        grid_rows = []
        for (pixel_values, grid_hw), offset in zip(
            processed_images,
            offsets,
            strict=True,
        ):
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=pixel_values,
                    offsets=[offset],
                    model_specific_data={"grid_hw": grid_hw},
                )
            )
            grid_rows.append(grid_hw)

        combined_grid_hw = torch.cat(grid_rows, dim=0) if grid_rows else None
        sampling_params = request_obj.sampling_params or {}
        if isinstance(sampling_params, dict):
            max_new_tokens = sampling_params.get("max_new_tokens")
        else:
            max_new_tokens = sampling_params.max_new_tokens
        max_new_tokens = 2048 if max_new_tokens is None else int(max_new_tokens)
        max_new_tokens = min(max(max_new_tokens, 1), 2048)
        mrope_positions, mrope_position_delta = build_u1_mrope_positions(
            input_ids_tensor,
            img_start_token_id=self.img_start_token_id,
            img_context_token_id=self.img_context_token_id,
            grid_hw=combined_grid_hw,
            downsample_ratio=self.downsample_ratio,
            future_decode_tokens=max_new_tokens + 1,
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids_tensor.tolist(),
            mm_items=mm_items,
            im_token_id=self.img_context_token_id,
            im_start_id=self.img_start_token_id,
            im_end_id=self.img_end_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )


__all__ = [
    "NEOChatMultimodalProcessor",
    "build_u1_mrope_positions",
    "load_image_native",
    "smart_resize",
]
