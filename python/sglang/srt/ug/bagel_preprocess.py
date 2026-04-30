# SPDX-License-Identifier: Apache-2.0

"""Local BAGEL preprocessing utilities used by the SRT-native UG path."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

BAGEL_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


def add_bagel_special_tokens(tokenizer: Any) -> tuple[Any, dict[str, int], int]:
    all_special_tokens = []
    for value in tokenizer.special_tokens_map.values():
        if isinstance(value, str):
            all_special_tokens.append(value)
        elif isinstance(value, list):
            all_special_tokens.extend(value)

    new_tokens = [
        token for token in BAGEL_SPECIAL_TOKENS if token not in all_special_tokens
    ]
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    return (
        tokenizer,
        {
            "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        },
        num_new_tokens,
    )


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    p = patch_size
    channels, height, width = image.shape
    if height % p != 0 or width % p != 0:
        raise ValueError(
            f"BAGEL patchify expects image size divisible by {p}, got {(height, width)}"
        )
    image = image.reshape(channels, height // p, p, width // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, p**2 * channels)


def get_flattened_position_ids_extrapolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
) -> torch.Tensor:
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    return (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()


def get_flattened_position_ids_interpolate(
    img_h: int,
    img_w: int,
    patch_size: int,
    max_num_patches_per_side: int,
) -> torch.Tensor:
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    boundaries = torch.arange(
        1 / max_num_patches_per_side,
        1.0,
        1 / max_num_patches_per_side,
    )
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    return (
        bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w
    ).flatten()


def pil_img2rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        return white
    return image.convert("RGB")


class BAGELMaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value: float) -> int:
        return max(self.stride, int(round(value / self.stride) * self.stride))

    def _apply_scale(self, width: int, height: int, scale: float) -> tuple[int, int]:
        return (
            self._make_divisible(round(width * scale)),
            self._make_divisible(round(height * scale)),
        )

    def forward(self, img, img_num: int = 1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(
            img,
            (new_height, new_width),
            self.interpolation,
            antialias=self.antialias,
        )


class BAGELImageTransform:
    def __init__(
        self,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        image_mean = image_mean or [0.5, 0.5, 0.5]
        image_std = image_std or [0.5, 0.5, 0.5]
        self.stride = image_stride
        self.resize_transform = BAGELMaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean,
            std=image_std,
            inplace=True,
        )

    def __call__(self, img, img_num: int = 1):
        img = pil_img2rgb(img)
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        return self.normalize_transform(img)
