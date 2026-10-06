"""In-process vision/audio towers for dots.note.omni."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.srt.models.dots3_common.dots_omni_audio import (
    OmniAudioConfig,
    OmniAudioModel,
    compute_audio_token_length,
)
from sglang.srt.models.dots3_common.dots_omni_vision import (
    DotsMoEVitConfig,
    DotsMoEVitModel,
)


def _read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def load_omni_component_config(model_dir: Path, component: str) -> dict:
    """Read a tower config from a flat dots.note.omni publish."""
    config = _read_json(model_dir / "config.json")
    nested_name = f"{component}_config"
    if nested_name not in config:
        raise KeyError(f"Missing {nested_name!r} in {model_dir / 'config.json'}")
    return config[nested_name]


class DotsNoteOmniVisionEncoder(DotsMoEVitModel):
    """Native MoE ViT used by dots.note.omni."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        config = DotsMoEVitConfig(**load_omni_component_config(model_dir, "vision"))
        super().__init__(config)
        self.to(torch.bfloat16)

    def load_converted_state(self, state: dict[str, torch.Tensor]):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Dots vision tower missing weights: {missing[:8]}")
        if unexpected:
            raise RuntimeError(
                f"Dots vision tower has unexpected weights: {unexpected[:8]}"
            )


class DotsNoteOmniAudioEncoder(OmniAudioModel):
    """Native Dots speech encoder and adapter."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        config = OmniAudioConfig(**load_omni_component_config(model_dir, "audio"))
        super().__init__(config)
        self.to(torch.bfloat16)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def load_converted_state(self, state: dict[str, torch.Tensor]):
        missing, unexpected = self.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Dots audio tower weight mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )


class DotsNoteOmniImagePreprocessor:
    """CPU image preprocessing matching the converted native ViT."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        config = _read_json(model_dir / "preprocessor_config.json")
        config = config["vision_config"]
        self.min_pixels = config["min_pixels"]
        self.max_pixels = config["max_pixels"]
        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.merge_size = config["merge_size"]
        self.pre_pixel_shuffle = config.get("pre_pixel_shuffle", True)
        self.image_mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.image_std = np.asarray(config["image_std"], dtype=np.float32)
        image_detail_path = model_dir / "image_detail.json"
        self.image_detail_config = (
            _read_json(image_detail_path).get("image_details", {})
            if image_detail_path.is_file()
            else {}
        )

    @staticmethod
    def _round_by_factor(value: int, factor: int) -> int:
        return round(value / factor) * factor

    @staticmethod
    def _ceil_by_factor(value: float, factor: int) -> int:
        return math.ceil(value / factor) * factor

    @staticmethod
    def _floor_by_factor(value: float, factor: int) -> int:
        return math.floor(value / factor) * factor

    def _resized_size(
        self,
        width: int,
        height: int,
        min_pixels: int,
        max_pixels: int,
        target_height=None,
        target_width=None,
    ):
        height = target_height or height
        width = target_width or width
        factor = self.patch_size * self.merge_size
        if min(height, width) < factor // 4:
            raise ValueError(
                f"Image height/width must be at least {factor // 4}, "
                f"got {height}x{width}"
            )
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Image aspect ratio must be smaller than 200")
        resized_h = max(factor, self._round_by_factor(height, factor))
        resized_w = max(factor, self._round_by_factor(width, factor))
        if resized_h * resized_w > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            resized_h = max(factor, self._floor_by_factor(height / beta, factor))
            resized_w = max(factor, self._floor_by_factor(width / beta, factor))
        elif resized_h * resized_w < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_h = self._ceil_by_factor(height * beta, factor)
            resized_w = self._ceil_by_factor(width * beta, factor)
            if resized_h * resized_w > max_pixels:
                beta = math.sqrt(resized_h * resized_w / max_pixels)
                resized_h = max(factor, self._floor_by_factor(resized_h / beta, factor))
                resized_w = max(factor, self._floor_by_factor(resized_w / beta, factor))
        return resized_h, resized_w

    def _process_image(self, image, detail="auto"):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image)}")
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        detail_config = self.image_detail_config.get(detail, {})
        resized_h, resized_w = self._resized_size(
            *image.size,
            min_pixels=detail_config.get("min_pixels", self.min_pixels),
            max_pixels=detail_config.get("max_pixels", self.max_pixels),
            target_height=detail_config.get("target_height"),
            target_width=detail_config.get("target_width"),
        )
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.image_mean) / self.image_std
        patches = array.transpose(2, 0, 1)[None]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        if self.pre_pixel_shuffle:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        else:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        pixel_values = torch.from_numpy(
            patches.reshape(
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
        )
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]]),
        }

    def _get_image_token_str(self, token_count: int):
        return "<|img|>" + "<|imgpad|>" * token_count + "<|endofimg|>"

    def process_images(self, images: Iterable, details=None):
        images = list(images)
        details = details or ["auto"] * len(images)
        pixel_values = []
        grids = []
        token_strings = []
        for image, detail in zip(images, details):
            processed = self._process_image(image, detail)
            grid = processed["image_grid_thw"]
            token_count = int(grid.prod().item()) // self.merge_size**2
            pixel_values.append(processed["pixel_values"])
            grids.append(grid)
            token_strings.append(self._get_image_token_str(token_count))
        return pixel_values, grids, token_strings


def get_audio_token_string(num_samples: int, config: OmniAudioConfig) -> str:
    count = compute_audio_token_length(
        num_samples,
        chunk_seconds=config.chunk_seconds,
        conv_temporal_stride=config.conv_temporal_stride,
        merge_factor=config.merge_factor,
    )
    return (
        config.audio_comp_start + config.audio_comp_span * count + config.audio_comp_end
    )
