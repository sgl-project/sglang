from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BatchFeature

IMAGE_PREPROCESSING = {
    "patch_size": 40,
    "include_extra_column": False,
    "rescale_image_frac": None,
    "rescale_image_max_upscaled_long_edge": None,
    "stretch_to_padded_grid": False,
    "overlay_rgba_images_on_checkerboard": True,
    "temporal_patch_size": 2,
    "requires_image_grid_thw": True,
}
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)


@dataclass(frozen=True)
class HydratedVisionTokenSpan:
    int_tokens: list[int]
    vision_patches_bthwc: torch.Tensor
    vision_grid_thw: torch.Tensor
    vision_original_thw: torch.Tensor


def _load_image_bytes(image: bytes | str | os.PathLike | Image.Image) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        mode = "RGBA" if image.has_transparency_data else "RGB"
        with image.convert(mode) as pixels:
            pixels.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(image, os.PathLike):
        return Path(image).read_bytes()
    if not isinstance(image, str):
        raise TypeError("Expected image bytes, a PIL image, a path, or an image URL")
    if image.startswith("data:"):
        header, separator, payload = image.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Expected a base64 data URI")
        return base64.b64decode(payload, validate=True)
    if image.startswith(("http://", "https://")):
        with urlopen(image, timeout=30) as response:
            return response.read()
    if "://" in image and not image.startswith("file://"):
        raise ValueError("Unsupported image URL scheme")
    return Path(image.removeprefix("file://")).read_bytes()


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    if image_bytes.startswith(b"\xff\xd8") and not image_bytes.endswith(b"\xff\xd9"):
        image_bytes += b"\xff\xd9"
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if not image.has_transparency_data:
            return np.array(image.convert("RGB"), dtype=np.uint8)
        rgba = np.array(image.convert("RGBA"), dtype=np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    return (
        (rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127) // 255
    ).astype(np.uint8)


class Inkling2ImageProcessor:
    model_input_names = [
        "vision_patches_bthwc",
        "vision_grid_thw",
        "vision_original_thw",
    ]
    patch_size = 40
    temporal_patch_size = 2

    @classmethod
    def from_pretrained(
        cls, path: str | os.PathLike, **kwargs
    ) -> Inkling2ImageProcessor:
        # The file only restates the fixed preprocessing, so checkpoints may omit it.
        if Path(path).is_dir():
            config_path = Path(path) / "image_preprocessing.json"
            if not config_path.exists():
                return cls()
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                config_path = Path(
                    hf_hub_download(str(path), "image_preprocessing.json", **kwargs)
                )
            except EntryNotFoundError:
                return cls()
        with config_path.open() as stream:
            config = json.load(stream)
        if config != IMAGE_PREPROCESSING:
            raise ValueError("Unsupported Inkling2 image preprocessing configuration")
        return cls()

    def name(self) -> str:
        return "blackpadded"

    def count_tokens_for_image(self, width: int, height: int) -> int:
        if width <= 0 or height <= 0:
            return 0
        return ((width + 39) // 40) * ((height + 39) // 40)

    def resolve_virtual_dims(self, dimensions: tuple[int, int]) -> tuple[int, int]:
        return dimensions

    def _encode(
        self,
        image: bytes | str | os.PathLike | Image.Image,
        image_token_id: int,
        expected_size: tuple[int, int] | None = None,
        virtual_size: tuple[int, int] | None = None,
    ) -> HydratedVisionTokenSpan:
        pixels = _decode_rgb(_load_image_bytes(image))
        height, width, _ = pixels.shape
        if expected_size is not None and (width, height) != expected_size:
            raise ValueError(
                f"Image size mismatch: expected {expected_size}, got {(width, height)}"
            )
        if virtual_size is not None:
            if any(type(size) is not int or size <= 0 for size in virtual_size):
                raise ValueError(
                    "Virtual image dimensions must both be positive integers"
                )
            if virtual_size != (width, height):
                from sglang.srt.multimodal.inkling2.resize import resize_lanczos3

                width, height = virtual_size
                pixels = resize_lanczos3(pixels, width, height)
        rows, columns = (height + 39) // 40, (width + 39) // 40
        padded = np.full((rows * 40, columns * 40, 3), -1, dtype=np.float32)
        padded[:height, :width] = pixels
        normalized = (padded / np.float32(255) - _MEAN) / _STD
        patches = (
            torch.from_numpy(normalized)
            .reshape(rows, 40, columns, 40, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(rows * columns, 1, 40, 40, 3)
            .to(torch.bfloat16)
            .expand(-1, 2, -1, -1, -1)
        )
        return HydratedVisionTokenSpan(
            int_tokens=[image_token_id] * (rows * columns),
            vision_patches_bthwc=patches,
            vision_grid_thw=torch.tensor([[1, rows, columns]], dtype=torch.int64),
            vision_original_thw=torch.tensor([[1, height, width]], dtype=torch.int64),
        )

    def preprocess_image_for_training(
        self,
        *,
        location: str,
        size: tuple[int, int],
        num_tokens: int,
        image_token_id: int,
        virtual_size: tuple[int, int] | None = None,
    ) -> HydratedVisionTokenSpan:
        encoded = self._encode(
            location,
            image_token_id,
            expected_size=size,
            virtual_size=virtual_size,
        )
        if len(encoded.int_tokens) != num_tokens:
            raise ValueError(
                "Hydrated image token count disagrees with rendered image span"
            )
        return encoded

    def preprocess(
        self,
        images: bytes | str | os.PathLike | Image.Image | list,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        if return_tensors != "pt":
            raise ValueError("Inkling2 image preprocessing returns PyTorch tensors")
        if not isinstance(images, (list, tuple)):
            images = [images]
        spans = [self._encode(image, 200005) for image in images]
        num_tokens = [len(span.int_tokens) for span in spans]
        data = {
            "vision_patches_bthwc": (
                torch.cat([span.vision_patches_bthwc for span in spans])
                if spans
                else torch.empty((0, 2, 40, 40, 3), dtype=torch.bfloat16)
            ),
            "vision_grid_thw": (
                torch.cat([span.vision_grid_thw for span in spans])
                if spans
                else torch.empty((0, 3), dtype=torch.int64)
            ),
            "vision_original_thw": (
                torch.cat([span.vision_original_thw for span in spans])
                if spans
                else torch.empty((0, 3), dtype=torch.int64)
            ),
            "num_tokens": num_tokens,
        }
        return BatchFeature(data=data, tensor_type=None)

    __call__ = preprocess
