"""HuggingFace-convention image processor for Inkling models."""

from __future__ import annotations

import io
import math
from typing import List, Optional, Union

import numpy as np
import torch
from numba import njit
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD


def _validate_image_rescale(
    rescale_image_frac: Optional[float],
    rescale_image_max_upscaled_long_edge: Optional[int],
) -> None:
    if rescale_image_frac is not None and (
        not math.isfinite(rescale_image_frac) or rescale_image_frac <= 0
    ):
        raise ValueError(
            "rescale_image_frac must be positive and finite or None, "
            f"got {rescale_image_frac}"
        )
    if rescale_image_max_upscaled_long_edge is None:
        return
    if rescale_image_max_upscaled_long_edge <= 0:
        raise ValueError(
            "rescale_image_max_upscaled_long_edge must be positive or None, "
            f"got {rescale_image_max_upscaled_long_edge}"
        )
    if rescale_image_frac is None or rescale_image_frac <= 1.0:
        raise ValueError(
            "rescale_image_max_upscaled_long_edge requires rescale_image_frac > 1, "
            f"got {rescale_image_frac}"
        )


def _scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: Optional[float],
    rescale_image_max_upscaled_long_edge: Optional[int],
) -> tuple[int, int]:
    """Return the long-edge-scaled ``(width, height)``."""
    if rescale_image_frac is None:
        return width, height

    long_edge = max(width, height)
    if long_edge == 0:
        return width, height

    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        # The cap limits growth but never shrinks an image already above it.
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))

    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height

    # Use half-away-from-zero rounding for positive dimensions. Python's round()
    # uses ties-to-even, which can produce a different output size at exact halves.
    def scale(value: int) -> int:
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def _load_image_bytes(image) -> bytes:
    """Coerce a single image input into raw PNG/JPEG bytes for preprocessing."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            raise ValueError(
                "InklingImageProcessor received a URL/data: image; resolve it to bytes "
                "upstream (e.g. via SGLang load_mm_data) before preprocessing."
            )
        path = image[len("file://") :] if image.startswith("file://") else image
        with open(path, "rb") as f:
            return f.read()

    from PIL import Image

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        image = Image.fromarray(arr.astype("uint8") if arr.dtype != np.uint8 else arr)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@njit(cache=True)
def _fill_patches_numba(
    arr: np.ndarray,
    patch_size: int,
    patches: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pad_norm: np.ndarray,
) -> None:
    h = arr.shape[0]
    w = arr.shape[1]
    nph = (h + patch_size - 1) // patch_size
    npw = w // patch_size + 1
    inv255 = np.float32(1.0 / 255.0)

    for k in range(nph * npw):
        i = k // npw
        j = k - i * npw
        y_base = i * patch_size
        x_base = j * patch_size

        for y in range(patch_size):
            iy = y_base + y
            for x in range(patch_size):
                ix = x_base + x
                if iy < h and ix < w:
                    for c in range(3):
                        raw = np.float32(arr[iy, ix, c]) * inv255
                        patches[k, y, x, c] = (raw - mean[c]) / std[c]
                else:
                    for c in range(3):
                        patches[k, y, x, c] = pad_norm[c]


def _encode_image_bytes(
    image_bytes: bytes,
    *,
    patch_size: int,
    rescale_image_frac: Optional[float],
    rescale_image_max_upscaled_long_edge: Optional[int],
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    _validate_image_rescale(
        rescale_image_frac,
        rescale_image_max_upscaled_long_edge,
    )

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scaled_size = _scaled_image_dimensions(
        *image.size,
        rescale_image_frac=rescale_image_frac,
        rescale_image_max_upscaled_long_edge=rescale_image_max_upscaled_long_edge,
    )
    if scaled_size != image.size:
        image = image.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.uint8, copy=True)
    height, width, _ = arr.shape

    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    num_patches = nph * npw

    patches = np.empty((num_patches, patch_size, patch_size, 3), dtype=np.float32)
    _fill_patches_numba(arr, patch_size, patches, IMAGE_MEAN, IMAGE_STD, PAD_NORM)

    return (
        torch.from_numpy(patches)
        .to(torch.bfloat16)
        .view(num_patches, 1, patch_size, patch_size, 3)
        .expand(num_patches, 2, patch_size, patch_size, 3)
    )


class InklingImageProcessor(BaseImageProcessor):
    r"""Turn raw images into ``vision_patches_bthwc`` for Inkling hMLP.

    ``rescale_image_frac`` scales the long edge while preserving aspect ratio.
    ``rescale_image_max_upscaled_long_edge`` optionally caps only upscaling and
    therefore requires a scale factor greater than one. The defaults, ``2.0`` and
    ``2048``, grow images toward a 2048-pixel long edge by at most 2x, while leaving
    images already at or above 2048 unchanged.
    """

    model_input_names = ["vision_patches_bthwc"]

    def __init__(
        self,
        patch_size: int = 40,
        rescale_image_frac: Optional[float] = 2.0,
        rescale_image_max_upscaled_long_edge: Optional[int] = 2048,
        **kwargs,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        _validate_image_rescale(
            rescale_image_frac,
            rescale_image_max_upscaled_long_edge,
        )
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge

    def _encode_one(self, image) -> torch.Tensor:
        return _encode_image_bytes(
            _load_image_bytes(image),
            patch_size=self.patch_size,
            rescale_image_frac=self.rescale_image_frac,
            rescale_image_max_upscaled_long_edge=self.rescale_image_max_upscaled_long_edge,
        )

    def preprocess(
        self,
        images: Union[ImageInput, List],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        del return_tensors, kwargs
        if not isinstance(images, (list, tuple)):
            images = [images]

        per_image_patches: List[torch.Tensor] = []
        num_patches: List[int] = []
        num_tokens: List[int] = []
        for img in images:
            vp = self._encode_one(img)
            n_patches = int(vp.shape[0])
            per_image_patches.append(vp)
            num_patches.append(n_patches)
            num_tokens.append(n_patches)

        if len(per_image_patches) == 1:
            vision_patches_bthwc = per_image_patches[0]
        elif per_image_patches:
            vision_patches_bthwc = torch.cat(per_image_patches, dim=0)
        else:
            vision_patches_bthwc = torch.empty(0)

        data = {
            "vision_patches_bthwc": vision_patches_bthwc,
            "num_patches": num_patches,
            "num_tokens": num_tokens,
        }
        return BatchFeature(data=data, tensor_type=None)
