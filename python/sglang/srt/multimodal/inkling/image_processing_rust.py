from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput

from sglang.srt.multimodal._core import inkling as _rs
from sglang.srt.multimodal.inkling.image_processing import _load_image_bytes


def _bits_to_bthwc(
    bits: np.ndarray, height: int, width: int, patch_size: int
) -> torch.Tensor:
    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    n = nph * npw
    return (
        torch.from_numpy(bits)
        .view(torch.bfloat16)
        .view(n, 1, patch_size, patch_size, 3)
        .expand(n, 2, patch_size, patch_size, 3)
    )


_pil_pool = ThreadPoolExecutor(max_workers=8)


def _pil_decode(raw: bytes) -> np.ndarray:
    return np.ascontiguousarray(np.array(Image.open(io.BytesIO(raw)).convert("RGB")))


class InklingRustImageProcessor(BaseImageProcessor):
    model_input_names = ["vision_patches_bthwc"]

    def __init__(
        self,
        patch_size: int = 40,
        rescale_image_frac: Optional[float] = 2.0,
        rescale_image_max_upscaled_long_edge: Optional[int] = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge

    def preprocess(
        self,
        images: Union[ImageInput, List],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        del return_tensors, kwargs
        if not isinstance(images, (list, tuple)):
            images = [images]

        raw_list = [_load_image_bytes(img) for img in images]

        # PIL releases the GIL while decoding.
        arrays = list(_pil_pool.map(_pil_decode, raw_list))

        per_image_patches: List[torch.Tensor] = []
        num_patches: List[int] = []
        num_tokens: List[int] = []
        content_hashes: List[int] = []

        for arr, raw in zip(arrays, raw_list):
            h, w, bits, content_hash = _rs.rescale_patchify_hash(
                arr,
                raw,
                self.patch_size,
                self.rescale_image_frac,
                self.rescale_image_max_upscaled_long_edge,
            )
            vp = _bits_to_bthwc(bits, h, w, self.patch_size)
            n_patches = int(vp.shape[0])
            per_image_patches.append(vp)
            num_patches.append(n_patches)
            num_tokens.append(n_patches)
            content_hashes.append(content_hash)

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
            "content_hashes": content_hashes,
        }
        return BatchFeature(data=data, tensor_type=None)
