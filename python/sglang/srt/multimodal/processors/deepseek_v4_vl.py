# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DeepSeek-V4 image preprocessing, matching inference/image_processor.py in
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp. Block lengths must match the wrapper's
embeddings, including compression padding and sentinels.
"""

import math
import re
from typing import List, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4_vl import DeepseekV4ForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Sentinel type ids, matching the reference image_processor.py.
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height,
    width,
    best_height,
    best_width,
    patch_size,
    downsample_ratio,
    max_n_token,
):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Builds the N-layout token types (final order) and the aligner-row order
    for IMAGE slots. Returns (types, perm) as python lists."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types.tolist(), perm.tolist()


class DeepseekV4VLImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        image_token_id = self._tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if image_token_id is None or image_token_id == self._tokenizer.unk_token_id:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        self.image_token_id = image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PLACEHOLDER,
            image_token_id=image_token_id,
        ).build(_processor)

        # vision preprocessing hyperparameters from the HF config
        self.patch_size = hf_config.vision_patch_size
        self.downsample_ratio = hf_config.vision_downsample_ratio
        self.max_n_token = hf_config.vision_max_n_token
        self.min_pixels = hf_config.vision_min_pixels
        self.max_wh_ratio = hf_config.vision_max_wh_ratio

    def _load_image(self, image):
        """One image -> (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w).

        Faithful port of image_processor.load_image, minus the byte loading
        (load_mm_data already decoded the image)."""
        p = self.patch_size
        if not isinstance(image, Image.Image):
            # Depending on the image backend / transport, load_mm_data may
            # hand us a CHW uint8 tensor instead of a PIL image.
            arr = torch.as_tensor(image)
            if arr.dim() == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
            image = Image.fromarray(arr.cpu().numpy().astype(np.uint8))
        image = image.convert("RGB")
        width, height = image.size
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = height * self.max_wh_ratio
        if 0 < width * height < self.min_pixels:
            ratio = (self.min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / p) * p
        best_height = math.ceil(height / p) * p
        n_llm_h, n_llm_w, best_height, best_width = safe_resize(
            height,
            width,
            best_height,
            best_width,
            p,
            self.downsample_ratio,
            self.max_n_token,
        )
        n_vit_h, n_vit_w = best_height // p, best_width // p
        if (
            self.max_wh_ratio is not None
            and image.width >= self.max_wh_ratio * image.height
        ):
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image, (best_width, best_height), color=(127, 127, 127)
            )
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            x.reshape(3, n_vit_h, p, n_vit_w, p)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, p, p)
        )
        return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        prompt = base_output.input_text
        images = base_output.images

        text_parts = re.split(re.escape(IMAGE_PLACEHOLDER), prompt)
        if len(text_parts) - 1 != len(images):
            raise ValueError(
                f"Found {len(text_parts) - 1} image placeholders "
                f"but got {len(images)} images"
            )

        input_ids: List[int] = []
        mm_items: List[MultimodalDataItem] = []
        for i, part in enumerate(text_parts):
            if part:
                input_ids.extend(self._tokenizer.encode(part))
            if i >= len(images):
                continue
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self._load_image(images[i])
            # The block layout depends on the absolute position (compress-pad
            # alignment), so it must be computed against the running length.
            types, perm = build_image_block(n_llm_h, n_llm_w, len(input_ids))
            start = len(input_ids)
            input_ids.extend([self.image_token_id] * len(types))
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    # Grid and compression alignment determine types/perm for
                    # this processor. Cached blocks include that layout as well
                    # as pixels, even when two layouts have the same length.
                    hash=hash_feature(
                        [
                            patches,
                            torch.tensor(
                                [n_vit_h, n_vit_w, start % COMPRESS_PAD_TO],
                                dtype=torch.int64,
                            ),
                        ]
                    ),
                    offsets=[(start, len(input_ids) - 1)],
                    model_specific_data={
                        "types": types,
                        "perm": perm,
                        "n_vit_h": n_vit_h,
                        "n_vit_w": n_vit_w,
                    },
                )
            )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids,
            im_token_id=self.image_token_id,
        )
