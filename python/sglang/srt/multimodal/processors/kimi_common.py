"""Kimi-specific grid-based multimodal data helpers.

Shared by KimiVLImageProcessor and KimiK2_5VLImageProcessor.
"""

from typing import Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)


class KimiGridMMDataMixin:
    """Mixin providing Kimi-specific grid-based multimodal data helpers.

    Expects the concrete class to supply:
      - self.hf_config  (with vision_config.merge_kernel_size)
      - self._tokenizer (with .encode())
    """

    def _num_image_tokens_from_grid(
        self, grid_thw: Union[torch.Tensor, np.ndarray, list, tuple]
    ) -> int:
        """Compute Kimi-style image token count from 2D/3D grid metadata."""
        merge_h, merge_w = self.hf_config.vision_config.merge_kernel_size

        if isinstance(grid_thw, torch.Tensor):
            vals = grid_thw.flatten().tolist()
        elif isinstance(grid_thw, np.ndarray):
            vals = grid_thw.reshape(-1).tolist()
        elif isinstance(grid_thw, (list, tuple)):
            vals = list(np.array(grid_thw).reshape(-1).tolist())
        else:
            raise TypeError(
                f"Unsupported grid type for kimi image tokens: {type(grid_thw)}"
            )

        if len(vals) >= 3:
            _t, h, w = vals[-3], vals[-2], vals[-1]
        elif len(vals) == 2:
            _t, h, w = 1, vals[0], vals[1]
        else:
            raise ValueError(
                f"Invalid grid metadata for kimi image tokens: {vals} "
                "(expected [t,h,w] or [h,w])"
            )

        h, w = int(h), int(w)
        return (h * w) // (merge_h * merge_w)

    def _build_kimi_mm_data_from_grids(
        self, prompt, embeddings, **kwargs
    ) -> MultimodalProcessorOutput:
        image_token_id = kwargs.get("image_token_id", 0)
        img_grid_thw = kwargs.get("img_grid_thw", None)

        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        image_token_counts = [
            self._num_image_tokens_from_grid(grid) for grid in img_grid_thw
        ]

        input_ids = []
        offsets = []
        img_idx = 0

        for token in prompt:
            if token != image_token_id:
                input_ids.append(token)
                continue

            if img_idx >= len(image_token_counts):
                raise ValueError(
                    "The number of image placeholders exceeds img_grid_thw entries."
                )

            num_tokens = image_token_counts[img_idx]
            start = len(input_ids)
            input_ids.extend([image_token_id] * num_tokens)
            offsets.append((start, len(input_ids) - 1))
            img_idx += 1

        if img_idx != len(image_token_counts):
            raise ValueError(
                "The number of image placeholders does not match img_grid_thw entries."
            )

        image_embeddings = embeddings[Modality.IMAGE]
        mm_items = []
        consumed = 0
        for start, end in offsets:
            num_tokens = end - start + 1
            embedding_slice = image_embeddings[consumed : consumed + num_tokens]
            consumed += num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=image_token_id,
        )
