"""Kimi-specific grid-based multimodal data helpers.

Shared by KimiVLImageProcessor and KimiK2_5VLImageProcessor.
"""

import hashlib
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.cache import (
    CONTENT_HASH_PREFIX,
    build_artifact_key,
    build_processor_fingerprint,
    parse_content_hash,
    snapshot_media,
)

_KIMI_IMAGE_IDENTITY_TAG = b"sglang-kimi-image-identity-v1"
_CALLER_IDENTITY_KEYS = ("hash", "pad_value", "pad_values", "identity")


def kimi_image_identity(content_config_digest: str, grid_thw: Sequence[int]) -> bytes:
    """Full SHA-256 image identity: content and preprocessing config, then grid."""
    digest = parse_content_hash(content_config_digest)
    if digest is None:
        raise ValueError("Kimi image identity requires a content digest")
    hasher = hashlib.sha256(_KIMI_IMAGE_IDENTITY_TAG)
    hasher.update(bytes.fromhex(digest[len(CONTENT_HASH_PREFIX) :]))
    for dim in grid_thw:
        hasher.update(int(dim).to_bytes(8, byteorder="big", signed=True))
    return hasher.digest()


def _item_grid_thw(item: MultimodalDataItem) -> List[int]:
    grid = item.model_specific_data.get("image_grid_thw")
    if grid is None:
        raise ValueError("Kimi image item is missing image_grid_thw")
    values = torch.as_tensor(grid).reshape(-1).tolist()
    if len(values) != 3:
        raise ValueError(f"Kimi image item needs one [t, h, w] grid, got {values}")
    return values


class KimiGridMMDataMixin:
    """Mixin providing Kimi-specific grid-based multimodal data helpers.

    Expects the concrete class to supply:
      - self.hf_config  (with vision_config.merge_kernel_size)
      - self._tokenizer (with .encode())
    """

    # Opt-in: processors whose image spans use full identities and wide pads.
    uses_wide_image_identity = False

    @staticmethod
    def reject_caller_image_identity(image_data, request_obj) -> None:
        """Image identities must come from this processor, never from callers."""
        if isinstance(request_obj, GenerateReqInput) and request_obj.mm_hashes:
            raise ValueError("Caller-supplied mm_hashes are not supported for Kimi.")
        for item in image_data or []:
            if isinstance(item, Mapping):
                for key in _CALLER_IDENTITY_KEYS:
                    if key in item:
                        raise ValueError(
                            f"Caller-supplied multimodal {key} is not supported "
                            "for Kimi."
                        )

    def _kimi_config_fingerprint(self) -> str:
        fingerprint = self.processor_fingerprint
        if fingerprint is None:
            fingerprint = build_processor_fingerprint(self, self.hf_config)
            self.processor_fingerprint = fingerprint
        return fingerprint

    def kimi_content_config_digest(self, media: Any) -> str:
        return build_artifact_key(
            snapshot_media(media).content_digest,
            modality="image",
            processor_fingerprint=self._kimi_config_fingerprint(),
        )

    def assign_kimi_image_identities(
        self, mm_items: List[MultimodalDataItem], images: Optional[List[Any]]
    ) -> None:
        """Give each image item a full content+config+grid identity.

        Source images identify processor-computed items; preprocessed or
        precomputed inputs are identified by their own tensor contents.
        """
        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            return
        image_items = [item for item in mm_items if item.is_image()]
        use_sources = (
            images is not None
            and len(images) == len(image_items)
            and all(item.format == MultimodalInputFormat.NORMAL for item in image_items)
        )
        for index, item in enumerate(image_items):
            if use_sources:
                source = images[index]
            elif item.feature is not None:
                source = item.feature
            else:
                source = item.precomputed_embeddings
            item.set_identity(
                kimi_image_identity(
                    self.kimi_content_config_digest(source), _item_grid_thw(item)
                )
            )

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        mm_items = super()._postprocess_mm_items_before_transport(
            mm_items, images=images
        )
        if self.uses_wide_image_identity:
            self.assign_kimi_image_identities(mm_items, images)
        return mm_items

    def resolve_image_token_counts(self, images):
        """Kimi's processor is remote-code and does not implement the
        transformers ``_get_num_multimodal_tokens`` convention; use its
        ``media_tokens_calculator`` instead.

        """
        assert images is not None
        media_tokens_calculator = (
            self._processor.media_processor.media_tokens_calculator
        )
        return [
            int(media_tokens_calculator({"type": "image", "image": image}))
            for image in images
        ]

    @staticmethod
    def count_image_placeholders(input_ids, image_token_id: int) -> Optional[int]:
        """Structural image tokens in a pre-tokenized prompt, None if it is text."""
        if not isinstance(input_ids, (list, torch.Tensor)):
            return None

        token_ids = np.asarray(
            (
                input_ids.detach().flatten().cpu()
                if isinstance(input_ids, torch.Tensor)
                else input_ids
            ),
            dtype=np.int64,
        )
        return int(np.count_nonzero(token_ids == image_token_id))

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
