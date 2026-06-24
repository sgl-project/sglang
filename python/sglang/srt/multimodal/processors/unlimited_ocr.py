"""Standalone UNLIMITED-OCR processor."""

import hashlib
import logging
import time
from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)

# image_mode -> (base_size, image_size, crop_mode)
_IMAGE_MODE_PRESETS = {
    "tiny": (512, 512, False),
    "small": (640, 640, False),
    "base": (1024, 1024, False),
    "large": (1280, 1280, False),
    "gundam": (1024, 640, True),
}
_DEFAULT_MODE = "gundam"


def _resolve_mode(images_config, num_images: int = 1) -> dict:
    """Return processor kwargs from images_config (or default)."""
    mode = _DEFAULT_MODE
    if images_config:
        mode = images_config.get("image_mode", _DEFAULT_MODE)
    key = mode.strip().lower()
    preset = _IMAGE_MODE_PRESETS.get(key)
    if preset is None:
        logger.error(
            "Unknown image_mode '%s'. Supported: %s",
            mode,
            ", ".join(_IMAGE_MODE_PRESETS),
        )
        raise ValueError(
            f"Unknown image_mode '{mode}'. "
            f"Supported: {', '.join(_IMAGE_MODE_PRESETS)}"
        )
    _MULTI_IMAGE_ALLOWED = ("tiny", "small", "base")
    if num_images > 1 and key not in _MULTI_IMAGE_ALLOWED:
        raise ValueError(
            f"image_mode='{mode}' is not supported with multiple images "
            f"(got {num_images} images). "
            f"Please use one of: {list(_MULTI_IMAGE_ALLOWED)}"
        )
    return dict(zip(("base_size", "image_size", "crop_mode"), preset))


class UnlimitedOCRProcessor(BaseMultimodalProcessor):
    """Multimodal processor for UNLIMITED-OCR model."""

    models = [UnlimitedOCRForCausalLM]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        """Initialize UnlimitedOCRProcessor."""
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    @staticmethod
    def _mix_config_into_hash(mm_items, processor_kwargs):
        """Mix images_config into mm_item hashes so that different configs
        produce different pad_values, avoiding radix/embedding cache collisions."""
        from sglang.srt.managers.mm_utils import hash_feature

        config_bytes = str(sorted(processor_kwargs.items())).encode()
        for item in mm_items:
            if item.feature is not None:
                base_hash = hash_feature(item.feature)
            elif item.precomputed_embeddings is not None:
                base_hash = hash_feature(item.precomputed_embeddings)
            else:
                continue
            combined = hashlib.sha256(
                base_hash.to_bytes(8, byteorder="big") + config_bytes
            ).digest()[:8]
            item.hash = int.from_bytes(combined, byteorder="big", signed=False)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj=None,
        *args,
        **kwargs,
    ):
        """Process multimodal data asynchronously."""
        total_start = time.perf_counter()
        images_config = (
            getattr(request_obj, "images_config", None) if request_obj else None
        )
        processor_kwargs = _resolve_mode(images_config, num_images=len(image_data))

        # Extract prefix from images_config (tokenized separately to avoid merge)
        prefix = images_config.get("prefix", "") if images_config else ""

        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )
        load_done = time.perf_counter()
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, **processor_kwargs
        )
        process_done = time.perf_counter()

        # Append prefix tokens separately so they don't merge with prompt tokens
        prefix_token_count = 0
        if prefix:
            prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
            prefix_token_count = len(prefix_ids)
            input_ids = torch.cat(
                [input_ids, torch.tensor(prefix_ids, dtype=input_ids.dtype)]
            )
        prefix_done = time.perf_counter()

        self._mix_config_into_hash(mm_items, processor_kwargs)
        hash_done = time.perf_counter()

        logger.info(
            "[UnlimitedOCRProcessor] process_mm_data_async timings: "
            "load=%.2f ms, process_combine=%.2f ms, prefix=%.2f ms, "
            "hash=%.2f ms, total=%.2f ms, mode=%s, input_tokens=%d, mm_items=%d",
            (load_done - total_start) * 1000,
            (process_done - load_done) * 1000,
            (prefix_done - process_done) * 1000,
            (hash_done - prefix_done) * 1000,
            (hash_done - total_start) * 1000,
            processor_kwargs,
            int(input_ids.numel()),
            len(mm_items),
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
            prefix_token_count=prefix_token_count,
        )
