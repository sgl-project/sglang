"""Unlimited-OCR v2 processor with DASH image-mode semantics."""

import logging
from typing import Any, ClassVar, Optional

import torch

from sglang.srt.configs.unlimited_ocr import merge_mm_processor_kwargs
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.unlimited_ocr_v2 import UnlimitedOCRV2ForCausalLM
from sglang.srt.multimodal.processors.unlimited_ocr import (
    _DEFAULT_MODE,
    _IMAGE_MODE_PRESETS,
    UnlimitedOCRProcessor,
)

logger = logging.getLogger(__name__)

# Alias for mm_processor_kwargs={"crop_mode": bool} (vLLM-compatible spelling):
# True selects gundam, False selects base.
_CROP_MODE_ALIAS = {True: "gundam", False: "base"}


def _resolve_mode_v2(images_config) -> dict:
    """Resolve v2 image mode, including vLLM crop-mode aliases."""
    mode = _DEFAULT_MODE
    if images_config:
        mode = images_config.get("image_mode") or _DEFAULT_MODE
        crop = images_config.get("crop_mode")
        if crop is not None and "image_mode" not in images_config:
            mode = _CROP_MODE_ALIAS[bool(crop)]
    key = str(mode).strip().lower()
    if key == "auto":
        key = _DEFAULT_MODE
    preset = _IMAGE_MODE_PRESETS.get(key)
    if preset is None:
        raise ValueError(
            f"Unknown image_mode '{mode}'. "
            f"Supported: auto, {', '.join(_IMAGE_MODE_PRESETS)}"
        )
    return dict(zip(("base_size", "image_size", "crop_mode"), preset))


def _release_slice_storage(value):
    """Clone a tensor view that keeps a larger parent storage alive."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.untyped_storage().nbytes() <= value.numel() * value.element_size():
        return value
    return value.clone()


class UnlimitedOCRV2ProcessorOutput(MultimodalProcessorOutput):
    """Processor output whose crop tiles go over SHM, not inline in the request."""

    shm_model_specific_data: ClassVar[bool] = True


class UnlimitedOCRV2Processor(UnlimitedOCRProcessor):
    """Multimodal processor for UNLIMITED-OCR v2 (DASH) checkpoints."""

    models: ClassVar = [UnlimitedOCRV2ForCausalLM]

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: list[MultimodalDataItem],
        *,
        images: Optional[list[Any]],
    ) -> list[MultimodalDataItem]:
        mm_items = super()._postprocess_mm_items_before_transport(
            mm_items, images=images
        )
        # Each page's tensors are views of the [pages, ...] bundle, and pickling a
        # view serializes the whole bundle: N pages would ship N^2 pages of crops.
        for item in mm_items:
            item.model_specific_data = {
                key: _release_slice_storage(value)
                for key, value in item.model_specific_data.items()
            }
        return mm_items

    async def process_mm_data_async(
        self, image_data: list[str | bytes], input_text, *args, **kwargs
    ):
        """Process multimodal data asynchronously (v2 mode resolution)."""
        request_obj = kwargs.get("request_obj")
        images_config = (
            merge_mm_processor_kwargs(
                request_obj.images_config, request_obj.mm_processor_kwargs
            )
            if request_obj is not None
            else None
        )
        processor_kwargs = _resolve_mode_v2(images_config)

        prefix = images_config.get("prefix", "") if images_config else ""

        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **processor_kwargs, v2_layout=True
        )

        if prefix:
            prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
            input_ids = torch.cat(
                [input_ids, torch.tensor(prefix_ids, dtype=input_ids.dtype)]
            )

        self._mix_config_into_hash(mm_items, processor_kwargs)

        return UnlimitedOCRV2ProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
