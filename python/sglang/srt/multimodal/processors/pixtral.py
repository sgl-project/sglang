import copy
from typing import Any, List, Optional, Union

from transformers import PreTrainedTokenizerBase

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.pixtral import (
    PixtralForConditionalGeneration,
    PixtralVisionModel,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class PixtralProcessor(BaseMultimodalProcessor):
    models = [PixtralVisionModel, PixtralForConditionalGeneration]
    gpu_image_decode = False  # Pixtral processes loaded image as PIL image explicitly

    PAD_TOKEN = "<pad>"
    DEFAULT_IMAGE_TOKEN = "[IMG]"

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IM_TOKEN_ID = getattr(
            hf_config, "image_token_index", PixtralVisionModel.DEFAULT_IMAGE_TOKEN_ID
        )

        self.vision_config = hf_config.vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        # spatial_merge_size may live on vision_config (Mistral native) or
        # on the top-level config (HF native Mistral3Config).
        self._spatial_merge_size = getattr(
            self.vision_config,
            "spatial_merge_size",
            getattr(hf_config, "spatial_merge_size", 1),
        )
        self._effective_patch_size = self.patch_size * self._spatial_merge_size

        self._processor.patch_size = self.patch_size
        if self._spatial_merge_size > 1:
            self._processor.spatial_merge_size = self._spatial_merge_size

        tokenizer = (
            _processor
            if isinstance(_processor, PreTrainedTokenizerBase)
            else _processor.tokenizer
        )
        self.image_token = getattr(_processor, "image_token", self.DEFAULT_IMAGE_TOKEN)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)
        tokenizer.add_special_tokens(
            {
                "pad_token": getattr(hf_config, "pad_token", self.PAD_TOKEN),
            }
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        mm_data = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data, self.mm_tokens, **self._request_processor_kwargs(request_obj)
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )

    def _request_processor_kwargs(self, request_obj) -> dict:
        """Per-request image resizing, e.g. ``{"size": {"longest_edge": 1024}}``."""
        mm_processor_kwargs = getattr(request_obj, "mm_processor_kwargs", None)
        if not mm_processor_kwargs:
            return {}
        size = mm_processor_kwargs.get("size")
        if set(mm_processor_kwargs) != {"size"} or not isinstance(size, dict):
            raise ValueError(
                "Pixtral mm_processor_kwargs supports only "
                '{"size": {"longest_edge": N}}'
            )
        longest_edge = size.get("longest_edge")
        # The vision tower's rotary table covers image_size pixels per side.
        if (
            set(size) != {"longest_edge"}
            or not isinstance(longest_edge, int)
            or isinstance(longest_edge, bool)
            or not 0 < longest_edge <= self.image_size
        ):
            raise ValueError(
                f"longest_edge must be an integer in [1, {self.image_size}], got {size}"
            )
        return {"images_kwargs": {"size": {"longest_edge": longest_edge}}}

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        if not images or len(images) <= 1:
            return mm_items

        image_items = [item for item in mm_items if item.modality == Modality.IMAGE]
        if len(image_items) == len(images):
            return mm_items
        if len(image_items) != 1:
            raise ValueError(
                "Pixtral multi-image processing expected one bundled IMAGE item or "
                f"{len(images)} split items, but found {len(image_items)}"
            )

        old_item = image_items[0]
        all_offsets = old_item.offsets
        old_feature = old_item.feature
        old_image_sizes = old_item.model_specific_data.get("image_sizes")
        if old_image_sizes is None:
            raise ValueError("Pixtral multi-image processing requires image_sizes")
        image_nrows = self._get_image_nrows(old_image_sizes)
        if old_feature is None or len(old_feature) != len(image_nrows):
            raise ValueError(
                "Pixtral multi-image feature count does not match the number of "
                f"images: features={0 if old_feature is None else len(old_feature)}, "
                f"images={len(image_nrows)}"
            )
        if all_offsets is None or sum(image_nrows) != len(all_offsets):
            raise ValueError(
                "Pixtral image patch rows do not match the computed offsets: "
                f"rows={sum(image_nrows)}, "
                f"offsets={0 if all_offsets is None else len(all_offsets)}"
            )

        split_items = [item for item in mm_items if item.modality != Modality.IMAGE]
        offset_idx = 0
        for image_idx, num_rows in enumerate(image_nrows):
            item_offsets = all_offsets[offset_idx : offset_idx + num_rows]
            offset_idx += num_rows
            new_item = copy.copy(old_item)
            new_item.feature = old_feature[image_idx : image_idx + 1]
            new_item.offsets = item_offsets
            new_item.model_specific_data = copy.copy(old_item.model_specific_data)
            if old_image_sizes is not None:
                new_item.model_specific_data["image_sizes"] = old_image_sizes[
                    image_idx : image_idx + 1
                ]
            new_item.hash = None
            new_item.pad_value = None
            split_items.append(new_item)
        if offset_idx != len(all_offsets):
            raise ValueError(
                "Pixtral multi-image split did not consume every offset: "
                f"consumed={offset_idx}, offsets={len(all_offsets)}"
            )
        return split_items

    def _get_image_nrows(self, image_sizes) -> List[int]:
        # Rows come from the processed (resized) image, which honors the
        # processor's longest_edge rather than vision_config.image_size.
        return [int(height) // self._effective_patch_size for height, _ in image_sizes]
