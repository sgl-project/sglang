import math
from typing import List, Union

from transformers import PreTrainedTokenizerBase
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

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
    BaseMultiModalProcessorOutput,
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
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            mm_data, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        base_output: BaseMultiModalProcessorOutput,
    ) -> List[MultimodalDataItem]:
        if len(base_output.images) <= 1:
            return mm_items

        old_item = next(item for item in mm_items if item.modality == Modality.IMAGE)
        all_offsets = old_item.offsets
        old_feature = old_item.feature
        old_image_sizes = old_item.model_specific_data.get("image_sizes")
        image_nrows = self._get_image_nrows(base_output.images)

        split_items = [item for item in mm_items if item.modality != Modality.IMAGE]
        offset_idx = 0
        for image_idx, num_rows in enumerate(image_nrows):
            item_offsets = all_offsets[offset_idx : offset_idx + num_rows]
            offset_idx += num_rows
            model_specific_data = {}
            if old_image_sizes is not None:
                model_specific_data["image_sizes"] = old_image_sizes[
                    image_idx : image_idx + 1
                ]
            split_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=old_feature[image_idx : image_idx + 1],
                    offsets=item_offsets,
                    model_specific_data=model_specific_data,
                )
            )
        return split_items

    def _get_image_nrows(self, images) -> List[int]:
        effective_patch = self.patch_size * self._spatial_merge_size
        image_nrows = []
        for image in images:
            width, height = image.size
            ratio = max(width / self.image_size, height / self.image_size)
            if ratio > 1:
                width = int(math.floor(width / ratio))
                height = int(math.floor(height / ratio))
            num_rows, _ = _get_pixtral_hf_num_image_tokens(
                (height, width),
                (effective_patch, effective_patch),
            )
            image_nrows.append(num_rows)
        return image_nrows
