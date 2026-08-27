import math
from typing import List, Union

from transformers import PreTrainedTokenizerBase
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
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
        mm_data = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
        )
        if mm_data.images:
            effective_patch = self.patch_size * self._spatial_merge_size
            image_nrows = []
            for img in mm_data.images:
                w, h = img.size
                ratio = max(w / self.image_size, h / self.image_size)
                if ratio > 1:
                    w = int(math.floor(w / ratio))
                    h = int(math.floor(h / ratio))
                nrows, _ = _get_pixtral_hf_num_image_tokens(
                    (h, w), (effective_patch, effective_patch)
                )
                image_nrows.append(nrows)

            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                mm_data, self.mm_tokens
            )

            # For multi-image: split single IMAGE mm_item into per-image items
            if len(mm_data.images) > 1:
                from sglang.srt.managers.schedule_batch import MultimodalDataItem

                old_item = next(
                    item for item in mm_items if item.modality == Modality.IMAGE
                )
                all_offsets = old_item.offsets
                old_feature = old_item.feature
                old_image_sizes = getattr(old_item, "image_sizes", None)

                mm_items = [
                    item for item in mm_items if item.modality != Modality.IMAGE
                ]
                offset_idx = 0
                for i, img in enumerate(mm_data.images):
                    nr = image_nrows[i]
                    item_offsets = all_offsets[offset_idx : offset_idx + nr]
                    offset_idx += nr
                    new_item = MultimodalDataItem(modality=Modality.IMAGE)
                    new_item.feature = old_feature[i : i + 1]
                    new_item.offsets = item_offsets
                    if old_image_sizes is not None:
                        new_item.model_specific_data["image_sizes"] = old_image_sizes[
                            i : i + 1
                        ]
                    mm_items.append(new_item)
        else:
            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                mm_data, self.mm_tokens
            )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )
