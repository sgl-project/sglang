"""sglang multimodal processor for Mage-VL."""
import re
from typing import List, Union

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.mage_vl import (
    MageVLForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class MageVLImageProcessor(SGLangBaseProcessor):
    """Multimodal preprocessor for Mage-VL in sglang.

    Key differences from OneVision 1.5:
      * No ``smart_resize`` (the HF processor invokes Qwen2VLImageProcessor
        internally, which already handles resize).
      * Registers ``patch_positions`` in ``ATTR_NAME_TO_MODALITY`` so the
        base class threads it into ``MultimodalDataItem.model_specific_data``
        for ``MageVLForConditionalGeneration.get_image_feature``.
      * mrope is qwen2_vl-style (matches OneVision 1.5).
    """

    models = [MageVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Mage-VL inherits the Qwen3 vision-token IDs.
        self.IM_START_TOKEN_ID = getattr(hf_config, "vision_start_token_id", 151652)
        self.IM_END_TOKEN_ID = getattr(hf_config, "vision_end_token_id", 151653)
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        # The expanded image-token block (vision_start + image_pad + vision_end).
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>",
        )

        # NB: This mutation must come AFTER super().__init__() — the base class
        # constructs ATTR_NAME_TO_MODALITY freshly during its __init__. Our
        # extension keys "patch_positions" -> Modality.IMAGE so the base
        # collect_mm_items_from_processor_output() (in process_and_combine_mm_data)
        # automatically carries Mage-VL's per-patch (t,h,w) positions into
        # MultimodalDataItem.model_specific_data. Task 7's get_image_feature
        # then reads item.patch_positions via MultimodalDataItem.__getattr__.
        self.ATTR_NAME_TO_MODALITY["patch_positions"] = Modality.IMAGE

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
            video_token_id=self.VIDEO_TOKEN_ID,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        # The HF processor handles PIL directly via Qwen2VLImageProcessor;
        # no smart_resize step is needed here.
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens,
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.IM_START_TOKEN_ID,
            model_type="qwen2_vl",
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None,
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=getattr(ret, "second_per_grid_ts", None),
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            mrope_positions=mrope_positions.squeeze(1),
            mrope_position_delta=mrope_position_delta,
        )