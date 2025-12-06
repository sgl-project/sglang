from typing import List, Union

from sglang.srt.models.hunyuan_vl import HunYuanVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


# Compatible with HunYuan VL
class HunYuanImageProcessor(SGLangBaseProcessor):
    models = [HunYuanVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.image_token_id = hf_config.image_token_id
        self.im_start_token_id = hf_config.image_start_token_id
        self.im_end_token_id = hf_config.image_end_token_id

        self.image_config = server_args.mm_process_config.get("image", {})
        self.video_config = server_args.mm_process_config.get("video", {})

        self.vision_config = hf_config.vision_config
        self.spatial_merge_size = self.vision_config.spatial_merge_size

        self.rope_scaling = hf_config.rope_scaling
        if (
            self.rope_scaling is not None
            and self.rope_scaling.get("xdrope_section", None) is not None
        ):
            self.xd_num = len(self.rope_scaling["xdrope_section"])

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<｜hy_place▁holder▁no▁102｜>",
            image_token_id=hf_config.image_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.im_start_token_id,
            "im_end_id": self.im_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "xdrope_positions": ret["position_ids"].squeeze(0),
        }
