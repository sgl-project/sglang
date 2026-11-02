import re
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3_fast.py
# will be removed in the future


class Gemma3SGLangImageProcessor(SGLangBaseProcessor):
    models = [Gemma3ForConditionalGeneration]

    npu_image_decode = True

    def _get_npu_image_decode_mode(self):
        mode = super()._get_npu_image_decode_mode()
        if self.server_args.device != "npu" or self.npu_image_decode is not True:
            return mode
        hf = self._processor
        if (
            type(self) is not Gemma3SGLangImageProcessor
            or getattr(self.hf_config, "model_type", None) != "gemma3"
            or type(hf).__module__ != "transformers.models.gemma3.processing_gemma3"
            or type(hf).__name__ != "Gemma3Processor"
        ):
            return mode
        img = getattr(hf, "image_processor", None)
        if (
            type(img).__module__ != "transformers.models.gemma3.image_processing_gemma3"
            or type(img).__name__ != "Gemma3ImageProcessor"
        ):
            return mode
        mean = getattr(img, "image_mean", None)
        std = getattr(img, "image_std", None)
        size = getattr(img, "size", None)
        if (
            getattr(img, "backend", None) == "torchvision"
            and getattr(size, "height", None) == 896
            and getattr(size, "width", None) == 896
            and getattr(img, "resample", None) == 2
            and getattr(img, "do_resize", None) is True
            and getattr(img, "do_normalize", None) is True
            and getattr(img, "do_rescale", None) is True
            and getattr(img, "do_convert_rgb", None) is None
            and getattr(img, "do_pan_and_scan", None) is None
            and isinstance(mean, (list, tuple))
            and tuple(mean) == (0.5, 0.5, 0.5)
            and isinstance(std, (list, tuple))
            and tuple(std) == (0.5, 0.5, 0.5)
            and getattr(img, "rescale_factor", None) == 1 / 255
            and not self.image_config
        ):
            return True
        return mode

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IM_START_TOKEN_ID = hf_config.boi_token_index
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index
        self.mm_tokens = MultimodalSpecialTokens(
            # The single, pre-expanded image token.
            image_token="<start_of_image>",
            image_token_id=hf_config.image_token_index,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<start_of_image>(?:(?:<image_soft_token>)*<end_of_image>)?"
            ),
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
        )
