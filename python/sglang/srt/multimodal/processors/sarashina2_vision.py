from typing import List, Union

from sglang.srt.models.sarashina2_vision import Sarashina2VisionForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Sarashina2VisionProcessor(BaseMultimodalProcessor):
    models = [Sarashina2VisionForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Sarashina2Vision specific tokens (default is <|file|>)
        self.IMAGE_TOKEN = "<|file|>"
        self.IM_TOKEN_ID = getattr(hf_config, "image_token_index", 14)
        self.IM_START_ID = getattr(hf_config, "start_image_token_index", 102397)
        self.IM_END_ID = getattr(hf_config, "end_image_token_index", 102398)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

        # Patch the processor's image processor to handle parameter compatibility
        if hasattr(_processor, "image_processor") and hasattr(
            _processor.image_processor, "_preprocess"
        ):
            original_preprocess = _processor.image_processor._preprocess

            def patched_preprocess(*args, **kwargs):
                # Filter kwargs to only include parameters that the custom _preprocess method accepts
                # Based on Sarashina2VisionImageProcessor._preprocess signature
                allowed_params = {
                    "do_resize",
                    "resample",
                    "do_rescale",
                    "rescale_factor",
                    "do_normalize",
                    "image_mean",
                    "image_std",
                    "do_convert_rgb",
                    "data_format",
                    "input_data_format",
                }
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k in allowed_params
                }
                return original_preprocess(*args, **filtered_kwargs)

            _processor.image_processor._preprocess = patched_preprocess

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        """Process image data for Sarashina2Vision model using standard SGLang pattern."""
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.mm_tokens.image_token_id,
            "im_start_id": self.IM_START_ID,
            "im_end_id": self.IM_END_ID,
        }
