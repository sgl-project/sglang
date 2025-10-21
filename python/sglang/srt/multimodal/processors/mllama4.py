from typing import List, Union

from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from sglang.srt.utils import get_bool_env_var

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")

class Mllama4ImageProcessor(BaseMultimodalProcessor):
    models = [Llama4ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.vision_config = hf_config.vision_config
        self.text_config = hf_config.text_config
        self.IM_START_TOKEN_ID = hf_config.boi_token_index
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index
        self.IM_TOKEN_ID = hf_config.image_token_index
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.image_token,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Process the prompt and images
        if SGL_USE_CUDA_IPC:
            async with self._cache_lock:
                mm_items, input_ids, _ = self.process_and_combine_mm_data(
                    base_output, self.mm_tokens
                )
        else:
            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.IM_TOKEN_ID,
        }
