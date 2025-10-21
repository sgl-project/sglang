from typing import List, Union

from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from sglang.srt.utils import get_bool_env_var

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")

class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self._processor.image_token,
            image_token_id=self._processor.image_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        
        if SGL_USE_CUDA_IPC:
            async with self._cache_lock:
                mm_items, input_ids, _ = self.process_and_combine_mm_data(
                    base_out, self.mm_tokens
                )
        else:
            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                base_out, self.mm_tokens
            )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.mm_tokens.image_token_id,
        }
