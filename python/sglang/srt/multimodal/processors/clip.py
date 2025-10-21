from typing import List, Union

from sglang.srt.models.clip import CLIPModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from sglang.srt.utils import get_bool_env_var

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")

class ClipImageProcessor(BaseMultimodalProcessor):
    models = [CLIPModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(image_token="<image>").build(
            _processor
        )

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )
        
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
        }
