from typing import List, Union

from sglang.srt.models.deepseek_janus_pro import MultiModalityCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from sglang.srt.utils import get_bool_env_var

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")

class JanusProImageProcessor(BaseMultimodalProcessor):
    models = [MultiModalityCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.image_token,
            image_token_id=_processor.image_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        **kwargs,
    ):
        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        
        if SGL_USE_CUDA_IPC:
            async with self._cache_lock: 
                mm_items, input_ids, _ = self.process_and_combine_mm_data(
                    base_out, self.mm_tokens, prompt=base_out.input_text
                )
        else:
            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                base_out, self.mm_tokens, prompt=base_out.input_text
            )
        
            
        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_start_id": self._processor.image_start_id,
            "im_end_id": self._processor.image_end_id,
            "im_token_id": self.mm_tokens.image_token_id,
        }
