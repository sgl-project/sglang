# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List, Union

from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from sglang.srt.utils import get_bool_env_var

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")


class DeepseekVL2ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekVL2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs
    ):
        base_output = self.load_mm_data(
            input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
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
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self._processor.image_token_id,
        }
