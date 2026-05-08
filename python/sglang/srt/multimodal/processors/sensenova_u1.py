# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.sensenova_u1 import NEOChatModel
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class SenseNovaU1MultimodalProcessor(BaseMultimodalProcessor):
    models = [NEOChatModel]
    gpu_image_decode = False

    async def process_mm_data_async(
        self,
        image_data=None,
        audio_data=None,
        input_text: str | list[int] = "",
        request_obj=None,
        **kwargs: Any,
    ) -> MultimodalProcessorOutput:
        del request_obj, kwargs
        if image_data or audio_data:
            raise ValueError("SenseNova U1 multimodal generation uses omni requests")
        if isinstance(input_text, list):
            input_ids = input_text
        else:
            input_ids = self._tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten().tolist()
        return MultimodalProcessorOutput(mm_items=[], input_ids=input_ids)
