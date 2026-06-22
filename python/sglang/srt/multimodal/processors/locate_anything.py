# SPDX-License-Identifier: Apache-2.0
import re
from typing import Dict, List, Union

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.locate_anything import LocateAnythingForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


# Compatible with LocateAnythingForConditionalGeneration
class LocateAnythingImageProcessor(SGLangBaseProcessor):
    models = [LocateAnythingForConditionalGeneration]
    # The LocateAnything HF processor is remote-code and does not support tensor inputs.
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # The model's chat template emits numbered ``<image-N>`` placeholders.
        # The HF LocateAnythingProcessor expands each into
        # ``<img>`` + N×``<IMG_CONTEXT>`` + ``</img>`` and only the
        # ``<IMG_CONTEXT>`` (id 151665) run carries vision embeddings, so the
        # offset/embedding token id is image_token_index while the prompt-level
        # placeholder we split on is ``<image-N>``.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=hf_config.image_token_index,
            image_token_regex=re.compile(r"<image-\d+>"),
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
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )
