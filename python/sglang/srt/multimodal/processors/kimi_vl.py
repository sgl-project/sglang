import re
<<<<<<< HEAD
from typing import Any, Dict, List, Optional, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
=======
from typing import Dict, List, Union

>>>>>>> origin/main
from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


# Compatible with KimiVLForConditionalGeneration
class KimiVLImageProcessor(SGLangBaseProcessor):
    models = [KimiVLForConditionalGeneration]

<<<<<<< HEAD
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|media_pad|>"
        self.IMAGE_TOKEN_REGEX = re.compile(r"(?:<\|media_pad\|>)+")
        self.IM_TOKEN_ID = _processor.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
=======
    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)
>>>>>>> origin/main

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
<<<<<<< HEAD
        max_req_input_len,
=======
>>>>>>> origin/main
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
<<<<<<< HEAD
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN, image_token_regex=self.IMAGE_TOKEN_REGEX
            ),
            max_req_input_len=max_req_input_len,
        )

        mm_items, input_ids = self.process_and_combine_mm_data(base_output)
=======
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
>>>>>>> origin/main

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
<<<<<<< HEAD
            "im_token_id": self.IM_TOKEN_ID,
=======
            "im_token_id": self.mm_tokens.image_token_id,
>>>>>>> origin/main
        }
