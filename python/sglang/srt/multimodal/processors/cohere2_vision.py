# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""SGLang multimodal processor for Cohere2Vision (Command-A-Vision)."""

import re
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

from sglang.srt.models.cohere2_vision import Cohere2VisionForConditionalGeneration


class Cohere2VisionSGLangImageProcessor(SGLangBaseProcessor):
    models = [Cohere2VisionForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Cohere2Vision uses three special tokens around an image:
        #   <|START_OF_IMG|> [<|IMG_PATCH|> * P^2 + <|IMG_LINE_BREAK|>] * N <|END_OF_IMG|>
        # where N = number of image patches and P = patch_size.  The Hugging
        # Face processor expands the placeholder ``<|IMG_PATCH|>`` to the full
        # patch sequence; we just need to recognise the expanded form here so
        # SGLang can pad the input ids properly.
        proc = _processor
        # Resolve token strings + ids from the HF processor's tokenizer.
        boi_token = proc.boi_token
        eoi_token = proc.eoi_token
        image_token = proc.image_token  # "<|IMG_PATCH|>"
        line_break_token = proc.img_line_break_token

        self.image_token_id = proc.image_token_id
        self.boi_token_id = proc.tokenizer.convert_tokens_to_ids(boi_token)
        self.eoi_token_id = proc.tokenizer.convert_tokens_to_ids(eoi_token)
        self.img_line_break_token_id = proc.tokenizer.convert_tokens_to_ids(
            line_break_token
        )

        # Regex that matches the full expanded image block (BOI .. EOI).
        image_token_regex = re.compile(
            f"{re.escape(boi_token)}"
            f"(?:(?:{re.escape(image_token)})*{re.escape(line_break_token)})*"
            f"{re.escape(eoi_token)}"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            image_token_id=self.image_token_id,
            image_token_regex=image_token_regex,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_start_id=self.boi_token_id,
            im_end_id=self.eoi_token_id,
        )
