# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""SGLang multimodal processor for Cohere2Vision (Command-A-Vision)."""

from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.cohere2_vision import Cohere2VisionForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class Cohere2VisionSGLangImageProcessor(SGLangBaseProcessor):
    models = [Cohere2VisionForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Cohere2Vision wraps each image as:
        #   <|START_OF_IMG|> [<|IMG_PATCH|> * P^2 + <|IMG_LINE_BREAK|>] * N <|END_OF_IMG|>
        # (N = patch count, P = patch_size). The HF processor expands the single
        # <|IMG_PATCH|> placeholder into that block.
        proc = _processor
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

        # Match the unexpanded <|IMG_PATCH|> placeholder so SGLang pairs each
        # one with its image_data entry before the HF processor expands it.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            image_token_id=self.image_token_id,
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
            discard_alpha_channel=True,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            im_start_id=self.boi_token_id,
            im_end_id=self.eoi_token_id,
        )
