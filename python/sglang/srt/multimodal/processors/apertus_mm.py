# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The SwissAI Initiative
"""Multimodal request processing for Apertus 1.5."""

from typing import Dict, List, Optional, Union

import torch

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.apertus_mm import Apertus1p5ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

_DEFAULT_IMAGE_TOKEN_ID = 131079
_DEFAULT_AUDIO_TOKEN_ID = 131085


class Apertus1p5SGLangProcessor(SGLangBaseProcessor):
    """Use the HF Apertus processor's exact discrete-token prompt expansion."""

    models = [Apertus1p5ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=getattr(
                hf_config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
            ),
            audio_token_id=getattr(
                hf_config, "audio_token_id", _DEFAULT_AUDIO_TOKEN_ID
            ),
        ).build(_processor)

    def process_mm_data(
        self,
        input_text: str,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        del videos
        processor = processor or self._processor
        outputs = processor(
            text=input_text,
            images=images,
            audio=audios,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        result: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            "input_ids": outputs["input_ids"]
        }

        if images:
            result["pixel_values"] = [
                image[:, : int(height), : int(width)].contiguous()
                for image, (height, width) in zip(
                    outputs["pixel_values"], outputs["image_sizes"]
                )
            ]
        if audios:
            result["input_features"] = [
                audio[0, : int(mask.sum())].contiguous()
                for audio, mask in zip(
                    outputs["input_features"], outputs["feature_attention_mask"]
                )
            ]
        return result

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
