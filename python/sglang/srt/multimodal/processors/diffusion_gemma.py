# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
from typing import Dict, List, Optional, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaForBlockDiffusion
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class DiffusionGemmaSGLangProcessor(SGLangBaseProcessor):
    """Image multimodal processor for DiffusionGemma (image-only, reuses the stock
    Gemma4 image processor resolved from the checkpoint)."""

    models = [DiffusionGemmaForBlockDiffusion]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.image_token_id,
            # Also match the HF-expanded placeholder (boi + soft tokens + eoi).
            image_token_regex=re.compile(
                r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
            ),
        ).build(_processor)
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        if kwargs.get("audio_data") or getattr(request_obj, "video_data", None):
            raise ValueError(
                "DiffusionGemma serving supports image input only "
                "(audio/video are not implemented)."
            )
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
