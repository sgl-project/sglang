# Copyright 2025 SGLang Team
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
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.models.gemma3n_mm import Gemma3nForConditionalGeneration


class Gemma3nSGLangProcessor(SGLangBaseProcessor):
    """Multimodal processor for Gemma3n supporting image and audio inputs."""

    models = [Gemma3nForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

        self.IMAGE_TOKEN = "<image_soft_token>"
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<start_of_image>(?:(?:<image_soft_token>)*<end_of_image>)?"
        )

        self.AUDIO_TOKEN = "<audio_soft_token>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<start_of_audio>(?:(?:<audio_soft_token>)*<end_of_audio>)?"
        )

        self.IM_TOKEN_ID = hf_config.image_token_id
        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.AUDIO_TOKEN_ID = hf_config.audio_token_id
        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        max_req_input_len: int = 0,
        *args,
        **kwargs,
    ):
        """Process multimodal data including images and audio."""

        audio_data = request_obj.audio_data
        if not image_data and not audio_data:
            return None

        if isinstance(image_data, str):
            image_data = [image_data]

        if isinstance(audio_data, str):
            audio_data = [audio_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=audio_data,
            max_req_input_len=max_req_input_len,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN,
                image_token_regex=self.IMAGE_TOKEN_REGEX,
                audio_token=self.AUDIO_TOKEN,
                audio_token_regex=self.AUDIO_TOKEN_REGEX,
            ),
        )

        combined_mm_item, input_ids = self.process_and_combine_mm_data(base_output)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": [combined_mm_item] if combined_mm_item is not None else [],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "audio_start_id": self.AUDIO_START_TOKEN_ID,
            "audio_end_id": self.AUDIO_END_TOKEN_ID,
        }
