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
"""Multimodal processor for DeepSeek-V4-Flash-Vision.

The prompt arrives already tokenized -- DeepSeek-V4 renders chat through
``entrypoints/openai/encoding_dsv4.py``, not a Jinja template -- with one
``<|deepseek_image|>`` token per image. Each of those expands into that image's
block of placeholder tokens, whose length depends both on the image's solved
grid and on where the block starts (it is aligned to ``COMPRESS_PAD_TO``), so
the expansion walks the token list left to right.

Every slot of the block is a placeholder, framing and padding included: the
model fills the whole block from :class:`DeepseekV4VisionModel`, which writes
its learned ``image_start`` / ``image_pad`` / ``image_newline`` / ``image_end``
embeddings into the non-content slots.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4_vl import DeepseekV4VLForCausalLM
from sglang.srt.multimodal.deepseek_v4_vl_image_processing import (
    DeepseekV4VisionParams,
    build_image_block,
    preprocess_image,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<｜deepseek_image｜>"


class DeepseekV4VLImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4VLForCausalLM]
    prefer_tokenized_input = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.vision_params = DeepseekV4VisionParams.from_hf_config(hf_config)
        image_token_id = self._tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if image_token_id is None or image_token_id == self._tokenizer.unk_token_id:
            raise ValueError(
                f"{IMAGE_TOKEN} is missing from the tokenizer of "
                f"{server_args.model_path}; this is not a DeepSeek-V4 vision "
                "checkpoint."
            )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_TOKEN, image_token_id=image_token_id
        ).build(_processor)

    def _preprocess_one(self, image) -> Dict[str, Any]:
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(
            image, self.vision_params
        )
        return {
            "patches": patches,
            "n_vit_h": n_vit_h,
            "n_vit_w": n_vit_w,
            "n_llm_h": n_llm_h,
            "n_llm_w": n_llm_w,
        }

    def build_items(
        self, input_ids: List[int], preprocessed: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[MultimodalDataItem]]:
        """Expand each image placeholder into its block; return ids and items."""
        image_token_id = self.mm_tokens.image_token_id
        num_placeholders = sum(1 for token in input_ids if token == image_token_id)
        if num_placeholders != len(preprocessed):
            raise ValueError(
                f"prompt carries {num_placeholders} image placeholder token(s) "
                f"but {len(preprocessed)} image(s) were provided; they must "
                "correspond one to one."
            )

        expanded: List[int] = []
        items: List[MultimodalDataItem] = []
        next_image = 0
        for token in input_ids:
            if token != image_token_id:
                expanded.append(token)
                continue
            image = preprocessed[next_image]
            next_image += 1
            # The block's own start position fixes how much leading alignment
            # padding it carries, so the expansion has to walk left to right.
            slot_types, aligner_perm, _ = build_image_block(
                image["n_llm_h"], image["n_llm_w"], len(expanded)
            )
            start = len(expanded)
            expanded.extend([image_token_id] * len(slot_types))
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=image["patches"],
                    offsets=[(start, len(expanded) - 1)],
                    model_specific_data={
                        "n_vit_h": image["n_vit_h"],
                        "n_vit_w": image["n_vit_w"],
                        "slot_types": slot_types,
                        "aligner_perm": aligner_perm,
                    },
                )
            )
        return expanded, items

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        if request_obj.video_data or kwargs.get("audio_data"):
            raise ValueError("DeepSeek-V4-Flash-Vision supports image input only")

        input_ids = input_text
        if not isinstance(input_ids, list):
            input_ids = self._tokenizer.encode(input_ids)

        base_output = await self.fast_load_mm_data(
            prompt="",
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=False,
            input_ids=input_ids,
        )

        loop = asyncio.get_running_loop()
        preprocessed = await asyncio.gather(
            *(
                loop.run_in_executor(self.io_executor, self._preprocess_one, image)
                for image in base_output.images
            )
        )

        expanded_ids, items = self.build_items(list(input_ids), list(preprocessed))
        return MultimodalProcessorOutput(
            input_ids=expanded_ids,
            mm_items=items,
            im_token_id=self.mm_tokens.image_token_id,
        )
