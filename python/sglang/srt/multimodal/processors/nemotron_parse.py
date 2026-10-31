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

import asyncio

import torch

from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.nemotron_parse import (
    NECK_COMPRESSION,
    RadioMBartPrefixForCausalLM,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class NemotronParseProcessor(BaseMultimodalProcessor):
    models = [RadioMBartPrefixForCausalLM]
    gpu_image_decode = False
    generates_input_ids_from_raw_prompt = True

    async def process_mm_data_async(
        self, image_data, input_text, audio_data=None, request_obj=None, **kwargs
    ):
        if audio_data or (request_obj is not None and request_obj.video_data):
            raise ValueError("Nemotron Parse supports document images only")
        if not image_data or len(image_data) != 1:
            raise ValueError("Nemotron Parse requires one document image per request")
        base = await self.load_mm_data(
            prompt="<pad>",
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token="<pad>", image_token_id=self.hf_config.image_token_index
            ).build(self._processor),
        )
        image_inputs = await asyncio.get_running_loop().run_in_executor(
            self.io_executor,
            lambda: self._processor.image_processor(base.images, return_tensors="pt"),
        )
        image_size = image_inputs["image_sizes"][0].tolist()
        content_size = image_inputs["content_sizes"][0].tolist()
        patch = self.hf_config.encoder.patch_size
        num_tokens = (image_size[0] // patch) * (
            image_size[1] // (patch * NECK_COMPRESSION)
        ) + 1
        if isinstance(input_text, str):
            input_ids = self._tokenizer.encode(input_text, add_special_tokens=False)
        else:
            input_ids = list(input_text)
        image_token = self.hf_config.image_token_index
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=image_inputs["pixel_values"],
            # Position interpolation depends on geometry, even for identical patches.
            hash=hash_feature(
                [image_inputs["pixel_values"], torch.tensor([image_size, content_size])]
            ),
            offsets=[(0, num_tokens - 1)],
            model_specific_data={
                "image_size": image_size,
                "content_size": content_size,
            },
        )
        return MultimodalProcessorOutput(
            input_ids=[image_token] * num_tokens + input_ids,
            mm_items=[item],
            im_token_id=image_token,
        )
