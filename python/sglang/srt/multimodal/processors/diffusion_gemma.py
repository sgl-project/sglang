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

from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

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
        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=hf_config.image_token_id,
        ).build(_processor)
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE

    @staticmethod
    def _ensure_rgb(im):
        # The vision patch embedder expects 3-channel (768 = 3*16*16) patches, but
        # grayscale inputs can reach the image processor as a 1-channel array that
        # skips do_convert_rgb. Coerce every image to 3-channel HWC here. Inputs may
        # be PIL, numpy, or a (possibly CUDA, possibly CHW) torch tensor.
        if isinstance(im, Image.Image):
            return im if im.mode == "RGB" else im.convert("RGB")
        if hasattr(im, "detach"):
            im = im.detach().to("cpu")
        arr = np.squeeze(np.asarray(im))
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def process_mm_data(self, input_text, images=None, **kwargs):
        if images:
            images = [self._ensure_rgb(im) for im in images]
        return super().process_mm_data(input_text, images=images, **kwargs)

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
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
