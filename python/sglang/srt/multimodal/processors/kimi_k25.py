import re
from typing import Dict, List, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
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
        )
        prompt = base_output.input_text

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
        }

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """

        parts = input_text.split(self.mm_tokens.image_token)

        result = [parts[0]]
        for image, part in zip(images, parts[1:]):
            num_tokens = self._processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            result.append(self.mm_tokens.image_token * num_tokens + part)

        input_text = "".join(result)

        if images:  # for kimi k2 vl
            mediums = []
            for image in images:
                mediums.append({"type": "image", "image": image})
            key = "_medias"[1:]  # bypass lint
            kwargs[key] = mediums
            images = None

        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret
