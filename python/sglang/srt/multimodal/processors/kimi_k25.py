import re
from typing import Dict, List, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLProcessor(SGLangBaseProcessor):
    """Kimi K2.5 multimodal processor supporting both images and videos."""

    models = [KimiK25ForConditionalGeneration]
    gpu_image_decode = False  # KimiK2.5VL HF processor does not support tensor inputs

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            video_token="<|media_pad|>",  # Kimi K2.5 uses same token for video
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            video_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
            video_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
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
            video_data=getattr(request_obj, "video_data", []) or [],
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
        }

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.
        Supports both images and videos.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        images = images or []
        videos = videos or []

        parts = input_text.split(self.mm_tokens.image_token)
        result = [parts[0]]

        # Calculate token counts for images
        for image, part in zip(images, parts[1:]):
            num_tokens = self._processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            result.append(self.mm_tokens.image_token * num_tokens + part)

        # Append remaining parts (for videos or text-only segments)
        if len(parts) > len(images) + 1:
            for video, part in zip(videos, parts[len(images) + 1 :]):
                num_tokens = self._processor.media_processor.media_tokens_calculator(
                    {"type": "video", "video": video}
                )
                result.append(self.mm_tokens.image_token * num_tokens + part)

            remaining_start = len(images) + 1 + len(videos)
            if len(parts) > remaining_start:
                result.extend(parts[remaining_start:])

        input_text = "".join(result)

        # Prepare media data for HF processor
        mediums = []
        for image in images:
            mediums.append({"type": "image", "image": image})
        for video in videos:
            mediums.append({"type": "video", "video": video})

        if mediums:
            key = "_medias"[1:]  # bypass lint
            kwargs[key] = mediums
            images = None
            videos = None

        ret = self.process_mm_data(
            input_text=input_text,
            images=images,
            audios=audios,
            videos=videos,
            **kwargs,
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret


# Backward compatibility alias
KimiK2_5VLImageProcessor = KimiK2_5VLProcessor
