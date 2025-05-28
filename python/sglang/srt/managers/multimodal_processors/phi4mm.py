import logging
from typing import List, Union

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.phi4mm import Phi4MMForCausalLM

logger = logging.getLogger(__name__)

_IMAGE_SPECIAL_TOKEN = "<|endoftext10|>"
_IMAGE_SPECIAL_TOKEN_ID = 200010


class Phi4MMImageProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.multimodal_tokens = MultimodalSpecialTokens(
            image_token=_IMAGE_SPECIAL_TOKEN,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        audio_data = request_obj.audio_data

        if not image_data and not audio_data:
            return None

        if not isinstance(image_data, list):
            image_data = [image_data]

        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        if audio_data:
            logger.warning(
                "Currently SGLang does not support audio data for Phi4MM. We are working on it. You can file an issue to help us prioritize."
            )
            audio_data = []

        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.multimodal_tokens,
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        input_ids = res["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=_IMAGE_SPECIAL_TOKEN_ID,
        )

        items = [
            MultimodalDataItem(
                pixel_values=res["input_image_embeds"],
                image_sizes=res["image_sizes"],
                image_emb_mask=res["image_attention_mask"],
                image_offsets=image_offsets,
                modality=Modality.IMAGE,
            )
        ]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "im_token_id": _IMAGE_SPECIAL_TOKEN_ID,
        }
