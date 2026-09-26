from typing import List, Tuple, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self._processor.image_token,
            image_token_id=self._processor.image_token_id,
        ).build(_processor)

    def get_mm_item_offsets(
        self,
        input_ids: torch.Tensor,
        mm_tokens: MultimodalSpecialTokens,
        modality: Modality,
    ) -> List[Tuple[int, int]]:
        if modality == Modality.IMAGE:
            # Each marker identifies one image, including adjacent markers.
            positions = (input_ids == mm_tokens.image_token_id).nonzero(as_tuple=True)[
                0
            ]
            return [(position, position) for position in positions.tolist()]
        return super().get_mm_item_offsets(input_ids, mm_tokens, modality)

    def resolve_image_token_counts(self, images: List) -> List[int]:
        # Image features occupy the encoder prefix; each text marker stays one token.
        return [1] * len(images)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_out = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_out, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
