from typing import Any, Dict, List, Optional, Type, cast

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    ImageDataItem,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.vila import VILAForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.server_args import ServerArgs


class VILAProcessor(ProcessorMixin):
    """A stub class for the VILA processor."""

    tokenizer: PreTrainedTokenizerBase


class VILAMultimodalProcessor(BaseMultimodalProcessor):
    models: List[Type[nn.Module]] = [VILAForConditionalGeneration]

    _processor: VILAProcessor

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        _processor: VILAProcessor,
    ) -> None:
        super().__init__(hf_config, server_args, _processor)
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

    async def process_mm_data_async(
        self,
        image_data: Optional[ImageDataItem | List[ImageDataItem]],
        input_text: str | List[int],
        request_obj: GenerateReqInput | EmbeddingReqInput,
        max_req_input_len: int,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self._processor.tokenizer.image_token
            ),
            max_req_input_len=max_req_input_len,
            image_data=image_data,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_output)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.IM_TOKEN_ID,
            "video_token_id": self.VIDEO_TOKEN_ID,
        }
