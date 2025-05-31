from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import ImageProcessingMixin
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TextInput

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.vila import VILAForConditionalGeneration
from sglang.srt.server_args import ServerArgs

##### BEGIN: Stub for remote code. #####


class VILAProcessorKwargsHF(ProcessingKwargs, total=False):
    _defaults = {}  # type: ignore


class VILAProcessorOutputHF(BatchFeature):
    input_ids: List[List[int]] | Tensor
    attention_mask: List[List[int]] | Tensor
    pixel_values: Optional[List[Tensor] | Tensor]


class VILAProcessorHF(ProcessorMixin):
    # Attributes.
    image_processor: ImageProcessingMixin
    tokenizer: PreTrainedTokenizerBase

    # Configuration parameters.
    image_pad_len: int
    max_tiles: int
    min_tiles: int

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[TextInput | List[TextInput]] = None,
        audio: None = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[VILAProcessorKwargsHF],
    ) -> VILAProcessorOutputHF:
        raise NotImplementedError

    def _pad_image_tokens_by_num_crops(
        self,
        text: List[TextInput],
        *,
        num_cropped_images: List[int],
    ) -> List[TextInput]:
        raise NotImplementedError

    def _pad_image_tokens_by_num_embeddings(
        self,
        text: List[TextInput],
    ) -> List[TextInput]:
        raise NotImplementedError

    def _process_images(
        self,
        images: ImageInput,
        **kwargs: Unpack[VILAProcessorKwargsHF],
    ) -> Tuple[BatchFeature, List[int]]:
        raise NotImplementedError


##### END: Stub for remote code. #####


class VILAProcessor(BaseMultimodalProcessor):
    models: List[Type[nn.Module]] = [VILAForConditionalGeneration]

    _processor: VILAProcessorHF

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        _processor: VILAProcessorHF,
    ) -> None:
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[Image, str]]],
        input_text: Union[str, List[int]],
        request_obj: GenerateReqInput,
        max_req_input_len: int,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        # Skip multimodal data processing if not provided.
        if image_data is None or image_data == []:
            return None

        base_mm_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=cast(str, self._processor.tokenizer.image_token)
            ),
            max_req_input_len=max_req_input_len,
            image_data=image_data,
        )

        inputs = self.process_mm_data(
            input_text=base_mm_output.input_text,
            images=base_mm_output.images,
        )

        mm_items: List[MultimodalDataItem] = (
            [
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    pixel_values=inputs.pixel_values,
                )
            ]
            if "pixel_values" in inputs
            else []
        )

        # Checkout python/sglang/srt/managers/schedule_batch.py:MultimodalInputs
        # and python/sglang/srt/managers/tokenizer_manager.py:TokenizerManager._tokenize_one_request()
        return dict(
            input_ids=cast(Tensor, inputs.input_ids)[0].tolist(),
            mm_items=mm_items,
        )
