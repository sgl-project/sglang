from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_base import ImageProcessingMixin
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TextInput
from transformers.utils.generic import TensorType

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
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
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ],
        input_text: Optional[
            Union[
                List[str],
                str,
                List[List[int]],
                List[int],
            ]
        ],
        request_obj: GenerateReqInput,
        max_req_input_len: int,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if image_data is None or input_text is None:
            return None

        normalized_image_data = self._normalize_image_data(image_data)
        normalized_input_text = self._normalize_input_text(input_text)

        inputs = self._processor.__call__(
            images=normalized_image_data,
            text=normalized_input_text,
        )

        input_ids = reshape_input_ids_like(
            cast(List[List[int]], inputs.input_ids), input_text
        )

        mm_items: List[MultimodalDataItem] = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                pixel_values=single_pixel_values.unsqueeze(0),
            )
            for single_pixel_values in cast(Tensor, inputs.pixel_values)
        ]

        # Checkout python/sglang/srt/managers/schedule_batch.py:MultimodalInputs
        # and python/sglang/srt/managers/tokenizer_manager.py:TokenizerManager._tokenize_one_request()
        return dict(
            input_ids=input_ids,
            mm_items=mm_items,
        )

    def _normalize_image_data(
        self,
        image_data: Union[
            List[List[Union[Image, str]]],
            List[Union[Image, str]],
            Union[Image, str],
        ],
    ) -> List[Image]:
        images = image_utils.load_images(image_data)
        flat_list: List[Image] = image_utils.make_flat_list_of_images(images)  # type: ignore
        return flat_list

    def _normalize_input_text(
        self,
        input_text: Union[
            List[str],
            str,
            List[List[int]],
            List[int],
        ],
    ) -> List[str]:
        if is_list_of(input_text, str):
            return cast(List[str], input_text)
        elif isinstance(input_text, str):
            return [input_text]
        elif is_list_of(input_text, int):
            return cast(
                List[str],
                self._processor.post_process_image_text_to_text(
                    [input_text], skip_special_tokens=False
                ),
            )
        elif is_list_of_lists_of(input_text, int):
            return cast(
                List[str],
                self._processor.post_process_image_text_to_text(
                    input_text, skip_special_tokens=False
                ),
            )

        assert False, "Argument 'input_text' is not a valid type."


def is_list_of(x: Any, t: Type) -> bool:
    return isinstance(x, list) and all(isinstance(i, t) for i in x)


def is_list_of_lists_of(x: Any, t: Type) -> bool:
    return isinstance(x, list) and all(is_list_of(i, t) for i in x)


def reshape_input_ids_like(
    input_ids: List[List[int]],
    like: Union[
        List[str],
        str,
        List[List[int]],
        List[int],
    ],
) -> Union[List[int], List[List[int]]]:
    if is_list_of(like, str) or is_list_of_lists_of(like, int):
        return input_ids
    elif isinstance(like, str) or is_list_of(like, int):
        return input_ids[0]

    assert False, "Argument 'like' is not a valid type."
