from typing import List, Optional, Tuple, Type, TypedDict, Union, Unpack, cast

import numpy as np
import torch
import torch.nn as nn
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from torch import Tensor
from transformers import (
    BatchFeature,
    PretrainedConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
    TensorType,
)
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import TextInput

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.models.vila import VILAForConditionalGeneration
from sglang.srt.server_args import ServerArgs


class VILAProcessorOutput(BatchFeature):
    """Multimodal input.

    Refer to python/sglang/srt/managers/schedule_batch.py:MultimodalInputs
    """

    input_ids: List[int]

    pixel_values: torch.Tensor
    data_hashes: Optional[List[int]]


class VILAProcessor(BaseMultimodalProcessor):
    models: List[Type[nn.Module]] = [VILAForConditionalGeneration]

    _processor: "_VILAProcessorHF"

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        _processor: "_VILAProcessorHF",
    ) -> None:
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(  # type: ignore[override]
        self,
        image_data: Optional[Union[List[str], str]],
        input_text: Optional[Union[List[str], str, List[List[int]], List[int]]],
        obj: GenerateReqInput,
        max_req_input_len: int,
        **kwargs,
    ) -> Optional["VILAProcessorOutput"]:
        assert input_text is not None, "input_text cannot be None."
        assert isinstance(input_text, str) or (
            isinstance(input_text, list) and all(isinstance(x, int) for x in input_text)
        ), "input_text must be a string or a list of integers as input_ids. Batch input is not supported."
        input_text = cast(Union[str, List[int]], input_text)

        # Handle empty image input.
        if image_data is None:
            return None

        # Process images.
        # TODO: Use self.load_mm_data
        if isinstance(image_data, str):
            image_data = [image_data]

        images = [image_utils.load_image(image_file) for image_file in image_data]

        image_inputs, num_cropped_images = self._processor._process_images(
            images,
            return_tensors=TensorType.PYTORCH,
        )

        # Process text.
        if not isinstance(input_text, str):
            input_text = self._processor.tokenizer.decode(input_text)

        input_text = self._processor._pad_image_tokens_by_num_crops(
            [input_text],
            num_cropped_images=num_cropped_images,
        )[0]

        input_text = self._processor._pad_image_tokens_by_num_embeddings(
            [input_text],
        )[0]

        input_ids: List[int] = self._processor.tokenizer.__call__(
            text=[input_text],
        ).input_ids[0]

        return VILAProcessorOutput(
            data=dict(
                input_ids=input_ids,
                pixel_values=image_inputs.pixel_values,
                data_hashes=[hash(image_file) for image_file in image_data],
            )
        )


class _VILAProcessorKwargsHF(ProcessingKwargs, total=False):
    _defaults = {}  # type: ignore


class _VILAProcessorOutputHF(BatchFeature):
    input_ids: List[List[int]] | NDArray[np.int64] | Tensor
    attention_mask: List[List[int]] | NDArray[np.int64] | Tensor
    pixel_values: Optional[List[NDArray[np.float32]] | NDArray[np.float32] | Tensor]


class _VILAProcessorHF(ProcessorMixin):
    tokenizer: PreTrainedTokenizer

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[TextInput | List[TextInput]] = None,
        audio: None = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[_VILAProcessorKwargsHF],
    ) -> _VILAProcessorOutputHF:
        raise NotImplementedError

    def _process_images(
        self,
        images: ImageInput,
        **kwargs: Unpack[_VILAProcessorKwargsHF],
    ) -> Tuple[BatchFeature, List[int]]:
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
