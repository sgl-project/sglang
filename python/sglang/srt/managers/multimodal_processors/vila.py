from typing import List, Optional, Tuple, Type, Union, Unpack, cast

import numpy as np
import torch
import torch.nn as nn
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from torch import Tensor
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin, TensorType
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import TextInput

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.models.vila import VILAForConditionalGeneration
from sglang.srt.server_args import ServerArgs


class VILAProcessorKwargsHF(ProcessingKwargs, total=False):
    _defaults = {}  # type: ignore


class VILAProcessorOutputHF(BatchFeature):
    input_ids: List[List[int]] | NDArray[np.int64] | Tensor
    attention_mask: List[List[int]] | NDArray[np.int64] | Tensor
    pixel_values: Optional[List[NDArray[np.float32]] | NDArray[np.float32] | Tensor]


class VILAProcessorHF(ProcessorMixin):
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


class VILAProcessorOutput(BatchFeature):
    """Multimodal input.

    Refer to python/sglang/srt/managers/schedule_batch.py:MultimodalInputs
    """

    input_ids: List[int]

    pixel_values: torch.Tensor
    data_hashes: Optional[List[int]]


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

        # If no image provided, no need to process.
        if image_data is None:
            return None

        # Normalize input data.
        image_data = [image_data] if isinstance(image_data, str) else image_data

        images = [image_utils.load_image(image_file) for image_file in image_data]

        input_text = (
            input_text
            if isinstance(input_text, str)
            else cast(
                List[str],
                self._processor.post_process_image_text_to_text(
                    [input_text], skip_special_tokens=False
                ),
            )[0]
        )

        inputs = self._processor.__call__(
            images=images,
            text=input_text,
            max_length=max_req_input_len,
            return_tensors=TensorType.PYTORCH,
            truncation=True,
        )

        return VILAProcessorOutput(
            data=dict(
                input_ids=list(inputs.input_ids[0]),
                pixel_values=inputs.pixel_values,
                data_hashes=[hash(image_file) for image_file in image_data],
            )
        )
