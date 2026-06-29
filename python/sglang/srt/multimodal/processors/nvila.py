from typing import Any

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.jet_vlm import JetVLMForConditionalGeneration
from sglang.srt.models.nvila import NVILAForConditionalGeneration
from sglang.srt.models.nvila_lite import NVILALiteForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

NUM_VIDEO_FRAMES = 8


def _to_numpy_frame(frame: Any) -> Any:
    """Convert a single video frame to numpy if it is a legacy decord frame.

    Decord frames expose a callable ``asnumpy()``; numpy / PIL frames do not and are
    returned unchanged. ``getattr`` + ``callable`` avoids mistaking a non-callable
    ``asnumpy`` attribute for a converter and evaluates the attribute only once.
    """
    converter = getattr(frame, "asnumpy", None)
    return converter() if callable(converter) else frame


class NVILAMultimodalProcessor(BaseMultimodalProcessor):
    models: list[type[nn.Module]] = [
        NVILAForConditionalGeneration,
        NVILALiteForConditionalGeneration,
        JetVLMForConditionalGeneration,
    ]

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        _processor: ProcessorMixin,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self._processor: ProcessorMixin

        tokenizer: PreTrainedTokenizerBase = getattr(self._processor, "tokenizer")

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=tokenizer.image_token,
            image_token_id=hf_config.image_token_id,
            video_token=tokenizer.video_token,
            video_token_id=hf_config.video_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj: GenerateReqInput,
        **kwargs,
    ) -> dict[str, Any] | None:
        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=request_obj.image_data,  # type: ignore
            video_data=request_obj.video_data,  # type: ignore
        )

        for i, video in enumerate(base_output.videos):  # type: ignore
            # `load_video` yields a VideoDecoderWrapper (decoded file/URL/bytes) or a
            # caller-supplied list/tuple of frames; materialize only those to numpy frames.
            # Legacy decord frames expose a callable `.asnumpy()`; numpy frames pass through
            # `_to_numpy_frame` unchanged. The old `[x.asnumpy() for x in video]` assumed
            # decord and raised AttributeError on the wrapper's numpy frames. The positive
            # check leaves everything else untouched -- a whole-clip ndarray/torch.Tensor, a
            # preprocessed dict, or a `None` placeholder -- since iterating those would slice
            # an array into rows, reduce a dict to its keys, or raise on `None`.
            if isinstance(video, (VideoDecoderWrapper, list, tuple)):
                base_output.videos[i] = [  # type: ignore
                    _to_numpy_frame(frame) for frame in video
                ]

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
            do_sample_frames=True,
            num_frames=NUM_VIDEO_FRAMES,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
        )
