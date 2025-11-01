import asyncio
from functools import partial
from typing import Any

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.models.nvila import NVILAForConditionalGeneration
from sglang.srt.models.nvila_lite import NVILALiteForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_video_reader

NUM_VIDEO_FRAMES = 8


class NVILAMultimodalProcessor(BaseMultimodalProcessor):
    models: list[type[nn.Module]] = [
        NVILAForConditionalGeneration,
        NVILALiteForConditionalGeneration,
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

    @staticmethod
    def _process_video_task(video_file):
        vr = get_video_reader(video_file)
        return [x.asnumpy() for x in vr]

    async def preprocess_video(self, video_file_handle):
        """
        Preprocess video using VideoReader from Decord backend.

        Args:
            video_file_handle: Either a video file path or a temporary file

        Returns:
            tuple: A tuple containing processed frames and metadata
        """
        video_file = video_file_handle
        if hasattr(video_file_handle, "name"):
            video_file = video_file_handle.name

        result = None
        try:
            if self.cpu_executor is not None:
                loop = asyncio.get_event_loop()
                task = partial(NVILAMultimodalProcessor._process_video_task, video_file)
                result = await loop.run_in_executor(self.cpu_executor, task)
            else:
                result = NVILAMultimodalProcessor._process_video_task(video_file)
        finally:
            if hasattr(video_file_handle, "name"):
                video_file_handle.close()

        return result

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

        for i, (video, _) in enumerate(base_output.videos):  # type: ignore
            base_output.videos[i] = await self.preprocess_video(video)  # type: ignore

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
            do_sample_frames=True,
            num_frames=NUM_VIDEO_FRAMES,
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
        }
