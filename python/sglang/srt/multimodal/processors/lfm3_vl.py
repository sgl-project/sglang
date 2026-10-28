# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multimodal processor for LFM3-VL (Qwen2-VL-style dynamic resolution).

LFM3-VL's HF image/video processors are constant-only subclasses of
`Qwen2VLImageProcessor` / `Qwen3VLVideoProcessor`, so preprocessing needs no
custom code here -- the base flow plus `pixel_values` + `*_grid_thw` is enough.

Two things make this thinner than the Qwen-VL processor:

  - the chat template emits **bare** `<|image|>` / `<|video|>` placeholders. There
    are no `<|vision_start|>` / `<|vision_end|>` wrappers, no tile-position or
    thumbnail tokens (the vocabulary inherits 100 `<|img_row_R_col_C|>` tokens
    from LFM2-VL, but this model's template never emits them), so we match on the
    flat placeholder and let the HF processor expand it to
    `prod(grid_thw) // merge_size**2` repeats.
  - there is no M-RoPE, so no position bookkeeping is needed. Video timestamps
    (`<{t:.1f} seconds>`) are expanded by the HF processor as ordinary *text*
    interleaved among the video placeholders, which means the placeholder mask is
    non-contiguous; scattering by token-id equality (what
    `MultiModalityDataPaddingPatternMultimodalTokens` does) handles that.
"""

import asyncio
from typing import Any, List, Optional, Union

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.lfm3_vl import Lfm3VlForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


class Lfm3VlImageProcessor(SGLangBaseProcessor):
    models = [Lfm3VlForConditionalGeneration]
    gpu_image_decode = False
    gpu_video_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=self.IMAGE_TOKEN_ID,
            video_token="<|video|>",
            video_token_id=self.VIDEO_TOKEN_ID,
        ).build(_processor)

        # `pixel_values`, `image_grid_thw`, `pixel_values_videos` and
        # `video_grid_thw` are already in the base ATTR_NAME_TO_MODALITY map, so
        # collect_mm_items_from_processor_output routes them without extra
        # registration. The model reads them back as item.image_grid_thw /
        # item.video_grid_thw.

    @staticmethod
    def _decode_videos(
        videos: List[Any], video_processor
    ) -> tuple[List[Any], List[Any]]:
        """Use HF's frame selection and clock metadata for both VL and Omni."""
        from transformers.video_utils import read_video_torchcodec

        decoded, metadata = [], []
        for video in videos:
            if not isinstance(video, VideoDecoderWrapper):
                raise TypeError(
                    f"Expected a video decoder handle, got {type(video).__name__}"
                )
            frames, info = read_video_torchcodec(
                video.source_bytes, sample_indices_fn=video_processor.sample_frames
            )
            decoded.append(frames)
            metadata.append(info)
        return decoded, metadata

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes]]],
        audio_data,
        input_text: str,
        request_obj,
        **kwargs,
    ):
        video_data = request_obj.video_data

        if not image_data and not video_data:
            input_ids = self._tokenizer(
                input_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            return MultimodalProcessorOutput(
                input_ids=input_ids.squeeze(0).tolist(),
                mm_items=[],
                im_token_id=self.IMAGE_TOKEN_ID,
                video_token_id=self.VIDEO_TOKEN_ID,
            )

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor, self._process_media, base_output
        )

    def _process_media(self, base_output):
        processor_kwargs: dict = {}
        if base_output.videos:
            base_output.videos, video_metadata = self._decode_videos(
                base_output.videos, self._processor.video_processor
            )
            processor_kwargs["videos_kwargs"] = {
                "video_metadata": video_metadata,
                "do_sample_frames": False,
            }

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, **processor_kwargs
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.IMAGE_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )
