# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multimodal processor for Gemma 4 Unified (12B, encoder-free).

Patchification (image) and waveform framing (audio) are performed by the HF
processor; this wrapper registers the special tokens and processor outputs.
Unlike the 26B/31B processor there is no Conformer, so the SSCP waveform
alignment padding is intentionally omitted.
"""

from typing import Dict, List, Optional, Union

import torch

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_unified_mm import Gemma4UnifiedForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


class Gemma4UnifiedSGLangProcessor(SGLangBaseProcessor):
    """Processor for Gemma 4 Unified supporting image, video, and audio."""

    models = [Gemma4UnifiedForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id
        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = getattr(
            hf_config, "eoa_token_id", getattr(hf_config, "eoa_token_index", None)
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            audio_token_id=hf_config.audio_token_id,
        ).build(_processor)

        # Register processor outputs so they are stored on MultimodalDataItem.
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["video_position_ids"] = Modality.VIDEO

    def _video_decoder_to_tensor(self, vdw: VideoDecoderWrapper) -> torch.Tensor:
        """Uniformly sample frames and return (N, C, H, W) uint8 tensor.

        SGLang's load_video returns a VideoDecoderWrapper the HF video processor
        does not recognise, so replicate HF uniform sampling, then delegate
        resize/patchify/position-ids to the HF video processor.
        """
        total = len(vdw)
        num_frames = getattr(
            getattr(self._processor, "video_processor", None), "num_frames", 32
        )
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = torch.arange(0, total, total / num_frames).int().tolist()
        frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        # No SSCP alignment padding: the encoder-free audio path frames the raw
        # waveform directly (audio_samples_per_token), with no conformer stride.
        if videos:
            videos = [
                (
                    self._video_decoder_to_tensor(v)
                    if isinstance(v, VideoDecoderWrapper)
                    else v
                )
                for v in videos
            ]
            kwargs.setdefault("do_sample_frames", False)
        return super().process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data if request_obj else None,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
