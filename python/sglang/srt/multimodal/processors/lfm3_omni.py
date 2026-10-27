# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Expand interleaved audio, image, and video inputs in their original order."""

import asyncio
from typing import ClassVar

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.lfm3_omni import Lfm3OmniForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.lfm3_audio import Lfm3AudioMultimodalProcessor
from sglang.srt.multimodal.processors.lfm3_vl import Lfm3VlImageProcessor


class Lfm3OmniMultimodalProcessor(Lfm3AudioMultimodalProcessor):
    gpu_image_decode = False
    gpu_video_decode = False
    video_preprocessing_device = "cpu"
    preserve_processor_input_ids = True

    models: ClassVar = [Lfm3OmniForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.image_processor = _processor.image_processor
        self.video_processor = _processor.video_processor
        self.vision_start_token = _processor.vision_start_token
        self.vision_end_token = _processor.vision_end_token
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = self._tokenizer.convert_tokens_to_ids(
            _processor.video_token
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        return {
            **super().preprocess_fingerprint_payload(),
            "lfm3_omni_vision_markers": self._processor.vision_markers,
            "lfm3_omni_audio_markers": self._processor.audio_markers,
            "lfm3_omni_vision_start_token": self.vision_start_token,
            "lfm3_omni_vision_end_token": self.vision_end_token,
            "lfm3_omni_image_processor": self.image_processor.to_dict(),
            "lfm3_omni_video_processor": (
                self.video_processor.to_dict() if self.video_processor else None
            ),
        }

    def process_mm_data(self, input_text, audios=None, **kwargs):
        return super().process_mm_data(
            input_text,
            audio=audios or None,
            sampling_rate=self.feature_extractor.sampling_rate,
            **kwargs,
        )

    def collect_mm_items_from_processor_output(self, data_dict, modality=None):
        features = data_dict.get("input_features")
        if features is None:
            return super().collect_mm_items_from_processor_output(data_dict, modality)
        items = super().collect_mm_items_from_processor_output(
            {
                key: value
                for key, value in data_dict.items()
                if key not in ("input_features", "feature_attention_mask")
            },
            modality,
        )
        lengths = data_dict["feature_attention_mask"].sum(-1).tolist()
        counts = [self.feature_extractor.num_encoder_tokens(int(n)) for n in lengths]
        positions = (
            (data_dict["input_ids"].flatten() == self.audio_token_id)
            .nonzero()
            .flatten()
        )
        # One clip may occupy several runs interleaved with video frame groups.
        for feature, length, clip_positions in zip(
            features, lengths, positions.split(counts), strict=True
        ):
            breaks = (clip_positions[1:] != clip_positions[:-1] + 1).nonzero().flatten()
            starts = torch.cat((clip_positions[:1], clip_positions[breaks + 1]))
            ends = torch.cat((clip_positions[breaks], clip_positions[-1:]))
            items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=feature[:, : int(length)].unsqueeze(0),
                    offsets=list(zip(starts.tolist(), ends.tolist())),
                )
            )
        return items

    def _process_media(self, base, use_audio_in_video):
        video_handles = base.videos
        processor_kwargs = {}
        if base.videos:
            base.videos, metadata = Lfm3VlImageProcessor._decode_videos(
                base.videos, self.video_processor
            )
            processor_kwargs["videos_kwargs"] = {
                "video_metadata": metadata,
                "do_sample_frames": False,
                "use_audio_in_video": use_audio_in_video,
            }
        if use_audio_in_video:
            from torchcodec.decoders import AudioDecoder, VideoDecoder

            audio = iter(base.audios)
            videos = iter(zip(video_handles, metadata, strict=True))
            clips, pieces = [], []
            prompt = base.input_text.replace(
                self.vision_start_token
                + self.mm_tokens.video_token
                + self.vision_end_token,
                self.mm_tokens.video_token,
            )
            rate = self.feature_extractor.sampling_rate
            for piece in self.mm_tokens.combined_regex.split(prompt):
                pieces.append(piece)
                if piece == self.mm_tokens.audio_token:
                    clips.append(next(audio))
                elif piece == self.mm_tokens.video_token:
                    video, info = next(videos)
                    source = video.source_bytes
                    header = VideoDecoder(source).metadata
                    try:
                        waveform = (
                            AudioDecoder(source, sample_rate=rate, num_channels=1)
                            .get_all_samples()
                            .data[0]
                            .numpy()
                        )
                    except Exception as error:
                        raise ValueError(
                            "Could not decode the video's soundtrack"
                        ) from error
                    start = round((header.begin_stream_seconds or 0) * rate)
                    frames, fps = info.total_num_frames, info.fps
                    clips.append(waveform[start : start + round(frames / fps * rate)])
                    pieces.append(self.mm_tokens.audio_token)
            base.audios = clips
            base.input_text = "".join(pieces)
        items, input_ids, _ = self.process_and_combine_mm_data(
            base, self.mm_tokens, **processor_kwargs
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=items,
            audio_token_id=self.audio_token_id,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    async def process_mm_data_async(
        self, audio_data, input_text, image_data=None, request_obj=None, **kwargs
    ):
        video_data = getattr(request_obj, "video_data", None)
        use_audio_in_video = getattr(request_obj, "use_audio_in_video", False)
        if use_audio_in_video and not video_data:
            raise ValueError("use_audio_in_video requires a video input")
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        for token, data in (
            (self.mm_tokens.audio_token, audio_data),
            (self.mm_tokens.image_token, image_data),
            (self.mm_tokens.video_token, video_data),
        ):
            if input_text.count(token) != len(data or []):
                raise ValueError(
                    f"LFM3 Omni requires one {token} placeholder per input; check the checkpoint chat template"
                )
        base = await self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.feature_extractor.sampling_rate,
        )
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor,
            self._process_media,
            base,
            use_audio_in_video,
        )
