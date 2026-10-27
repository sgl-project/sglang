# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Audio input expansion for LFM3, including multiple clips and ASR requests."""

import asyncio
from typing import ClassVar

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.lfm3_audio import Lfm3AudioForConditionalGeneration
from sglang.srt.multimodal.lfm3_audio import AUDIO_TOKEN
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Lfm3AudioMultimodalProcessor(BaseMultimodalProcessor):
    models: ClassVar = [Lfm3AudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.feature_extractor = _processor.feature_extractor
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        if (
            self.feature_extractor.feature_size
            != hf_config.audio_encoder_config.num_mel_bins
        ):
            raise ValueError("Audio feature size must match the encoder mel bins")
        if self.feature_extractor.encoder_stride != 8:
            raise ValueError(
                "LFM3 audio encoder requires a feature extractor stride of 8"
            )
        self.audio_token_id = self._tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
        if self._tokenizer.encode(AUDIO_TOKEN, add_special_tokens=False) != [
            self.audio_token_id
        ]:
            raise ValueError(f"LFM3 audio tokenizer must contain {AUDIO_TOKEN}")
        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=AUDIO_TOKEN,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        return {
            **super().preprocess_fingerprint_payload(),
            "lfm3_audio_preprocessor": self.feature_extractor.to_dict(),
        }

    def _extract_audio_features(self, waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if (
            waveform.ndim != 1
            or not waveform.numel()
            or not torch.isfinite(waveform).all()
        ):
            raise ValueError("LFM3 audio expects a finite, nonempty mono waveform")
        return self.feature_extractor(
            waveform,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).input_features[0]

    def _expand_audio(self, prompt, audios):
        pieces = prompt.split(AUDIO_TOKEN)
        if len(pieces) - 1 != len(audios):
            raise ValueError(
                "LFM3 audio placeholder count must match the number of clips"
            )
        input_ids = []
        items = []
        for index, piece in enumerate(pieces):
            input_ids.extend(self._tokenizer.encode(piece, add_special_tokens=False))
            if index < len(audios):
                features = self._extract_audio_features(audios[index])
                tokens = self.feature_extractor.num_encoder_tokens(features.shape[-1])
                start = len(input_ids)
                input_ids.extend([self.audio_token_id] * tokens)
                items.append(
                    MultimodalDataItem(
                        modality=Modality.AUDIO,
                        feature=features,
                        offsets=[(start, start + tokens - 1)],
                    )
                )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=items,
            audio_token_id=self.audio_token_id,
        )

    async def process_mm_data_async(
        self, audio_data, input_text, image_data=None, request_obj=None, **kwargs
    ):
        if image_data or (request_obj is not None and request_obj.video_data):
            raise ValueError(
                "Lfm3AudioForConditionalGeneration supports audio and text inputs only"
            )
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text)
        audio_data = audio_data or []
        if (input_text or "").count(AUDIO_TOKEN) != len(audio_data):
            raise ValueError(
                "LFM3 audio placeholder count must match the number of clips"
            )
        base = await self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.feature_extractor.sampling_rate,
        )
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor,
            self._expand_audio,
            input_text,
            base.audios,
        )
