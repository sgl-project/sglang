# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LFM3 Omni audio/image/video/text inputs using the native SGLang scheduler."""

from sglang.srt.models.lfm3_audio import Lfm3AudioForConditionalGeneration
from sglang.srt.models.lfm3_vl import Lfm3VisionFeaturesMixin
from sglang.srt.models.lfm3_vl_vision import Lfm3VlProjector, Lfm3VlVisionEncoder
from sglang.srt.utils import add_prefix


class Lfm3OmniForConditionalGeneration(
    Lfm3VisionFeaturesMixin, Lfm3AudioForConditionalGeneration
):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        self.vision_encoder = Lfm3VlVisionEncoder(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_encoder", prefix),
        )
        self.vision_projector = Lfm3VlProjector(
            config,
            quant_config=quant_config,
            prefix=add_prefix("vision_projector", prefix),
        )

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        output = super().forward(input_ids, positions, forward_batch, **kwargs)
        # Match the reference's text-only mode: this model has no speech decoder.
        logits = getattr(output, "next_token_logits", None)
        if logits is not None:
            logits[:, self.config.audio_start_token_id] = -float("inf")
            logits[:, self.config.mixed_start_token_id] = -float("inf")
        return output

    def load_weights(self, weights):
        remaining, vision_loaded = self._load_vision_weights(weights)
        return vision_loaded | super().load_weights(remaining)


EntryClass = Lfm3OmniForConditionalGeneration
