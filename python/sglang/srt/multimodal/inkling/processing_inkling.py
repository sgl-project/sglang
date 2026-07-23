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
"""HuggingFace-convention processor for Inkling multimodal models.

Composes the image processor + audio feature extractor + (optional) tokenizer.
``tokenizer`` is optional because the text chat renderer is out of scope here.
"""

from __future__ import annotations

from typing import List, Optional

from sglang.srt.multimodal.inkling.feature_extraction import (
    InklingAudioFeatureExtractor,
)
from sglang.srt.multimodal.inkling.image_processing import InklingImageProcessor


class InklingProcessor:
    """Bundle Inkling image + audio preprocessing with the MM token ids from the config."""

    def __init__(
        self,
        image_processor: Optional[InklingImageProcessor] = None,
        audio_feature_extractor: Optional[InklingAudioFeatureExtractor] = None,
        tokenizer=None,
    ):
        self.image_processor = image_processor or InklingImageProcessor()
        self.audio_feature_extractor = (
            audio_feature_extractor or InklingAudioFeatureExtractor()
        )
        self.tokenizer = tokenizer

    def process_images(self, images: List):
        """Raw images -> BatchFeature(vision_patches_bthwc, num_patches, num_tokens)."""
        return self.image_processor.preprocess(images, return_tensors="pt")

    def process_audios(self, audios: List):
        """Raw audios -> BatchFeature(dmel_bins, num_audio_tokens)."""
        return self.audio_feature_extractor(audios)
