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

from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor


class Gemma4UnifiedSGLangProcessor(Gemma4SGLangProcessor):
    """Multimodal processor for the encoder-free unified Gemma4 (image/video/audio).

    Identical to :class:`Gemma4SGLangProcessor` except for audio padding: the
    unified model has no SSCP conformer, so the waveform is simply chunked into
    fixed ``audio_samples_per_token`` (640) frames.  Padding the waveform up to a
    multiple of that frame size keeps ``ceil(num_samples / spt)`` consistent with
    the number of valid frames the feature extractor emits.
    """

    models = [Gemma4UnifiedForConditionalGeneration]

    def _get_audio_pad_multiple(self) -> int:
        fe = getattr(self._processor, "feature_extractor", None)
        return getattr(fe, "audio_samples_per_token", 640)
