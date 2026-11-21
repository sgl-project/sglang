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
"""Image preprocessing for the Nemotron 3.5 reranker."""

from sglang.srt.models.mistral3_reranker import Mistral3ForSequenceClassification
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor


class Nemotron35RerankerProcessor(PixtralProcessor):
    models = [Mistral3ForSequenceClassification]

    def _call_processor(self, processor, input_text, processor_device, kwargs) -> dict:
        result = super()._call_processor(
            processor, input_text, processor_device, kwargs
        )
        # Padding must not participate in bidirectional attention or mean pooling.
        mask = result["attention_mask"].bool()
        result["input_ids"] = result["input_ids"][mask].unsqueeze(0)
        result["attention_mask"] = result["attention_mask"][mask].unsqueeze(0)
        return result
