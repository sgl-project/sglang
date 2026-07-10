# Copyright 2023-2024 SGLang Team
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

"""Inference-only GFusion model.

GFusion reuses the DeepSeekV3/DeepSeekV2 MLA implementation and weight-loading
path unchanged; it only switches the attention layers to bidirectional
attention for dLLM decoding and returns full logits for every position.
"""

from typing import Optional

from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class GFusionForDiffusionLM(DeepseekV2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # dLLM decoding: bidirectional (non-causal) attention over each block and
        # logits for every position in the block.
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            layer.self_attn.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            layer.self_attn.attn_mha.attn_type = AttentionType.ENCODER_ONLY

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)


EntryClass = GFusionForDiffusionLM
