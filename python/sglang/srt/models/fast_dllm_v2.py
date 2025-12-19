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
"""
NVIDIA Fast-dLLM v2 model for SGLang.

This model wrapper registers the Fast_dLLM_V2_QwenForCausalLM architecture
so SGLang can load Fast_dLLM_v2 models. The model inherits from Qwen2ForCausalLM
since Fast_dLLM_v2 is based on Qwen2.5-7B-Instruct.

Reference: https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
"""

import logging
from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


class FastDLLMV2(Qwen2ForCausalLM):
    """
    Fast-dLLM v2 model for SGLang.

    This model is based on Qwen2 architecture with block diffusion capabilities.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # dLLM algorithms require full vocabulary logits in
        # dllm_extend mode. Qwen2ForCausalLM constructs LogitsProcessor with
        # return_full_logits=False by default, so enable it for Fast_dLLM_v2.
        self.logits_processor.return_full_logits = True


# Register with HuggingFace architecture name
FastDLLMV2.__name__ = "Fast_dLLM_QwenForCausalLM"

EntryClass = FastDLLMV2
