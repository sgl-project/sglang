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
Fast-dLLM v2 model for SGLang.

This model wrapper registers the Fast_dLLM_QwenForCausalLM architecture
so SGLang can load Fast_dLLM_v2 models. The model inherits from Qwen2ForCausalLM
since Fast_dLLM_v2 is based on Qwen2.5-7B-Instruct.

Reference: https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
"""

import logging
from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


class FastDLLMForCausalLM(Qwen2ForCausalLM):
    """
    Fast-dLLM v2 model for SGLang.

    This model is based on Qwen2 architecture with block diffusion capabilities.
    Currently runs in standard auto-regressive mode.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        # Fast_dLLM specific config
        self.bd_size = getattr(config, "bd_size", 32)
        self.mask_token_id = getattr(config, "mask_token_id", 151665)

        logger.info(
            f"FastDLLM initialized: bd_size={self.bd_size}, "
            f"mask_token_id={self.mask_token_id}"
        )


# Register with HuggingFace architecture name
FastDLLMForCausalLM.__name__ = "Fast_dLLM_QwenForCausalLM"

EntryClass = FastDLLMForCausalLM
