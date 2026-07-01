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

import logging
from typing import Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_nextn import (
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class JoyAIDenseNextNDecoderLayer(DeepseekV2DecoderLayer):

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if is_nextn:
            return False
        return super()._is_layer_sparse(layer_id, is_nextn)


class JoyAILLMFlashForCausalLMNextN(DeepseekV3ForCausalLMNextN):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        layer_prefix = self._get_nextn_layer_prefix(config, prefix)
        self.model.decoder = JoyAIDenseNextNDecoderLayer(
            config,
            0,
            quant_config=self.model.quant_config,
            is_nextn=True,
            prefix=layer_prefix,
            alt_stream=self.model.alt_stream,
        )

    @staticmethod
    def _get_nextn_layer_prefix(config, prefix):
        return add_prefix("decoder", add_prefix("model", prefix))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights)


EntryClass = [JoyAILLMFlashForCausalLMNextN]
