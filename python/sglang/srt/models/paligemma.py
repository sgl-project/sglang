# Copyright 2023-2025 SGLang Team
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
"""PaliGemma - Google's Cutting-Edge Open Vision Language Model. """

import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    PaliGemmaConfig,
    SiglipVisionModel,
)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.utils import add_prefix



class PaliGemmaForConditionalGeneration(nn.module):
    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str ="",
    ) -> None:
        
        super().__init__()
        self.config = config
        self.languag_model = GemmaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        pass #TODO(Xiao)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs

        if forward_batch.forward_mode.is_extend():
            pass #TODO(xiao)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        vision_path = self.config.mm_vision_tower
    
        self.vision_tower = SiglipVisionModel.from_pretrained(
            vision_path, torch_dtype=torch.float16
            ).cuda()
        pass #TODO(Xiao)

EntryClass = PaliGemmaForConditionalGeneration
    
